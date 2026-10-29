"""强度预测消融实验的统一重跑（多种折划分）。

背景与关键事实
--------------
1. **路径实验的教训**：单一折划分会给出不可靠结论。环境场要在 25 个 job
   （5 折 × 5 划分）下才显露出真实效应（−10.1 km，p=0.0005）。

2. **强度训练不存在"训练不足"问题**：配置已是 60 epoch + patience=15。
   各模型 best_epoch 仅 2-5，说明真正的问题是**过拟合**（不是训练不够）。
   因此本脚本的重点是「多种折划分」，而非「更多 epochs」。

3. 因此每个变体用 `--compare` 一次训练 4 个模型
   （fusion_transformer / feature_transformer / gru / persistence），
   除特征变体外其余配置相同，可直接对比。

三个变体
--------
  base      无附加特征（基线）
  warmcore  + 10 维暖心结构特征
  terrain   + 7 维地形特征

用法
----
    # 推荐（5 折 × 2 划分 = 10 job/变体）
    python scripts/intensity/rerun_ablations.py --device cuda

    # 更稳健（5 折 × 3 划分 = 15 job/变体，约 30 分钟）
    python scripts/intensity/rerun_ablations.py --device cuda --fold-seeds 42 0 7
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 变体定义：名称 -> (描述, 额外命令行参数)
VARIANTS = {
    "base":     ("无附加特征", []),
    "warmcore": ("+暖心特征(10维)", ["--warmcore"]),
    "terrain":  ("+地形特征(7维)", ["--terrain", "__TERRAIN__"]),
}


def out_dir() -> str:
    import yaml
    c = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    return str(c["output"]["dir"])


def main() -> int:
    ap = argparse.ArgumentParser(description="强度消融统一重跑")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seeds", nargs="*", type=int, default=[42, 0])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--terrain", default=".scratch/env_terrain.parquet")
    ap.add_argument("--out", default="intensity_rerun")
    args = ap.parse_args()

    od = out_dir()              # 配置里的 output.dir（通常就是 "outputs"）
    out_root = ROOT / od        # 不要再拼 "outputs"，否则变成 outputs/outputs
    out_root.mkdir(parents=True, exist_ok=True)
    n_jobs = args.folds * len(args.fold_seeds)

    print("=" * 74)
    print("强度消融统一重跑")
    print("=" * 74)
    print(f"  折数      : {args.folds}")
    print(f"  划分 seed : {args.fold_seeds} ({len(args.fold_seeds)} 种)")
    print(f"  job/变体  : {n_jobs}")
    print(f"  变体      : {args.variants}")
    print(f"  每个 job 训练 4 个模型 -> 共 {n_jobs * 4 * len(args.variants)} 次训练")
    print("=" * 74, flush=True)

    summary = {}
    for name in args.variants:
        desc, extra = VARIANTS[name]
        run_name = f"{args.out}_{name}"
        cmd = [
            sys.executable, "-u",
            str(ROOT / "scripts" / "intensity" / "train_intensity.py"),
            "--compare",
            "--cv", str(args.folds),
            "--fold-seeds", *[str(x) for x in args.fold_seeds],
            "--set", f"device={args.device}", f"satellite.run_name={run_name}",
        ]
        for e in extra:
            cmd.append(str(ROOT / args.terrain) if e == "__TERRAIN__" else e)

        print(f"\n{'=' * 74}\n变体 {name}: {desc}\n{'=' * 74}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env={**os.environ, "MPLCONFIGDIR": "/tmp/mpl-cache"})
        el = time.time() - t0
        if proc.returncode != 0:
            print(f"  [失败] 退出码 {proc.returncode}", flush=True)
            print(proc.stderr[-800:], flush=True)
            summary[name] = {"ok": False, "error": proc.stderr[-500:]}
            continue
        res_path = out_root / run_name / f"cv{args.folds}_summary.json"
        if not res_path.exists():
            print(f"  [警告] 找不到结果文件 {res_path}", flush=True)
            summary[name] = {"ok": False, "error": "no result"}
            continue
        d = json.loads(res_path.read_text())
        summary[name] = {"ok": True, "seconds": round(el, 1), "desc": desc,
                         "summary": d["summary"]}
        print(f"  [完成] {el:.0f}s", flush=True)
        for tag, st in d["summary"].items():
            print(f"    {tag:28s} {st['overall_mean']:.3f}±{st['overall_std']:.3f}",
                  flush=True)

    (out_root / f"{args.out}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---------- 对比表 ----------
    print("\n" + "=" * 74)
    print("跨变体对比（5折×多划分，测试集 MAE m/s）")
    print("=" * 74)
    # 实际标签形如 "fusion_transformer+云图"，用前缀匹配
    order = ["gru", "persistence", "fusion_transformer", "feature_transformer"]
    all_keys = set()
    for v in args.variants:
        all_keys |= set(summary.get(v, {}).get("summary", {}).keys())
    heads = [k for pref in order for k in sorted(all_keys) if k.startswith(pref)]
    if not heads:
        print("无可用结果")
        return 1
    print(f"{'变体':<12} " + " ".join(f"{h[:14]:>16}" for h in heads))
    print("-" * 74)
    for name in args.variants:
        v = summary.get(name, {})
        if not v.get("ok"):
            print(f"{name:<12} (失败)")
            continue
        cells = []
        for h in heads:
            st = v["summary"].get(h)
            cells.append(f"{st['overall_mean']:>8.3f}±{st['overall_std']:.3f}"
                         if st else f"{'—':>16}")
        print(f"{name:<12} " + " ".join(f"{c:>16}" for c in cells))

    # 相对 base 的变化
    base = summary.get("base", {})
    if base.get("ok"):
        print("\n相对 base 的变化：")
        for name in args.variants:
            v = summary.get(name, {})
            if name == "base" or not v.get("ok"):
                continue
            for h in heads:
                b = base["summary"].get(h)
                c = v["summary"].get(h)
                if b and c:
                    d = c["overall_mean"] - b["overall_mean"]
                    pct = d / b["overall_mean"] * 100
                    mark = "更好" if d < 0 else "更差"
                    print(f"  {name:<10} {h:<22} {d:+7.3f} ({pct:+6.2f}%) {mark}")
    print(f"\n汇总: {out_root / f'{args.out}_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
