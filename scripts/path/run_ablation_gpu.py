"""多 seed 消融实验的 GPU 批量执行器。

相比逐个 seed 串行跑，本脚本把「配置 × seed」拆成独立进程：
  * GPU 足够大时可同时跑多个（每进程占用约 100-300MB 显存）
  * CPU 上也能并行，但受核心数限制，建议 GPU 使用
  * 每个进程独立，互不影响；某个失败不影响其他

用法:
    # 推荐：GPU 上 3 个进程并行，跑 5 个 seed
    python scripts/path/run_ablation_gpu.py --env .scratch/env_era5.parquet \
        --seeds 0 1 2 3 4 --workers 3 --device cuda

    # CPU 上（16 核可开 4 个）
    python scripts/path/run_ablation_gpu.py --env .scratch/env_era5.parquet \
        --seeds 0 1 2 3 4 --workers 4 --device cpu

输出:
    outputs/env_ablation/ablation.json   （与 ablate_environment.py 格式一致）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIGS = ["baseline(仅轨迹)", "+环境场", "仅环境场"]


def run_single(seed: int, env: str, device: str, epochs: int,
               out_root: Path, threads: int = 0) -> dict:
    """跑一个 seed（三配置），结果写到独立目录。"""
    run_name = f"env_ablation/seed{seed}"   # 每个 seed 独立目录，避免覆盖
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "path" / "ablate_environment.py"),
        "--env", env,
        "--seeds", str(seed),
        "--set", f"train.epochs={epochs}", f"device={device}",
        f"output.dir={out_root}",
        f"output.run_name={run_name}",
    ]
    env_vars = {"MPLCONFIGDIR": "/tmp/mpl-cache"}
    if threads > 0:
        # 限制每个进程的线程数，避免 CPU 并行时互相争抢
        env_vars["OMP_NUM_THREADS"] = str(threads)
        env_vars["MKL_NUM_THREADS"] = str(threads)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env_vars})
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"seed": seed, "ok": False, "error": proc.stderr[-800:], "sec": elapsed}
    # 读取结果：ablate 现在把 ablation.json 写到 output.dir/<run_name>/
    res_path = out_root / run_name / "ablation.json"
    if not res_path.exists():
        return {"seed": seed, "ok": False,
                "error": f"找不到结果文件 {res_path}", "sec": elapsed}
    data = json.loads(res_path.read_text())
    # 校验：结果里记录的 seed 必须与请求一致，防止并发写错文件
    got = data.get("seeds")
    if got != [seed]:
        return {"seed": seed, "ok": False,
                "error": f"结果文件 seed 不匹配: 期望 {[seed]}, 实际 {got}", "sec": elapsed}
    return {"seed": seed, "ok": True, "data": data, "sec": elapsed}


def main() -> int:
    ap = argparse.ArgumentParser(description="多 seed 消融实验并行执行")
    ap.add_argument("--env", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=3, help="并行进程数")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="outputs/parallel")
    ap.add_argument("--threads", type=int, default=0,
                    help="每个进程的 CPU 线程数（0=自动按核心数/进程数分配）。"
                         "CPU 并行时设小可避免线程争抢")
    args = ap.parse_args()

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"并行消融：{len(args.seeds)} 个 seed × {len(CONFIGS)} 配置")
    if args.threads <= 0 and args.device == "cpu":
        # CPU 并行：核心数平均分给各进程。
        # 实测 torch 在 16 核机器上 8 线程最快（16 线程反而慢一倍，
        # 超过物理核数后调度开销占主导），因此上限取 8。
        cores = os.cpu_count() or 4
        args.threads = max(1, min(8, cores // max(1, args.workers)))
    print(f"并行度: {args.workers} | 设备: {args.device} | 每轮 {args.epochs} epoch"
          + (f" | 每进程 {args.threads} 线程" if args.threads > 0 else ""))
    print("=" * 70, flush=True)

    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_single, s, args.env, args.device, args.epochs,
                          out_root, args.threads): s
                for s in args.seeds}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["seed"]] = r
            status = "✓" if r["ok"] else "✗"
            print(f"[{status}] seed {r['seed']} 用时 {r['sec']:.0f}s"
                  + ("" if r["ok"] else f"  错误: {r['error'][:200]}"), flush=True)

    total = time.time() - t0
    ok_seeds = sorted(s for s, r in results.items() if r["ok"])
    print(f"\n完成 {len(ok_seeds)}/{len(args.seeds)} 个 seed，总用时 {total/60:.1f} 分钟", flush=True)
    if not ok_seeds:
        return 1

    # 防呆：若有多个 seed 结果完全相同，说明并发写冲突，直接报错
    sigs = {}
    for s_ in ok_seeds:
        vals = tuple(round(results[s_]["data"]["results"][cfg][0]["mae_overall"], 3)
                     for cfg in CONFIGS)
        sigs.setdefault(vals, []).append(s_)
    dup = {v: ss for v, ss in sigs.items() if len(ss) > 1}
    if dup:
        print("\n[错误] 检测到多个 seed 的结果完全相同，说明发生了文件覆盖：")
        for v, ss in dup.items():
            print(f"  seed {ss} 结果一致: {v}")
        print("请检查输出目录是否隔离，重跑前先清空 outputs/parallel/")
        return 1

    # 汇总
    summary = {}
    for cfg in CONFIGS:
        vals = [results[s]["data"]["results"][cfg][0]["mae_overall"] for s in ok_seeds]
        summary[cfg] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "per_seed": vals}

    print("\n" + "=" * 70)
    print(f"消融结果（{len(ok_seeds)} 个 seed）")
    print("=" * 70)
    for cfg, v in summary.items():
        print(f"  {cfg:<16} {v['mean']:7.1f} ± {v['std']:5.1f} km")

    # 配对检验
    if len(ok_seeds) > 1:
        from scipy import stats
        b = np.array(summary["baseline(仅轨迹)"]["per_seed"])
        e = np.array(summary["+环境场"]["per_seed"])
        t, p = stats.ttest_rel(e, b)
        d = (e - b).std(ddof=1)
        print(f"\n+环境场 vs baseline: 差 {(e-b).mean():+.1f} km, "
              f"p={p:.4f}, d={((e-b).mean()/d if d>1e-9 else 0):+.2f}, "
              f"环境场更好 {(e<b).sum()}/{len(e)}")
        if p < 0.05:
            print("  → 统计显著，环境场确实带来改进")
        elif p < 0.2:
            print("  → 趋势明显但未达显著，需要更多数据")
        else:
            print("  → 无显著差异")

    # 汇总写到 --out 指定目录下，避免多次实验互相覆盖
    out_json = out_root / "parallel_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": args.env,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "device": args.device,
        "summary": summary,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
