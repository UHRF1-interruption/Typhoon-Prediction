"""分层消融：检查环境场是否只在特定子集有效。

之前所有消融都只在**全局**做，得出"环境场无效"的结论。
但全局平均可能掩盖子集效应，例如：
  * 日本以南转向频繁，环境场（引导气流）可能特别有用
  * 中等强度台风路径飘忽，环境场可能帮助稳定

本脚本一次训练后，按海域/强度/纬度拆开对比 baseline 与 +环境场。

用法:
    python scripts/path/stratified_ablation.py --env ../../.scratch/env_era5.parquet \
        --device cuda --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.features import haversine_km  # noqa: E402
from src.evaluation.stratified import compare_stratified, stratified_report  # noqa: E402


def err24_km(pred_abs, true_abs):
    """24h 球面误差（km）。pred_abs / true_abs 均为 (N, 4, 2) 绝对坐标。"""
    return haversine_km(pred_abs[:, -1, 0], pred_abs[:, -1, 1],
                        true_abs[:, -1, 0], true_abs[:, -1, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="分层消融")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--env", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="outputs/stratified_ablation")
    ap.add_argument("--folds", type=int, default=0,
                    help="K 折交叉验证的折数（推荐 5）。>0 时忽略 --seeds")
    args = ap.parse_args()

    import torch
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")

    from ablate_environment import (  # 同目录
        _align_timezones, build_bundle, run_one, split_by_storm_random,
    )

    cfg = load_config(args.config, args.set)
    cfg.device = device
    from src.config import resolve_path
    tracks = pd.read_parquet(resolve_path(cfg, cfg.data.raw_dir, ROOT)
                             / f"tracks_{cfg.data.basin}_clean.parquet")
    env = pd.read_parquet(args.env)
    tracks, env = _align_timezones(tracks, env)

    # 裁剪到环境场覆盖范围
    lo, hi = env["time"].min(), env["time"].max()
    sids = set(env["sid"].unique())
    tracks = tracks[(tracks["time"] >= lo) & (tracks["time"] <= hi)
                    & tracks["sid"].isin(sids)].reset_index(drop=True)
    print(f"[strat-ablate] 轨迹 {len(tracks)} 条 / {tracks['sid'].nunique()} 台风")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # K 折模式下，每折当作一个"seed"，用固定划分保证可比
    if args.folds and args.folds > 1:
        from ablate_environment import build_bundle, kfold_by_storm
        base_bundle, _ = build_bundle(tracks, env, True, False, cfg, 0)
        folds = kfold_by_storm(base_bundle, n_folds=args.folds, seed=0)
        jobs = [(k, f) for k, f in enumerate(folds)]
        print(f"[strat-ablate] K 折模式: {args.folds} 折")
    else:
        jobs = [(seed, None) for seed in args.seeds]

    per_seed = []
    for seed, fixed in jobs:
        print(f"\n{'=' * 60}\n{'=' * 60}")
        row = {"seed": seed}
        for tag, kw in [("baseline", dict(use_track=True, use_env=False)),
                        ("env", dict(use_track=True, use_env=True))]:
            pred_path = out_dir / f"pred_{tag}_s{seed}.npz"
            ov = list(args.set) + [f"train.epochs={args.epochs}"]
            r = run_one(cfg, tracks, env, tag=tag, seed=seed,
                        overrides=ov, save_pred_for=str(pred_path),
                        fixed_splits=fixed, **kw)
            row[tag] = r
            print(f"  {tag:10s} MAE={r['mae_overall']:.2f} km")
        per_seed.append(row)

    # ---------- 逐 seed 分层对比 ----------
    print("\n" + "=" * 70)
    print("分层分析（每个 seed 单独看，最后汇总）")
    print("=" * 70)
    agg = {}
    for row in per_seed:
        seed = row["seed"]
        a = np.load(out_dir / f"pred_baseline_s{seed}.npz")
        b = np.load(out_dir / f"pred_env_s{seed}.npz")
        if len(a["pred_abs"]) != len(b["pred_abs"]):
            print(f"  [警告] seed {seed} 样本数不一致，跳过")
            continue
        last = a["last_abs"]
        err_a = err24_km(a["pred_abs"], a["true_abs"])
        err_b = err24_km(b["pred_abs"], b["true_abs"])

        # 海域
        from src.evaluation.stratified import classify_region
        lon = last[:, 1] % 360
        region = classify_region(last[:, 0], lon)
        region = np.where(lon > 180, "远洋(菲东)", region)

        # 纬度带
        latb = pd.cut(last[:, 0], bins=[0, 15, 25, 35, 60],
                      labels=["0-15N", "15-25N", "25-35N", "35N+"]).astype(object)

        for name, labels in [("海域", region), ("纬度", np.asarray(latb, object))]:
            tab = compare_stratified(err_a, err_b, labels,
                                     name_a="baseline", name_b="+环境场", min_n=30)
            if len(tab):
                tab["seed"] = seed
                tab["dim"] = name
                agg.setdefault(name, []).append(tab)

    # ---------- 汇总各 seed ----------
    for name, tabs in agg.items():
        df = pd.concat(tabs, ignore_index=True)
        g = df.groupby("分组", sort=False).agg(
            样本数=("样本数", "mean"),
            baseline=("baseline", "mean"),
            env=("+环境场", "mean"),
            差值=("差值", "mean"),
            相对百分比=("相对%", "mean"),
            seed数=("seed", "count"),
        ).reset_index()
        g = g[g["seed数"] >= max(1, len(args.seeds) // 2)]
        print(f"\n{'=' * 70}")
        print(f"【{name}】跨 {len(args.seeds)} seed 汇总")
        print("=" * 70)
        print(g.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
        g.to_csv(out_dir / f"stratified_{name}.csv", index=False, encoding="utf-8")

    summary = [{"seed": r["seed"], "baseline": r["baseline"]["mae_overall"],
                "env": r["env"]["mae_overall"]} for r in per_seed]
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts" / "path"))
    raise SystemExit(main())
