"""纬度自适应路径预测：在关键纬度带使用环境场，其余用纯轨迹。

动机
----
分层消融发现（10 seed，24h 误差）：

| 纬度带 | baseline | +环境场 | 差值 | 相对 |
|---|---|---|---|---|
| 15-25N | 205.4 | 201.8 | -3.6 | -1.7% |
| **25-35N** | **378.4** | **357.2** | **-21.2** | **-5.6%** |
| 35N+ | 466.0 | 469.1 | +3.0 | +0.7% |

25-35N 是副高边缘的台风转向带，引导气流决定路径，环境场贡献最大
（逐 seed 配对 t 检验 p=0.0098，8/10 seed 一致改善）。

而在 35N+ 环境场反而略差（中纬度受西风槽主导，0.25° 的 500hPa
平均场刻画不足）。

模型设计
--------
训练两个模型：
  * M_track : 纯轨迹（8 维特征）
  * M_env   : 轨迹 + 环境场（17 维特征）

推理时按**预报起始纬度**路由（该纬度在预报时刻已知，无需未来信息）：
  * USE_LAT_MIN <= lat < USE_LAT_MAX  ->  M_env
  * 其余                              ->  M_track

默认带为 [15, 35)，由消融实验确定（见 LANDMARK 注释）。

用法
----
    # 训练 + 评估（K 折，最严谨）
    python scripts/path/latitude_adaptive.py --env .scratch/env_era5.parquet \
        --folds 5 --device cuda

    # 单次划分
    python scripts/path/latitude_adaptive.py --env .scratch/env_era5.parquet \
        --seeds 0 1 2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "path"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.data.features import haversine_km  # noqa: E402
from src.evaluation.stratified import classify_region  # noqa: E402

# 环境场生效的纬度带（由分层消融确定）
USE_LAT_MIN, USE_LAT_MAX = 15.0, 35.0


def eval_policy(pred_track, pred_env, true_abs, last_abs,
                lat_min=USE_LAT_MIN, lat_max=USE_LAT_MAX):
    """返回各策略的逐时效误差 (N,4)。

    策略:
      track   纯轨迹
      env     全程环境场
      adapt   纬度自适应（本脚本提出的方案）
    """
    def dist(pred):
        d = haversine_km(pred[..., 0], pred[..., 1],
                         true_abs[..., 0], true_abs[..., 1])
        return d

    lat = last_abs[:, 0]
    use_env = (lat >= lat_min) & (lat < lat_max)
    out = {
        "track": dist(pred_track),
        "env": dist(pred_env),
        "adapt": dist(np.where(use_env[:, None, None], pred_env, pred_track)),
    }
    return out, use_env


def summarize(name_arr: dict, n_seeds: int) -> pd.DataFrame:
    """汇总逐时效与平均误差。

    注意: 误差数组是 (N, 4)，必须按第 2 轴（时效）取均值，
    写成 arr[0].mean() 会取到"第 0 个样本的 4 个时效"，是错的。
    """
    rows = []
    for name, arr in name_arr.items():
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"{name}: 期望形状 (N,4)，实际 {arr.shape}")
        rows.append({
            "策略": name,
            "6h": float(arr[:, 0].mean()),
            "12h": float(arr[:, 1].mean()),
            "18h": float(arr[:, 2].mean()),
            "24h": float(arr[:, 3].mean()),
            "平均": float(arr.mean()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="纬度自适应路径预测")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--env", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold-seeds", nargs="*", type=int, default=[],
                    help="用多个不同的折划分 seed 重复 K 折，检验结果对划分的稳健性。"
                         "不指定时用 cfg.seed 单个划分")
    ap.add_argument("--out", default="outputs/path/latitude_adaptive")
    args = ap.parse_args()

    import torch
    from ablate_environment import (  # noqa: E402
        _align_timezones, build_bundle, kfold_by_storm, run_one,
    )

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, args.set)
    cfg.device = device

    tracks = pd.read_parquet(resolve_path(cfg, cfg.data.raw_dir, ROOT)
                             / f"tracks_{cfg.data.basin}_clean.parquet")
    env = pd.read_parquet(args.env)
    tracks, env = _align_timezones(tracks, env)
    lo, hi = env["time"].min(), env["time"].max()
    sids = set(env["sid"].unique())
    tracks = tracks[(tracks["time"] >= lo) & (tracks["time"] <= hi)
                    & tracks["sid"].isin(sids)].reset_index(drop=True)
    print(f"[lat-adapt] 轨迹 {len(tracks)} 条 / {tracks['sid'].nunique()} 台风")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.folds and args.folds > 1:
        base_bundle, _ = build_bundle(tracks, env, True, False, cfg, 0)
        # 用 cfg.seed，与 ablate_environment.py 的 K 折保持一致，便于横向对比
        fseeds = args.fold_seeds or [int(cfg.seed)]
        jobs = []
        base_bundle, _ = build_bundle(tracks, env, True, False, cfg, 0)
        for fs in fseeds:
            folds = kfold_by_storm(base_bundle, n_folds=args.folds, seed=fs)
            # (job_id, 训练随机种子, 固定划分) —— job_id 仅用于文件名，
            # 必须与传给 run_one 的整数 seed 分开，否则 np.random 会报错
            jobs += [(f"{fs}_{k}", int(fs), f) for k, f in enumerate(folds)]
        print(f"[lat-adapt] K 折: {args.folds} 折 × {len(fseeds)} 个划分 seed "
              f"{fseeds} = {len(jobs)} 个 job")
    else:
        jobs = [(str(s), int(s), None) for s in args.seeds]

    per_job = []
    for job_id, seed_int, fixed in jobs:
        print(f"\n{'=' * 62}\njob={job_id}\n{'=' * 62}")
        preds = {}
        for tag, kw in [("track", dict(use_track=True, use_env=False)),
                        ("env", dict(use_track=True, use_env=True))]:
            ov = list(args.set) + [f"train.epochs={args.epochs}"]
            pth = out_dir / f"pred_{tag}_j{job_id}.npz"
            r = run_one(cfg, tracks, env, tag=tag, seed=seed_int, overrides=ov,
                        save_pred_for=str(pth), fixed_splits=fixed, **kw)
            preds[tag] = np.load(pth)
            print(f"  {tag:6s} 平均MAE={r['mae_overall']:.1f} km")
        errs, use_env = eval_policy(preds["track"]["pred_abs"],
                                    preds["env"]["pred_abs"],
                                    preds["track"]["true_abs"],
                                    preds["track"]["last_abs"])
        per_job.append({"job": job_id, "errs": errs, "use_env": use_env,
                        "last": preds["track"]["last_abs"]})

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print("各策略误差（km）")
    print("=" * 70)
    allerr = {k: np.concatenate([j["errs"][k] for j in per_job]) for k in ("track", "env", "adapt")}
    tab = summarize(allerr, len(per_job))
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # 配对检验（逐 job）
    print("\n" + "=" * 70)
    print("配对检验（每个 job 一个点，相对 track 的 24h 误差差）")
    print("=" * 70)
    from scipy import stats
    t24 = np.array([j["errs"]["track"][:, -1].mean() for j in per_job])
    e24 = np.array([j["errs"]["env"][:, -1].mean() for j in per_job])
    a24 = np.array([j["errs"]["adapt"][:, -1].mean() for j in per_job])
    print(f"{'job':>8} {'track':>9} {'env':>9} {'adapt':>9}")
    for i in range(len(per_job)):
        print(f"{str(per_job[i]['job']):>8} {t24[i]:>9.1f} {e24[i]:>9.1f} {a24[i]:>9.1f}")
    for name, arr in [("全程环境场", e24), ("纬度自适应", a24)]:
        d = arr - t24
        t, p = stats.ttest_rel(arr, t24)
        print(f"\n{name} vs 纯轨迹: 差 {d.mean():+.1f} km "
              f"({(d.mean()/t24.mean()*100):+.2f}%), 改善 {(d<0).sum()}/{len(d)}, "
              f"p={p:.4f}, d={d.mean()/d.std(ddof=1):+.2f}")

    # 自适应实际覆盖比例
    use = np.concatenate([j["use_env"] for j in per_job])
    print(f"\n自适应策略中走环境场分支的样本: {use.mean():.1%}")
    print(f"纬度带: [{USE_LAT_MIN}, {USE_LAT_MAX})")

    tab.to_csv(out_dir / "policy_compare.csv", index=False, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "policy": tab.to_dict(orient="records"),
        "lat_band": [USE_LAT_MIN, USE_LAT_MAX],
        "jobs": [j["job"] for j in per_job],
        "track24": t24.tolist(), "env24": e24.tolist(), "adapt24": a24.tolist(),
        "env_coverage": float(use.mean()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
