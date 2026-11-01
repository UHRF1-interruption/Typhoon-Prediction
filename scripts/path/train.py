"""命令行入口: 训练模型。

用法:
  python scripts/train.py                                      # 用默认配置
  python scripts/train.py --set model.name=gru train.epochs=10 # 换基线
  python scripts/train.py --set device=cpu                     # 强制 CPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _data_path(cfg, value):
    """把配置路径解析成绝对路径，支持 TYPHOON_DATA_ROOT 挂载外部数据。"""
    from src.config import resolve_path
    return resolve_path(cfg, value, ROOT)

sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import dump_config, load_config  # noqa: E402
from src.data.dataset import load_bundle, split_by_year  # noqa: E402
from src.training.losses import format_metrics_table  # noqa: E402
from src.training.trainer import train  # noqa: E402


def _build_with_env(cfg, processed: Path, env_path: Path):
    """把环境场特征并入轨迹后重建样本。

    存在的理由：`data/processed/samples.npz` 只有 8 维轨迹特征，不含 u500/v500。
    无标签物理惩罚的 steering 项需要引导层风，故必须在此并入。
    """
    import pandas as pd

    from src.data.environment import ENV_FEATURES
    from src.data.features import build_samples, fit_feature_scaler
    from src.data.dataset import FeatureSpec

    from src.config import resolve_path
    raw = resolve_path(cfg, cfg.data.raw_dir, ROOT)
    tracks = pd.read_parquet(raw / f"tracks_{cfg.data.basin}_clean.parquet")
    env = pd.read_parquet(env_path)
    cols = [c for c in ENV_FEATURES if c in env.columns]
    if not cols:
        raise ValueError(f"环境文件无可用特征，实际列: {list(env.columns)}")
    # 两边的 time 时区/精度不一致（轨迹是 tz-aware UTC），统一成本地时区无关的
    # naive datetime 再合并，避免 merge 报 datetime64 不匹配
    tracks = tracks.copy()
    env = env.copy()
    for d in (tracks, env):
        t = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
        d["time"] = t
    merged = tracks.merge(env[["sid", "time"] + cols], on=["sid", "time"], how="left")
    covered = merged[cols[0]].notna().mean()
    print(f"[data] 并入环境场 {len(cols)} 维: {cols}")
    print(f"[data] 环境场覆盖 {covered:.1%} 的行、{merged.loc[merged[cols[0]].notna(),'sid'].nunique()} 个台风")
    if covered < 0.5:
        # 不丢弃未覆盖行会让 build_samples 用 0 填充缺失环境场，
        # 这类"假零"在标准化后是一个固定值，会把回归结果稀释掉
        # （实测不限制时引导系数的 R² 从 0.69 掉到 0.06）。
        keep_sids = set(merged.loc[merged[cols[0]].notna(), "sid"].unique())
        merged = merged[merged["sid"].isin(keep_sids)].reset_index(drop=True)
        print(f"[data] 已限制到有环境场的台风: {len(keep_sids)} 个，"
              f"{len(merged)} 行")

    base = ["dlat", "dlon", "dist_prev", "sin_heading", "cos_heading",
            "wind_ms", "dpres_hpa", "lat_norm"]
    feats = base + cols
    b = build_samples(merged, in_len=int(cfg.data.in_len), out_len=int(cfg.data.out_len),
                      interval_hours=float(cfg.data.interval_hours), feature_names=feats)
    spec = fit_feature_scaler(b["x"])
    spec.names = feats
    return b, spec


def main() -> int:
    ap = argparse.ArgumentParser(description="训练台风路径预测模型")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置，如 model.name=gru")
    ap.add_argument("--split", choices=["year", "storm"], default="year",
                    help="划分方式。year=按年份（默认，与历史口径一致）；"
                         "storm=按台风随机划分。**并入环境场后建议用 storm**："
                         "环境场常只覆盖部分年份，按年份切会得到空验证/测试集。")
    ap.add_argument("--env", default=None,
                    help="并入环境场 parquet（如 data/processed/env_era5.parquet）。"
                         "无标签物理惩罚的 steering 项需要输入含 u500/v500，"
                         "必须用本参数并入环境场才会生效。")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    processed = _data_path(cfg, cfg.data.processed_dir)
    if args.env:
        bundle, spec = _build_with_env(cfg, processed, _data_path(cfg, args.env))
    else:
        bundle, spec = load_bundle(processed)
    print(f"[data] 载入 {len(bundle['x'])} 个样本，特征维度 {bundle['x'].shape[-1]}")

    splits = None
    if args.split == "storm":
        from src.data.dataset import split_by_storm_random
        splits = split_by_storm_random(bundle, seed=int(cfg.seed))
    result = train(cfg, bundle, spec, splits)

    out_dir = result["out_dir"]
    dump_config(cfg, out_dir / "used_config.yaml")
    print("\n" + "=" * 70)
    print(f"训练完成: 最佳 epoch={result['best_epoch']}, 验证误差={result['best_val_mae_km']:.1f} km")
    if result["test_metrics"]:
        m = result["test_metrics"]
        row = {str(cfg.model.name): {"mae_by_step": m["mae_by_step"], "horizon": m["horizon"],
                                     "mae_overall": m["mae_overall"]}}
        print(format_metrics_table(row, float(cfg.data.interval_hours)))
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
