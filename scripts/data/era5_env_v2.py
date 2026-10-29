"""命令行入口: 提取 v2 环境特征（业务化改进版）。

与 v1 的区别: 深层加权引导气流、多半径圈、副高脊线、低层切变、方向编码。

用法:
    # 提取并与 v1 对比
    python scripts/data/era5_env_v2.py --compare

    # 只提取 v2
    python scripts/data/era5_env_v2.py

输出:
    <processed_dir>/env_era5_v2.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="提取 v2 环境特征")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--src", default=None, help="ERA5 目录")
    ap.add_argument("--out", default=None, help="输出 parquet")
    ap.add_argument("--compare", action="store_true", help="同时提取 v1 做对比")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个点（调试）")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    era5_dir = (Path(args.src) if args.src
                else resolve_path(cfg, cfg.data.raw_dir, ROOT) / "era5")
    out_path = (Path(args.out) if args.out
                else resolve_path(cfg, cfg.data.processed_dir, ROOT) / "env_era5_v2.parquet")
    tracks_path = resolve_path(cfg, cfg.data.raw_dir, ROOT) / f"tracks_{cfg.data.basin}_clean.parquet"

    if not tracks_path.exists():
        print(f"[错误] 缺少轨迹数据 {tracks_path}")
        return 1
    tracks = pd.read_parquet(tracks_path)

    from src.data.arco_era5 import ARCOPaths
    from src.data.arco_era5_v2 import ENV_FEATURES_V2, build_env_table_v2

    paths = ARCOPaths(era5_dir)
    if args.limit:
        tracks = tracks.head(args.limit)

    print(f"[v2] 提取环境特征（{len(ENV_FEATURES_V2)} 维）...")
    env_v2 = build_env_table_v2(paths, tracks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    env_v2.to_parquet(out_path, index=False)
    print(f"\n[v2] 已保存 {len(env_v2)} 行 -> {out_path}")

    print("\n[v2] 特征统计:")
    for c in ENV_FEATURES_V2:
        col = env_v2[c]
        n_ok = int(np.isfinite(col).sum())
        if n_ok:
            print(f"  {c:16s} 有效 {n_ok:5d}  均值={col.mean():9.3f}  "
                  f"范围=[{col.min():9.2f}, {col.max():9.2f}]")
        else:
            print(f"  {c:16s} 全部缺失")

    if args.compare:
        from src.data.arco_era5 import ENV_FEATURES, build_env_table
        print("\n[v1] 提取对比用特征...")
        env_v1 = build_env_table(paths, tracks, verbose=False)
        print(f"\n{'=' * 70}")
        print("v1 与 v2 对比")
        print("=" * 70)
        print(f"v1: {len(ENV_FEATURES)} 维 -> {', '.join(ENV_FEATURES)}")
        print(f"v2: {len(ENV_FEATURES_V2)} 维 -> {', '.join(ENV_FEATURES_V2)}")
        print(f"\n样本数: v1={len(env_v1)}, v2={len(env_v2)}")
        # 共同行的相关性
        merged = pd.merge(env_v1, env_v2, on=["sid", "time"], suffixes=("_v1", "_v2"))
        if len(merged) and "h500_mean_v1" in merged and "h500_mean_v2" in merged:
            r = np.corrcoef(merged["h500_mean_v1"], merged["h500_mean_v2"])[0, 1]
            print(f"h500_mean 相关性 (v1 vs v2): {r:.4f}")
            print("  （接近 1 说明多半径圈平均没有改变整体量级，只是更精细）")
        if len(merged) and "sst_v1" in merged and "sst_v2" in merged:
            r = np.corrcoef(merged["sst_v1"], merged["sst_v2"])[0, 1]
            print(f"sst 相关性 (v1 vs v2): {r:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
