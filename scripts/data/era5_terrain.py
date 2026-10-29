"""命令行入口: 提取台风路径上的地形特征。

前置: ERA5 目录下需有静态场
    <era5_dir>/geopotential_at_surface/0.0
    <era5_dir>/land_sea_mask/0.0
（两份静态场各约 3MB，可从 ARCO-ERA5 数据集直接获取）

用法:
    python scripts/data/era5_terrain.py
    python scripts/data/era5_terrain.py --src /mnt/datasets/typhoon/raw/era5 \
        --out /mnt/datasets/typhoon/processed/env_terrain.parquet

输出: <processed_dir>/env_terrain.parquet
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
    ap = argparse.ArgumentParser(description="提取地形特征")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--src", default=None, help="ERA5 目录（含静态场）")
    ap.add_argument("--out", default=None, help="输出 parquet")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条轨迹（调试）")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    era5_dir = (Path(args.src) if args.src
                else resolve_path(cfg, cfg.data.raw_dir, ROOT) / "era5")
    out_path = (Path(args.out) if args.out
                else resolve_path(cfg, cfg.data.processed_dir, ROOT) / "env_terrain.parquet")
    tracks_path = resolve_path(cfg, cfg.data.raw_dir, ROOT) / f"tracks_{cfg.data.basin}_clean.parquet"

    if not tracks_path.exists():
        print(f"[错误] 缺少轨迹数据 {tracks_path}")
        return 1

    from src.data.arco_era5 import ARCOPaths
    from src.data.terrain import (
        LANDMASK_VAR, STATIC_FILE, TERRAIN_FEATURES, TERRAIN_VAR, build_terrain_table,
    )

    # 先检查静态场是否存在
    paths = ARCOPaths(era5_dir)
    missing = [v for v in (TERRAIN_VAR, LANDMASK_VAR)
               if not (era5_dir / v / STATIC_FILE).exists()]
    if missing:
        print("[错误] 缺少静态场文件:")
        for v in missing:
            print(f"    {era5_dir / v / STATIC_FILE}")
        print("\n生成下载清单后重下（清单已自动包含这两个文件）:")
        print("        geopotential_at_surface/0.0")
        print("        land_sea_mask/0.0")
        return 1

    tracks = pd.read_parquet(tracks_path)
    if args.limit:
        tracks = tracks.head(args.limit)
    print(f"[terrain] 轨迹 {len(tracks)} 条 / {tracks['sid'].nunique()} 个台风")

    table = build_terrain_table(paths, tracks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    print(f"\n[terrain] 已保存 {len(table)} 行 -> {out_path}")

    print("\n[terrain] 特征统计:")
    for c in TERRAIN_FEATURES:
        col = table[c]
        print(f"  {c:16s} 均值={col.mean():9.2f}  "
              f"范围=[{col.min():9.2f}, {col.max():9.2f}]")

    # 分布概况：有多少轨迹点受地形阻挡
    if "blocked" in table:
        n_blk = int(table["blocked"].sum())
        print(f"\n[terrain] 前方有显著地形阻挡的轨迹点: {n_blk} / {len(table)} "
              f"({n_blk / len(table) * 100:.1f}%)")
    if "land_frac" in table:
        n_land = int((table["land_frac"] > 0.1).sum())
        print(f"[terrain] 陆地占比 >10% 的轨迹点: {n_land} / {len(table)} "
              f"({n_land / len(table) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
