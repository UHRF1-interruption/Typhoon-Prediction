"""命令行入口: 解析本地 ARCO-ERA5 分块，生成环境场特征表。

前置条件: ARCO-ERA5 分块已放到
          data/raw/era5/ 下（保持 <变量名>/<idx>.0.0.0 结构）。

用法:
    # 1) 先检查下载完整性
    python scripts/parse_arco_era5.py --check

    # 2) 解析并生成环境表（默认输出 env_era5.parquet）
    python scripts/parse_arco_era5.py

    # 3) 指定下载目录与输出
    python scripts/parse_arco_era5.py --src data/raw/era5 --out data/processed/env_era5.parquet

产出: data/processed/env_era5.parquet
      可直接用于 ablate_environment.py --env data/processed/env_era5.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.arco_era5 import (  # noqa: E402
    LEVELS, STEP_HOURS, TIME_ORIGIN, VAR_DIRS, ARCOPaths,
    build_env_table, index_to_time,
)


def check(src: Path) -> int:
    """检查下载完整性：各变量的时次数是否一致、是否有缺失。"""
    print("=" * 68)
    print(f"检查 ARCO-ERA5 下载目录: {src}")
    print("=" * 68)
    if not src.exists():
        print(f"[错误] 目录不存在")
        return 1

    paths = ARCOPaths(src)
    stats = {}
    for var, dname in VAR_DIRS.items():
        d = src / dname
        if not d.exists():
            print(f"  {dname:28s} [缺失] 目录不存在")
            continue
        files = [p for p in d.iterdir() if p.name.split(".")[0].isdigit()]
        if not files:
            print(f"  {dname:28s} [空]")
            continue
        idxs = sorted(int(p.name.split(".")[0]) for p in files)
        total = sum(p.stat().st_size for p in files) / 1e9
        stats[var] = set(idxs)
        print(f"  {dname:28s} {len(idxs):5d} 个文件  {total:7.2f} GB  "
              f"{index_to_time(idxs[0]):%Y-%m-%d} ~ {index_to_time(idxs[-1]):%Y-%m-%d}")

    if not stats:
        print("\n[错误] 没有找到任何数据文件")
        return 1

    # 一致性检查：各变量应覆盖相同的时次
    common = set.intersection(*stats.values())
    union = set.union(*stats.values())
    print(f"\n  共同时次: {len(common)} / 并集 {len(union)}")
    if len(common) < len(union):
        print("  [警告] 各变量时次不一致，缺失的时次将无法计算完整特征：")
        for var, s in stats.items():
            miss = union - s
            if miss:
                m = sorted(miss)
                print(f"    {VAR_DIRS[var]:28s} 缺 {len(miss)} 个（如 {m[:5]}）")
    else:
        print("  [OK] 所有变量时次完全一致")

    # 抽样验证一个文件能否正确解压
    sample_idx = sorted(common)[len(common) // 2] if common else None
    if sample_idx is not None:
        p = paths.chunk("geopotential", sample_idx)
        if p.exists():
            try:
                from src.data.arco_era5 import N_LAT, N_LON, read_chunk
                z = read_chunk(p, (len(LEVELS), N_LAT, N_LON))
                h500 = z[LEVELS.index(500)] / 9.80665
                print(f"\n  抽样验证 idx={sample_idx} ({index_to_time(sample_idx)}):")
                print(f"    解压成功，500hPa 位势 {np.nanmin(h500):.0f} ~ "
                      f"{np.nanmax(h500):.0f} m（合理范围约 4700-6000）")
            except Exception as exc:
                print(f"\n  [错误] 抽样解压失败: {type(exc).__name__}: {exc}")
                return 1
    print("=" * 68)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="解析 ARCO-ERA5 生成环境场特征表")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--src", default=None,
                    help="ARCO 下载目录（默认按配置解析，可用 TYPHOON_DATA_ROOT 覆盖）")
    ap.add_argument("--out", default=None,
                    help="输出路径（默认 <processed_dir>/env_era5.parquet）")
    ap.add_argument("--check", action="store_true", help="只检查下载完整性")
    args = ap.parse_args()

    from src.config import resolve_path
    _cfg0 = load_config(args.config, args.set)
    src = Path(args.src) if args.src else (resolve_path(_cfg0, _cfg0.data.raw_dir, ROOT) / "era5")
    if args.check:
        return check(src)

    cfg = _cfg0
    tracks_path = ROOT / cfg.data.raw_dir / f"tracks_{cfg.data.basin}_clean.parquet"
    if not tracks_path.exists():
        print(f"[错误] 缺少轨迹数据 {tracks_path}")
        return 1

    import pandas as pd

    tracks = pd.read_parquet(tracks_path)
    print(f"[parse] 轨迹 {len(tracks)} 条 / {tracks['sid'].nunique()} 个台风")

    paths = ARCOPaths(src)
    try:
        env = build_env_table(paths, tracks)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}")
        return 1

    out = (Path(args.out) if args.out
           else (resolve_path(cfg, cfg.data.processed_dir, ROOT) / "env_era5.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)
    env.to_parquet(out, index=False)
    print(f"\n[parse] 已保存 {len(env)} 行 -> {out}")

    from src.data.environment import ENV_FEATURES
    print("\n[parse] 环境特征统计:")
    for c in ENV_FEATURES:
        col = env[c]
        n_ok = int(np.isfinite(col).sum())
        if n_ok:
            print(f"  {c:16s} 有效 {n_ok:6d}  均值={col.mean():9.2f}  "
                  f"范围=[{col.min():8.2f}, {col.max():8.2f}]")
        else:
            print(f"  {c:16s} 全部缺失")

    print("\n下一步:")
    print(f"  .venv/bin/python scripts/ablate_environment.py --env {args.out} --seeds 0 1 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
