"""命令行入口: 提取并缓存卫星云图 patch。

从 9574 个 NetCDF 中为每个时次裁出以台风中心为中心的云图 patch，
存成 npz 供训练直接读取（避免每次训练都解压 22GB 压缩包）。

用法:
    python scripts/build_sat_patches.py                 # 默认 32x32, 3 个红外通道
    python scripts/build_sat_patches.py --size 48       # 更大视野
    python scripts/build_sat_patches.py --limit 200     # 只处理前 200 帧（调试）

输出: data/processed/sat_patches.npz
  keys  (N,)   "<台风名>|<时间>"
  patches (N, C, S, S) float32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _data_path(cfg, value):
    """把配置路径解析成绝对路径，支持 TYPHOON_DATA_ROOT 挂载外部数据。"""
    from src.config import resolve_path
    return resolve_path(cfg, value, ROOT)

sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.fysatty import extract_patch, list_satellite_files  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="提取卫星云图 patch 并缓存")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--size", type=int, default=None, help="patch 边长（像素，4km/像素）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 帧，调试用")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    size = args.size or int(cfg.satellite.patch_size)
    raw_dir = _data_path(cfg, cfg.data.raw_dir) / "bsts"
    out_path = _data_path(cfg, cfg.data.processed_dir) / cfg.satellite.patch_cache

    zip_dir = _data_path(cfg, cfg.satellite.zip_dir)
    # 清单文件存在就用它，否则直接扫描目录（便于你后续追加新数据包）
    zip_list = raw_dir / "zip_list.txt"
    if zip_list.exists():
        paths = [l.strip() for l in zip_list.read_text().splitlines() if l.strip()]
        paths = [p for p in paths if Path(p).exists()]
    else:
        paths = sorted(str(p) for p in zip_dir.glob("*.zip"))
    if not paths:
        print(f"[patch] 在 {zip_dir} 没有找到卫星数据包 *.zip")
        return 1
    files = list_satellite_files(paths)
    if args.limit:
        files = files.iloc[: args.limit]

    # 同一时次多卫星：优先保留 FY4-B（分辨率更高），否则第一个
    files = files.copy()
    files["prio"] = files["satellite"].map({"FY4-B": 0, "FY4-A": 1, "FY2-H": 2, "FY2-G": 3}).fillna(9)
    files = files.sort_values(["name", "time", "prio"])
    files = files.drop_duplicates(subset=["name", "time"], keep="first").reset_index(drop=True)
    print(f"[patch] 去重后 {len(files)} 帧（每时次选优先级最高的卫星），patch={size}x{size}")

    t0 = time.time()
    keys, patches = [], []
    n_fail = 0
    for i, row in enumerate(files.itertuples(index=False), 1):
        try:
            r = extract_patch(row.zip_path, row.member, half_size=size // 2, out_size=size)
            keys.append(f"{row.name}|{row.time}")
            patches.append(r["patch"].astype(np.float32))
        except Exception as exc:
            n_fail += 1
            if n_fail <= 5:
                print(f"[patch] 失败 {row.member.split('/')[-1][:45]}: {type(exc).__name__}")
        if i % 500 == 0 or i == len(files):
            el = time.time() - t0
            eta = el / i * (len(files) - i)
            print(f"[patch] {i}/{len(files)} 已用 {el / 60:.1f} 分钟, 预计还需 {eta / 60:.1f} 分钟")

    if not patches:
        print("[patch] 没有成功提取任何帧")
        return 1

    arr = np.stack(patches)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, keys=np.array(keys), patches=arr)
    print(f"[patch] 完成: {len(keys)} 帧, 形状 {arr.shape}, "
          f"文件 {out_path.stat().st_size / 1e6:.1f} MB, 失败 {n_fail} 帧")
    print(f"[patch] 总耗时 {(time.time() - t0) / 60:.1f} 分钟 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
