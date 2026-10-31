"""给已有的微波 patch npz 补上台风名字段。

背景
----
build_patch_microwave.py 生成的 keys 是 "<IBTrACS sid>|<时间>"，而现有红外
缓存（sat_patches64.npz）的 keys 是 "<台风名>|<时间>"。两者命名体系不同，
直接按 keys 取交集会得到 0 —— 微波和红外永远对不上。

本脚本用**位置+时间**建立 sid -> 台风名 的映射（不靠 tcid 前缀，那个会重名）：
  对每个微波帧，取文件内 meta 的 (lat, lon, time)，在红外标量表
  sat_scalars.parquet 里找 ±30 min 内最近的记录，距离 <150 km 视为命中，
  再按 sid 取众数得到名字。实测中位距离 20.9 km，覆盖 203/213 个 sid。

产物：在每个 sat_patches_mw64_*.npz 里追加 names 数组（与 keys 一一对应）。
未匹配到的实测只有 1.2%（多为红外未覆盖的台风），记为 ''。

用法:
    python scripts/data/annotate_mw_names.py
    python scripts/data/annotate_mw_names.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _data_path(cfg, value):
    from src.config import resolve_path
    return resolve_path(cfg, value, ROOT)


sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def build_map(patches, scalars, tol_min=30, tol_km=150.0):
    """返回 sid -> 台风名。"""
    sc = scalars.copy()
    sc["t"] = pd.to_datetime(sc["time"], utc=True)
    sc = sc.sort_values("t")
    recs = []
    for f in patches:
        d = np.load(f)
        for i, k in enumerate(d["keys"]):
            sid, ts = str(k).split("|", 1)
            m = json.loads(d["meta"][i])
            recs.append((sid, pd.Timestamp(ts, tz="UTC"), m["lat"], m["lon"]))
    mw = pd.DataFrame(recs, columns=["sid", "t", "lat", "lon"])
    hits = []
    for r in mw.itertuples(index=False):
        w = sc[(sc["t"] >= r.t - pd.Timedelta(minutes=tol_min))
               & (sc["t"] <= r.t + pd.Timedelta(minutes=tol_min))]
        if w.empty:
            continue
        dd = haversine_km(r.lat, r.lon, w["cent_lat"].values, w["cent_lon"].values)
        j = int(np.argmin(dd))
        if dd[j] < tol_km:
            hits.append((r.sid, str(w["name"].iloc[j]).upper(), float(dd[j])))
    h = pd.DataFrame(hits, columns=["sid", "name", "dist"])
    if h.empty:
        return {}, mw, h
    g = h.groupby("sid")["name"].agg(lambda x: x.value_counts().index[0])
    return dict(g), mw, h


def main() -> int:
    ap = argparse.ArgumentParser(description="给微波 npz 补 names 字段")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    proc = _data_path(cfg, cfg.data.processed_dir)
    patches = sorted(glob.glob(str(proc / "sat_patches_mw64_*.npz")))
    scalars = pd.read_parquet(proc / "sat_scalars.parquet")
    if not patches:
        print(f"[错误] {proc} 下没有 sat_patches_mw64_*.npz")
        return 1

    print(f"[输入] {len(patches)} 个微波 npz")
    sid2name, mw, hits = build_map(patches, scalars)
    if hits.empty:
        print("[错误] 没有任何帧能匹配到红外标量表")
        return 1
    print(f"[映射] 命中 {len(hits)}/{len(mw)} 帧, "
          f"中位距离 {hits['dist'].median():.0f} km, sid {len(sid2name)} 个")

    total = unmapped = 0
    for f in patches:
        p = Path(f)
        d = np.load(f)
        if "names" in d.files:
            print(f"  {p.name}: 已有 names，跳过")
            continue
        names = np.array([sid2name.get(str(k).split("|")[0], "") for k in d["keys"]])
        n0 = int((names == "").sum())
        total += len(names)
        unmapped += n0
        if args.dry_run:
            print(f"  {p.name}: {len(names)} 帧, 未映射 {n0} (dry-run)")
            continue
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, keys=d["keys"], patches=d["patches"],
                            channel_mask=d["channel_mask"], meta=d["meta"],
                            names=names)
        tmp.replace(p)
        print(f"  {p.name}: {len(names)} 帧, 未映射 {n0}")
    print(f"\n[完成] 共 {total} 帧, 未映射 {unmapped} "
          f"({unmapped / max(1, total) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
