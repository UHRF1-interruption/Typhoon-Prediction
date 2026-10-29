"""把地形特征对齐到卫星强度数据。

为什么需要对齐
--------------
卫星数据（sat_scalars.parquet）标识台风用**名字**（如 AMPIL、BARIJAT），
地形表（env_terrain.parquet）用 IBTrACS 的**编号**（如 2024214N21110），
两者无法直接 join。

但两者都有 (time, lat, lon)，因此用**时空最近邻**匹配：
对每个卫星时次，在一天内的地形记录中找位置最近的点，
距离 < 0.5° 视为同一条台风。

实测可匹配约 77% 的卫星时次（部分台风不在 IBTrACS 中或缺位置）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 与 src/data/terrain.py 的 TERRAIN_FEATURES 保持一致
TERRAIN_COLS = [
    "terrain_mean", "terrain_max", "terrain_fwd_mean",
    "terrain_high_frac", "blocked", "terrain_grad", "land_frac",
]

MAX_DIST_DEG = 0.5      # 超过此距离不认为是同一台风
MAX_TIME_H = 12.0       # 允许的最大时间偏差（小时）


def align_terrain_to_satellites(scalars: pd.DataFrame,
                                terrain: pd.DataFrame,
                                verbose: bool = True) -> pd.DataFrame:
    """返回 scalars 加上地形列后的表；无法匹配的行地形列为 NaN。

    scalars: 含 name, time, cent_lat, cent_lon
    terrain: 含 time, lat, lon, TERRAIN_COLS
    """
    from scipy.spatial import cKDTree

    s = scalars.copy()
    t = terrain.copy()
    s["time"] = pd.to_datetime(s["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    t["time"] = pd.to_datetime(t["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    # 统一到每台风的逐时次（卫星同一时刻可能有多颗卫星）
    g = (s.groupby(["name", "time"])
         .agg(cent_lat=("cent_lat", "mean"), cent_lon=("cent_lon", "mean"))
         .reset_index())
    g["_date"] = g["time"].dt.floor("D")

    found = 0
    out_rows = []
    t_by_date = {d: sub for d, sub in t.groupby(t["time"].dt.floor("D"))}

    for _, row in g.iterrows():
        d = row["_date"]
        cand = t_by_date.get(d)
        best = None
        if cand is not None and len(cand):
            tree = cKDTree(cand[["lat", "lon"]].to_numpy(float))
            dist, idx = tree.query([row["cent_lat"], row["cent_lon"]], k=1)
            if dist <= MAX_DIST_DEG:
                rec = cand.iloc[int(idx)]
                dt_h = abs((rec["time"] - row["time"]).total_seconds()) / 3600.0
                if dt_h <= MAX_TIME_H:
                    best = rec
        if best is not None:
            found += 1
            vals = {c: float(best[c]) for c in TERRAIN_COLS if c in best.index}
        else:
            vals = {c: np.nan for c in TERRAIN_COLS}
        out_rows.append({"name": row["name"], "time": row["time"], **vals})

    aligned = pd.DataFrame(out_rows)
    if verbose:
        rate = found / max(1, len(g))
        print(f"[terrain-align] 卫星时次匹配 {found}/{len(g)} ({rate:.1%})")

    # 合并回原始 scalars（按 name+time）
    merged = s.merge(aligned, on=["name", "time"], how="left")
    if verbose:
        ok = merged[TERRAIN_COLS].notna().all(axis=1).mean()
        print(f"[terrain-align] 合并后完整地形特征 {ok:.1%}")
    return merged


def forward_fill_terrain(df: pd.DataFrame, max_gap_h: float = 6.0) -> pd.DataFrame:
    """对同一台风相邻时次做小幅前向填充。

    地形在短时间（<6h）内变化很小，填充可提升覆盖率；
    跨越较大时间间隔时不填，避免引入错误。
    """
    out = []
    for name, g in df.groupby("name", sort=False):
        g = g.sort_values("time").copy()
        dt = g["time"].diff().dt.total_seconds() / 3600.0
        for c in TERRAIN_COLS:
            if c not in g:
                continue
            filled = g[c].ffill()
            # 只在时间间隔较小处填充
            too_far = dt > max_gap_h
            filled[too_far] = np.nan
            g[c] = filled
        out.append(g)
    return pd.concat(out, ignore_index=True)
