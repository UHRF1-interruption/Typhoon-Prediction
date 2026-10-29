"""地形特征：从 ARCO-ERA5 的静态地表位势与海陆掩膜提取。

为什么要地形
------------
| 机制 | 影响 | 典型例子 |
|---|---|---|
| 山脉阻挡 | 台风穿越时强度骤降 10-20 m/s | 台湾中央山脉(3.9km)、吕宋科迪勒拉(2.9km) |
| 陆地摩擦 | 登陆后持续减弱 | 华南沿海、海南 |
| 地形抬升 | 迎风坡降水暴增 | 台湾、闽浙山区 |
| 海岸线绕流 | 近岸路径偏折 | 台湾海峡、巴士海峡 |

数据来源（ARCO-ERA5 静态场，全数据集仅一份）
-------------------------------------------
  geopotential_at_surface/0.0   float32 (721,1440)  m²/s²  除以 9.80665 得米
  land_sea_mask/0.0             float32 (721,1440)  1=陆地 0=海洋

两者都无时间维，网格与气象场完全一致（0.25°全球）。

特征设计要点
------------
不是简单取平均——台风对地形的响应有强方向性：

  * `terrain_mean`  多半径圈加权平均：反映台风所在环境的总体地形
  * `terrain_max`   **前方**定向扇区内的最大地形：直接反映阻挡强度
  * `blocked`       前方是否有显著高于台风中心的地形（二值化的阻挡信号）
  * `terrain_grad`  地形梯度大小：指示山脉走向，区分"横穿"vs"绕行"
  * `land_frac`     多半径圈内陆地占比：量化登陆程度

「前方」由台风的实际移动方向决定，因此需要传入移动方位角。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .arco_era5 import (
    G0, LAT_START, LAT_STEP, LON_STEP, N_LAT, N_LON, ARCOPaths, latlon_to_rc,
)

# 地形特征（顺序固定）
TERRAIN_FEATURES = [
    "terrain_mean",      # 多半径圈平均地形高度（m）
    "terrain_max",       # 前方扇区内最大地形高度（m）
    "terrain_fwd_mean",  # 前方扇区内平均地形（m）
    "terrain_high_frac",  # 前方 >500m 格点占比（成片山地指标）
    "blocked",           # 前方是否存在显著阻挡（见 compute_terrain_features）
    "terrain_grad",      # 地形梯度（m/deg），山脉陡峭度
    "land_frac",         # 陆地占比
]

# 静态场文件名（无时间维，固定为 0.0）
STATIC_FILE = "0.0"
TERRAIN_VAR = "geopotential_at_surface"
LANDMASK_VAR = "land_sea_mask"

# 前方扇区参数
FWD_HALF_ANGLE_DEG = 60.0    # 前方 ±60° 视为"正前方"
FWD_RADIUS_DEG = 8.0         # 前方观察距离（约 880km）
BLOCK_THRESHOLD_M = 500.0     # 高于此值视为显著地形阻挡

# 多半径圈（与 arco_era5_v2 一致）
RINGS = [(3.0, 0.5), (6.0, 0.3), (10.0, 0.2)]


def _static_path(paths: ARCOPaths, var: str) -> Path:
    return paths.root / var / STATIC_FILE


def load_static(paths: ARCOPaths, var: str, verbose: bool = True) -> np.ndarray:
    """读取静态场（地形或海陆掩膜），返回 (721,1440) 数组。"""
    p = _static_path(paths, var)
    if not p.exists():
        raise FileNotFoundError(
            f"缺少静态场 {p}\n"
            f"请从 ARCO-ERA5 数据集获取该静态场，\n"
            f"这两份静态场可从 ARCO-ERA5 数据集获取（各约 3MB，无时间维）")
    from .arco_era5 import read_chunk
    arr = read_chunk(p, (N_LAT, N_LON))
    if verbose:
        print(f"[terrain] 载入 {var}: 范围 [{np.nanmin(arr):.1f}, {np.nanmax(arr):.1f}]")
    return arr


def load_terrain_fields(paths: ARCOPaths, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """一次性载入地形高度（米）与海陆掩膜。"""
    z = load_static(paths, TERRAIN_VAR, verbose)
    terrain_m = z / G0                       # m²/s² -> 米
    terrain_m = np.maximum(terrain_m, 0.0)   # 海洋处可能为负，归零
    land = load_static(paths, LANDMASK_VAR, verbose)
    return terrain_m, land


def _ring_index(r0: int, c0: int, radius_deg: float, inner_deg: float = 0.0):
    rr = max(1, int(round(radius_deg / abs(LAT_STEP))))
    ri = max(0, int(round(inner_deg / abs(LAT_STEP))))
    rows, cols = [], []
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            d2 = dy * dy + dx * dx
            if d2 > rr * rr or d2 <= ri * ri:
                continue
            r = r0 + dy
            if 0 <= r < N_LAT:
                rows.append(r)
                cols.append((c0 + dx) % N_LON)
    if not rows:
        return np.array([r0]), np.array([c0 % N_LON])
    return np.array(rows), np.array(cols)


def _forward_sector(r0: int, c0: int, bearing_deg: float,
                    half_angle: float = FWD_HALF_ANGLE_DEG,
                    radius: float = FWD_RADIUS_DEG):
    """返回台风前方扇区内的格点索引。

    bearing_deg: 移动方位角，北=0，顺时针（东=90, 南=180, 西=270）。
    注意: 纬度行号随纬度**递减**（第0行是90°N），因此北方向对应 dy<0。
    """
    rr = max(1, int(round(radius / abs(LAT_STEP))))
    has_bearing = bearing_deg is not None and np.isfinite(bearing_deg)
    rows, cols = [], []
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            d2 = dy * dy + dx * dx
            if d2 > rr * rr or d2 == 0:
                continue
            r = r0 + dy
            if not (0 <= r < N_LAT):
                continue
            if has_bearing:
                # 实际方向向量：北为 dy<0，东为 dx>0
                ang = np.degrees(np.arctan2(dx, -dy)) % 360.0
                diff = abs((ang - bearing_deg + 180.0) % 360.0 - 180.0)
                if diff > half_angle:
                    continue
            rows.append(r)
            cols.append((c0 + dx) % N_LON)
    if not rows:
        return np.array([r0]), np.array([c0 % N_LON])
    return np.array(rows), np.array(cols)


def compute_terrain_features(terrain_m: np.ndarray, land: np.ndarray,
                             lat: float, lon: float,
                             bearing_deg: float | None = None) -> np.ndarray:
    """计算单个位置的地形特征向量（顺序同 TERRAIN_FEATURES）。"""
    r0, c0 = latlon_to_rc(lat, lon)
    f = dict.fromkeys(TERRAIN_FEATURES, np.nan)

    # 多半径圈平均地形 + 陆地占比
    vals, wts, land_vals = [], [], []
    prev = 0.0
    for radius, w in RINGS:
        rs, cs = _ring_index(r0, c0, radius, inner_deg=prev)
        t = np.nanmean(terrain_m[rs, cs])
        l = np.nanmean(land[rs, cs])
        if np.isfinite(t):
            vals.append(t)
            wts.append(w)
        if np.isfinite(l):
            land_vals.append(l)
        prev = radius
    if vals:
        f["terrain_mean"] = float(np.average(vals, weights=wts))
    if land_vals:
        f["land_frac"] = float(np.mean(land_vals))

    # 前方扇区
    frs, fcs = _forward_sector(r0, c0, bearing_deg)
    if len(frs):
        ft = terrain_m[frs, fcs]
        f["terrain_max"] = float(np.nanmax(ft))
        f["terrain_fwd_mean"] = float(np.nanmean(ft))
        # 阻挡判定：不能只看单个最高点（远处一座山就会触发），
        # 要求"前方有一片足够高、足够广的地形"。三个条件任一成立即视为阻挡：
        #   (a) 前方平均地形 > 40m —— 整片陆地/丘陵
        #   (b) >500m 的格点占比 > 3% —— 出现成片山地
        #   (c) 前方最高地形 > 800m 且占比 > 1% —— 高耸但范围较小的山脉
        # 阈值依据实测：台湾东侧 ~193m/12.7%，吕宋东侧 ~50m/2.7%，纯洋面 <7m
        high_frac = float(np.nanmean(ft > BLOCK_THRESHOLD_M))
        f["terrain_high_frac"] = high_frac
        blocked = (
            (f["terrain_fwd_mean"] > 40.0)
            or (high_frac > 0.03)
            or (f["terrain_max"] > 800.0 and high_frac > 0.01)
        )
        f["blocked"] = float(blocked)

    # 地形梯度（用中心附近 3° 范围）
    d = 3.0
    dr = max(1, int(round(d / abs(LAT_STEP))))
    dc = max(1, int(round(d / LON_STEP)))
    rn, rs_row = max(0, r0 - dr), min(N_LAT - 1, r0 + dr)
    cs_slice = (np.arange(c0 - dc, c0 + dc + 1)) % N_LON
    dtdy = (np.nanmean(terrain_m[rn, cs_slice]) - np.nanmean(terrain_m[rs_row, cs_slice])) / (2 * d)
    dtdx = (np.nanmean(terrain_m[r0, (c0 + dc) % N_LON]) -
            np.nanmean(terrain_m[r0, (c0 - dc) % N_LON])) / (2 * d)
    f["terrain_grad"] = float(np.hypot(dtdx, dtdy))

    return np.array([f[k] for k in TERRAIN_FEATURES], dtype=np.float32)


def build_terrain_table(paths: ARCOPaths, tracks, verbose: bool = True):
    """对轨迹逐点提取地形特征（需台风移动方向）。

    tracks 需含: sid, time, lat, lon
    移动方向由相邻时刻的位移计算；缺少下一时刻的用上一时刻近似。
    """
    import pandas as pd

    from .features import bearing_deg

    tracks = tracks.copy()
    tracks["time"] = pd.to_datetime(tracks["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    tracks = tracks.sort_values(["sid", "time"]).reset_index(drop=True)

    terrain_m, land = load_terrain_fields(paths, verbose)

    # 先算每个点的移动方位角（用下一时刻位置；末点沿用前一段方向）
    # 关键: 结果必须按 tracks 的原始行序写回，不能按分组顺序拼接
    n_total = len(tracks)
    brg_all = np.full(n_total, np.nan, dtype=float)
    starts = tracks.groupby("sid", sort=False).indices   # sid -> 行索引数组
    for sid, pos in starts.items():
        pos = np.asarray(pos)
        lat = tracks["lat"].to_numpy(float)[pos]
        lon = tracks["lon"].to_numpy(float)[pos]
        n = len(pos)
        if n == 0:
            continue
        nxt_lat = np.concatenate([lat[1:], lat[-1:]])
        nxt_lon = np.concatenate([lon[1:], lon[-1:]])
        b = bearing_deg(lat, lon, nxt_lat, nxt_lon) if n > 1 else np.full(n, np.nan)
        if n > 1:
            b[-1] = b[-2]
        brg_all[pos] = b

    rows = []
    lat_all = tracks["lat"].to_numpy(float)
    lon_all = tracks["lon"].to_numpy(float)
    for i in range(n_total):
        rows.append(compute_terrain_features(terrain_m, land, lat_all[i], lon_all[i], brg_all[i]))
        if verbose and (i + 1) % 1000 == 0:
            print(f"[terrain] {i + 1}/{n_total}", flush=True)

    arr = np.stack(rows)
    out = pd.DataFrame(arr, columns=TERRAIN_FEATURES)
    out.insert(0, "time", tracks["time"].values)
    out.insert(0, "sid", tracks["sid"].values)
    out["lat"] = lat_all
    out["lon"] = lon_all
    if verbose:
        print(f"[terrain] 完成 {len(out)} 行")
    return out
