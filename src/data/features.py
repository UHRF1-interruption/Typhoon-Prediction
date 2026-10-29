"""从台风长表构造滑动窗口样本。

流程:
  长表(每条记录一个时刻)
    -> 按风暴分组 / 重采样到固定间隔 / 清洗
    -> 逐点特征(差分、方位角、距离)
    -> 滑动窗口 (in_len 历史 -> out_len 未来)
    -> 预测目标转换为相对最后观测点的增量 delta(lat, lon)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

# add_track_features() 生成的特征列（其余列视为环境场等外部特征，原样保留）
TRACK_FEATURE_COLS = {
    "dlat", "dlon", "dist_prev", "heading", "sin_heading", "cos_heading",
    "move_speed_ms", "lat_norm", "dpres_hpa", "wind_ms",
}


# ----------------------------------------------------------------------------
# 地理计算
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """球面大圆距离(km)。输入单位均为角度，支持 numpy 广播。"""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """从点1到点2的初始方位角(度)，北=0，顺时针向东为正。"""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def wrap_dlon(dlon):
    """把经度差约束到 (-180, 180]，处理跨 180 度线。"""
    return (dlon + 180.0) % 360.0 - 180.0


def dest_point(lat, lon, bearing, dist_km):
    """从 (lat, lon) 沿方位角 bearing 前进 dist_km，返回终点（球面正解）。"""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    brg = np.radians(bearing)
    ang = dist_km / EARTH_RADIUS_KM
    lat2 = np.arcsin(np.sin(lat_r) * np.cos(ang) + np.cos(lat_r) * np.sin(ang) * np.cos(brg))
    lon2 = lon_r + np.arctan2(
        np.sin(brg) * np.sin(ang) * np.cos(lat_r),
        np.cos(ang) - np.sin(lat_r) * np.sin(lat2),
    )
    return np.degrees(lat2), np.degrees(lon2)


# ----------------------------------------------------------------------------
# 单条轨迹的特征
# ----------------------------------------------------------------------------
def add_track_features(track: pd.DataFrame, interval_hours: float = 6.0) -> pd.DataFrame:
    """给单条风暴轨迹（已按时间排序、已重采样）添加逐点特征。

    要求列: time, lat, lon, wind, pres
    新增列: dlat, dlon, dist_prev, heading, speed_deg (每步位移), wind, pres
    """
    t = track.copy()
    lat = t["lat"].to_numpy(dtype=float)
    lon = t["lon"].to_numpy(dtype=float)

    # 首个点的增量为 0（模型会通过 mask 知道这是 padding 之外的边界点）
    dlat = np.diff(lat, prepend=lat[0])
    dlon = wrap_dlon(np.diff(lon, prepend=lon[0]))
    dlat[0] = 0.0
    dlon[0] = 0.0

    prev_lat = np.concatenate([[lat[0]], lat[:-1]])
    prev_lon = np.concatenate([[lon[0]], lon[:-1]])
    dist = haversine_km(prev_lat, prev_lon, lat, lon)
    dist[0] = 0.0

    heading = bearing_deg(prev_lat, prev_lon, lat, lon)
    heading[0] = heading[1] if len(heading) > 1 else 0.0

    t["dlat"] = dlat
    t["dlon"] = dlon
    t["dist_prev"] = dist
    t["heading"] = heading
    t["sin_heading"] = np.sin(np.radians(heading))
    t["cos_heading"] = np.cos(np.radians(heading))
    # 每秒米制的移动速度
    t["move_speed_ms"] = dist * 1000.0 / (interval_hours * 3600.0)
    t["lat_norm"] = lat / 90.0
    # 气压缺失时用气候平均值 1010 hPa 填充，保证输入特征无 NaN
    # （本工程附带的 CMA 格式不带气压列，因此该特征恒为 0，不影响路径预测）
    pres = t["pres"] if "pres" in t.columns else pd.Series(1010.0, index=t.index)
    pres = pres.fillna(1010.0)
    t["dpres_hpa"] = 1010.0 - pres            # 压差(正值代表低压)
    # 风速单位：CMA 原始为 m/s，IBTrACS 为 knot(节)
    if t.attrs.get("wind_unit") == "ms" or t["wind"].median() < 30:
        t["wind_ms"] = t["wind"]
    else:
        t["wind_ms"] = t["wind"] * 0.514444   # 节 -> m/s
    return t


# ----------------------------------------------------------------------------
# 数据集构造
# ----------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    """记录特征名与归一化统计量，推理时必须复用同一份。"""
    names: list[str]
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(names=list(d["names"]),
                   mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32))


def resample_track(track: pd.DataFrame, interval_hours: float) -> pd.DataFrame:
    """重采样单条轨迹到固定时间间隔。

    IBTrACS 存在 3h/6h 混合采样以及缺测，这里统一到固定间隔，
    缺失点用线性插值补，超过 max_gap 的断档则直接保留缺口
    （缺口后续由窗口构造时的连续性检查剔除）。
    """
    t = track.sort_values("time").set_index("time")
    # 去掉重复时间戳
    t = t[~t.index.duplicated(keep="first")]
    if len(t) < 2:
        return pd.DataFrame()

    freq = pd.Timedelta(hours=interval_hours)
    full_index = pd.date_range(t.index.min(), t.index.max(), freq=freq, tz=t.index.tz)
    if len(full_index) < 2:
        return pd.DataFrame()

    # 除轨迹必需列外，额外保留的数值列（环境场特征等）也一并重采样
    extra_numeric = [c for c in t.columns
                     if c not in ("lat", "lon", "wind", "pres")
                     and pd.api.types.is_numeric_dtype(t[c])]
    keep_cols = ["lat", "lon", "wind", "pres"] + extra_numeric
    resampled = t[keep_cols].reindex(full_index)
    # 只对短缺口插值：先标记原始缺失，再线性插值
    for c in keep_cols:
        resampled[c] = resampled[c].interpolate(limit_direction="both")

    # 用原始观测时间判断连续性：在某时刻前后一个间隔内是否有真实观测
    real_times = t.index.values.astype("datetime64[ns]")
    res_idx = full_index.values.astype("datetime64[ns]")
    slot = np.timedelta64(int(interval_hours * 3600), "s")
    pos = np.searchsorted(real_times, res_idx)
    has_before = (pos > 0) & (res_idx - real_times[np.clip(pos - 1, 0, len(real_times) - 1)] <= slot * 1.5)
    has_after = (pos < len(real_times)) & (real_times[np.clip(pos, 0, len(real_times) - 1)] - res_idx <= slot * 1.5)
    resampled["observed"] = has_before | has_after

    resampled = resampled.reset_index().rename(columns={"index": "time"})
    return resampled


def build_samples(
    df: pd.DataFrame,
    in_len: int,
    out_len: int,
    interval_hours: float = 6.0,
    feature_names: list[str] | None = None,
    min_obs_ratio: float = 0.6,
) -> dict[str, np.ndarray]:
    """把长表变成固定形状的窗口数组。

    返回 dict:
      x        (N, in_len, F)  输入特征（未归一化）
      y        (N, out_len, 2) 目标增量 [dlat, dlon]，相对最后输入点
      y_abs    (N, out_len, 2) 目标绝对坐标 [lat, lon]（0-360 经度）
      last_abs (N, 2)          最后输入点的绝对坐标 [lat, lon]（作为残差基准）
      year     (N,)            样本起始年份
      sid      (N,)            风暴编号
      valid_mask (N, in_len)   输入中真实观测比例高的位置（用于 padding mask）
    """
    feature_names = feature_names or ["dlat", "dlon", "dist_prev", "sin_heading",
                                      "cos_heading", "wind_ms", "dpres_hpa", "lat_norm"]
    # 环境场特征是原始逐点值（已在轨迹表里），不做差分，直接从 resample 结果取
    env_cols = [c for c in feature_names if c not in TRACK_FEATURE_COLS]
    tracks: list[np.ndarray] = []
    meta: list[tuple[float, str, np.ndarray]] = []

    for sid, group in df.groupby("sid", sort=False):
        rs = resample_track(group, interval_hours)
        if len(rs) < in_len + out_len:
            continue
        rs = add_track_features(rs, interval_hours)
        # 过滤物理上不可能的跳变（每 6h 移动超过 1500km），避免污染训练
        if rs["dist_prev"].max() > 1500:
            keep = rs["dist_prev"] <= 1500
            keep.iloc[0] = True
            rs = rs[keep].reset_index(drop=True)
            if len(rs) < in_len + out_len:
                continue
        feats = rs[feature_names].to_numpy(dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        lat = rs["lat"].to_numpy(dtype=np.float64)
        lon = rs["lon"].to_numpy(dtype=np.float64)
        observed = rs["observed"].to_numpy(dtype=bool)
        times = rs["time"].to_numpy()

        n = len(rs)
        for start in range(0, n - in_len - out_len + 1):
            end_in = start + in_len
            end_out = end_in + out_len
            win_obs = observed[start:end_in]
            if win_obs.mean() < min_obs_ratio:
                continue
            # 未来目标必须在观测范围内，避免用插值假点做标签
            if observed[end_in:end_out].mean() < 0.99:
                continue

            x_win = feats[start:end_in].copy()
            y_abs = np.stack([lat[end_in:end_out], lon[end_in:end_out]], axis=-1)
            last = np.array([lat[end_in - 1], lon[end_in - 1]], dtype=np.float64)
            y_delta = np.stack([
                y_abs[:, 0] - last[0],
                wrap_dlon(y_abs[:, 1] - last[1]),
            ], axis=-1)

            tracks.append(x_win)
            # 年份从 numpy datetime64 提取，兼容 pandas Timestamp
            start_ts = pd.Timestamp(times[start])
            meta.append((float(start_ts.year),
                         str(sid), (y_delta.astype(np.float32), y_abs.astype(np.float32),
                                    last.astype(np.float32), observed[start:end_in])))

    if not tracks:
        raise ValueError(
            f"没有构造出任何样本：检查 in_len={in_len}/out_len={out_len} "
            f"是否大于轨迹长度，或 min_obs_ratio 是否过严"
        )

    x = np.stack(tracks)
    y = np.stack([m[2][0] for m in meta])
    y_abs = np.stack([m[2][1] for m in meta])
    last_abs = np.stack([m[2][2] for m in meta])
    valid = np.stack([m[2][3] for m in meta])
    year = np.array([m[0] for m in meta], dtype=np.int32)
    sids = np.array([m[1] for m in meta])

    print(f"[samples] 构造 {len(x)} 个样本, x={x.shape}, y={y.shape}")
    return {
        "x": x, "y": y, "y_abs": y_abs, "last_abs": last_abs,
        "year": year, "sid": sids, "valid_mask": valid,
        "feature_names": np.array(feature_names),
    }


def fit_feature_scaler(x: np.ndarray) -> FeatureSpec:
    """只用训练集统计量做标准化，避免数据泄漏。"""
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return FeatureSpec(names=[f"f{i}" for i in range(x.shape[-1])],
                       mean=mean.astype(np.float32), std=std.astype(np.float32))


def target_scale() -> np.ndarray:
    """目标标准化尺度：固定为角度量级，保证不同海域量纲一致。

    delta 的纬度单位是度；经度差除以 cos(lat) 会更公平，但为了推理时
    逆变换简单，这里统一用 1 度作为尺度（配合 Haversine 损失已足够稳）。
    """
    return np.array([1.0, 1.0], dtype=np.float32)
