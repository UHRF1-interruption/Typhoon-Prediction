"""卫星云图强度预测：数据构造 + CNN-Transformer 模型。

任务：用过去 N 步（默认 8 步 × 3h = 24h）的
      - 轨迹/强度标量序列（中心经纬度、气压、风速、移动速度、方位角、切向加速度）
      - 每个时次的台风中心云图 patch（4 通道，可选）
预测未来 M 步（默认 4 步 × 3h = 12h）的强度（最大风速 m/s）。

为什么这样设计：
  * 路径预测受引导气流控制，纯轨迹学不到；但强度变化与云图结构（眼区、
    对称性、对流强度）直接相关，卫星通道能提供额外信息量；
  * Transformer 处理时间序列（跨步注意力），CNN 提取每帧云图的空间特征，
    两者融合后回归强度——这是「云图 + 时序」的常见可行结构。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import bearing_deg, haversine_km, wrap_dlon

# 标量特征（每个时间步）
SCALAR_FEATURES = [
    "wind", "prs", "lat", "lon", "move_km", "sin_heading", "cos_heading",
    "wind_delta",
]

# 地形特征（可选加入）。使用时由 terrain_align.py 对齐到 scalars 表，
# 并通过 SatConfig.use_terrain=True 启用。
TERRAIN_FEATURES = [
    "terrain_mean", "terrain_max", "terrain_fwd_mean",
    "terrain_high_frac", "blocked", "terrain_grad", "land_frac",
]

# 暖心结构特征（可选）。由 src/data/warm_core.py 从红外 patch 反演，
# 通过 SatConfig.use_warmcore=True 启用；需在 scalars 表中预先提供这些列。
WARMCORE_FEATURES = [
    "wc_center_tb", "wc_ring_tb", "wc_eye_diff", "wc_min_tb",
    "wc_cold_frac", "wc_cdo_frac", "wc_grad", "wc_asym",
    "wc_center_r", "wc_contrast",
]


@dataclass
class SatConfig:
    in_len: int = 8           # 过去 8 步（3h 一步 = 24h）
    out_len: int = 4          # 未来 4 步（12h）
    interval_h: float = 3.0
    min_wind: float = 10.0
    patch_size: int = 32      # 云图 patch 边长发（4km/像素 -> 128km）
    use_patch: bool = True
    use_terrain: bool = False  # 是否把地形特征加入标量输入
    use_warmcore: bool = False  # 是否把红外暖心结构特征加入标量输入


def regularize_intensity(seq: pd.DataFrame, cfg: SatConfig) -> pd.DataFrame:
    """把单台风的时次序列重采样到固定 3h 间隔。

    seq 列: time, wind, prs, lat, lon
    """
    s = seq.sort_values("time").set_index("time")
    s = s[~s.index.duplicated(keep="first")]
    if len(s) < 2:
        return pd.DataFrame()
    freq = pd.Timedelta(hours=cfg.interval_h)
    idx = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    if len(idx) < 2:
        return pd.DataFrame()
    # 保留轨迹必需列 + 任何额外数值列（地形特征等），否则后续特征会丢
    base_cols = ["wind", "prs", "lat", "lon"]
    extra_cols = [c for c in s.columns
                  if c not in base_cols and pd.api.types.is_numeric_dtype(s[c])]
    out = s[base_cols + extra_cols].reindex(idx)
    # 只保留前后都有真实观测的时次，避免长间隔插值制造假数据
    real = s.index.values.astype("datetime64[ns]")
    r = out.index.values.astype("datetime64[ns]")
    pos = np.searchsorted(real, r)
    slot = np.timedelta64(int(cfg.interval_h * 3600), "s")
    ok = np.zeros(len(r), dtype=bool)
    for i, t in enumerate(r):
        p = pos[i]
        near_before = p > 0 and (t - real[p - 1]) <= slot * 1.5
        near_after = p < len(real) and (real[min(p, len(real) - 1)] - t) <= slot * 1.5
        ok[i] = bool(near_before or near_after)
    out = out[ok]
    if len(out) < 2:
        return pd.DataFrame()
    out = out.interpolate(limit_direction="both")
    return out.reset_index().rename(columns={"index": "time"})


def add_intensity_features(df: pd.DataFrame, interval_h: float) -> pd.DataFrame:
    """逐点特征：移动距离、方位角、强度变化率。"""
    t = df.copy()
    lat = t["lat"].to_numpy(float)
    lon = t["lon"].to_numpy(float)
    prev_lat = np.concatenate([[lat[0]], lat[:-1]])
    prev_lon = np.concatenate([[lon[0]], lon[:-1]])
    dist = haversine_km(prev_lat, prev_lon, lat, lon)
    dist[0] = 0.0
    heading = bearing_deg(prev_lat, prev_lon, lat, lon)
    heading[0] = heading[1] if len(heading) > 1 else 0.0
    wind = t["wind"].to_numpy(float)
    wind_delta = np.concatenate([[0.0], np.diff(wind)])

    t["move_km"] = dist
    t["sin_heading"] = np.sin(np.radians(heading))
    t["cos_heading"] = np.cos(np.radians(heading))
    t["wind_delta"] = wind_delta
    t["lon"] = np.mod(lon, 360.0)
    return t


def build_intensity_samples(scalars: pd.DataFrame, cfg: SatConfig,
                            patches: dict | None = None) -> dict:
    """构造强度预测样本。

    scalars: 含 name, time, wind_spd, cent_prs, cent_lat, cent_lon 的逐时次表。
             若 cfg.use_terrain=True，还需含 TERRAIN_FEATURES 中的列
             （用 terrain_align.align_terrain_to_satellites 生成）。
    patches: {(name, time) -> (4, S, S)}，cfg.use_patch=True 时必须提供
    返回:
      s     (N, in_len, F) 标量特征
      p     (N, in_len, 4, S, S) 云图 patch（可选）
      y     (N, out_len) 未来强度 m/s
      last_wind (N,) 最后观测强度（残差基准）
      name  (N,) 台风名
    """
    # 同一时次多卫星取均值；地形列若存在也一并保留
    agg_spec = {"wind": ("wind_spd", "mean"), "prs": ("cent_prs", "mean"),
                "lat": ("cent_lat", "mean"), "lon": ("cent_lon", "mean")}
    terr_cols = []
    if cfg.use_terrain:
        terr_cols = [c for c in TERRAIN_FEATURES if c in scalars.columns]
        if not terr_cols:
            raise ValueError(
                "cfg.use_terrain=True 但 scalars 里没有地形列。"
                "请先用 src/data/terrain_align.py 的 align_terrain_to_satellites() 对齐")
        for c in terr_cols:
            agg_spec[c] = (c, "mean")
    wc_cols = []
    if cfg.use_warmcore:
        wc_cols = [c for c in WARMCORE_FEATURES if c in scalars.columns]
        if not wc_cols:
            raise ValueError(
                "cfg.use_warmcore=True 但 scalars 里没有暖心特征列。"
                "请先用 src/data/warm_core.py 的 build_warmcore_table() 生成")
        for c in wc_cols:
            agg_spec[c] = (c, "mean")
    g = scalars.groupby(["name", "time"]).agg(**agg_spec).reset_index()


    S, P, Y, names, times_out = [], [], [], [], []
    for name, seq in g.groupby("name"):
        reg = regularize_intensity(seq, cfg)
        if reg.empty or len(reg) < cfg.in_len + cfg.out_len:
            continue
        reg = add_intensity_features(reg, cfg.interval_h)
        use_cols = list(SCALAR_FEATURES)
        if cfg.use_terrain:
            # 地形列已在 g 中；插值后若整列为空则填 0（表示无地形）
            for c in terr_cols:
                if c in reg.columns:
                    reg[c] = reg[c].fillna(0.0)
                    use_cols.append(c)
        if cfg.use_warmcore:
            for c in wc_cols:
                if c in reg.columns:
                    reg[c] = reg[c].interpolate(limit_direction="both").fillna(0.0)
                    use_cols.append(c)
        feat = reg[use_cols].to_numpy(np.float32)
        wind = reg["wind"].to_numpy(np.float32)
        n = len(reg)
        for start in range(0, n - cfg.in_len - cfg.out_len + 1):
            end_in = start + cfg.in_len
            end_out = end_in + cfg.out_len
            s_win = feat[start:end_in]
            y_win = wind[end_in:end_out]
            if not (np.isfinite(s_win).all() and np.isfinite(y_win).all()):
                continue
            if cfg.use_patch:
                if patches is None:
                    raise ValueError("use_patch=True 时必须提供 patches")
                frame_times = reg["time"].iloc[start:end_in].tolist()
                got = [patches.get((name, ts)) for ts in frame_times]
                if any(g is None for g in got):
                    continue      # 缺帧跳过，保证输入完整
                # 丢掉几乎全空的帧（卫星缺测会造成整幅 patch 为 0）
                if any((np.asarray(g) > 0).mean() < 0.5 for g in got):
                    continue
                S.append(s_win)
                P.append(np.stack(got))
            else:
                S.append(s_win)
            Y.append(y_win)
            names.append(name)
            times_out.append(reg["time"].iloc[end_in - 1])

    if not Y:
        raise ValueError("没有构造出样本：检查 in_len/out_len 与数据长度")

    out = {
        "s": np.stack(S).astype(np.float32),
        "y": np.stack(Y).astype(np.float32),
        "name": np.array(names),
        "end_time": np.array([str(t) for t in times_out]),
    }
    # 最后输入时刻的强度作为残差基准（残差预测更稳）
    out["last_wind"] = out["s"][:, -1, 0].astype(np.float32)
    if cfg.use_patch:
        out["p"] = np.stack(P).astype(np.float32)
        # 预先计算云图手工特征 (N, T, F_img)，供特征版模型使用
        from .patch_features import stack_patch_features
        out["pfeat"] = stack_patch_features(out["p"]).astype(np.float32)
    msg = f"[sat] 强度样本 {len(out['y'])}, 标量 {out['s'].shape}"
    if cfg.use_patch:
        msg += f", 云图 {out['p'].shape}, 云图特征 {out['pfeat'].shape}"
    print(msg)
    return out


def fit_scalar_scaler(s: np.ndarray, feature_names: list[str]) -> dict:
    """只用训练集拟合标量标准化参数。"""
    flat = s.reshape(-1, s.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"names": feature_names, "mean": mean.tolist(), "std": std.tolist()}
