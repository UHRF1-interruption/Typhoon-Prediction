"""ARCO-ERA5 本地文件解析：从下载的 zarr 分块中提取台风环境场特征。

文件格式（ARCO-ERA5 公开数据集的分块命名约定）
------------------------------------
下载目录结构:
    era5_data/
    ├── geopotential/          <time_idx>.0.0.0   13 层 × 721 × 1440, float32, blosc 压缩
    ├── u_component_of_wind/   <time_idx>.0.0.0
    ├── v_component_of_wind/   <time_idx>.0.0.0
    ├── sea_surface_temperature/  <time_idx>.0.0.0   单层
    ├── mean_sea_level_pressure/  <time_idx>.0.0.0   单层
    └── level/0

关键换算
--------
* **时间**: time_idx 对应 1959-01-01 00:00 UTC + time_idx × 6 小时
* **气压层**: 分块内含全部 13 层，顺序为
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa
  因此 500hPa 在解压数组的第 7 个位置（200hPa=3, 850hPa=10）
* **网格**: 全球 0.25°。
    - 纬度: 第 0 行是 90°N，向南递减 => row = (90 - lat) / 0.25
    - 经度: 第 0 列是 0°E，向东递增 => col = (lon % 360) / 0.25
* **单位**: geopotential 是 m²/s²，需除以 9.80665 转成位势米；
  sst 是开尔文，需减 273.15；msl 是帕斯卡，需除以 100 转 hPa。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .environment import ENV_FEATURES

# --- 常量（来自 ARCO-ERA5 数据集元信息） ---
TIME_ORIGIN = dt.datetime(1959, 1, 1, 0, 0)
STEP_HOURS = 6

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
IDX_200 = LEVELS.index(200)
IDX_500 = LEVELS.index(500)
IDX_850 = LEVELS.index(850)

# 网格
LAT_START, LAT_STEP, N_LAT = 90.0, -0.25, 721
LON_START, LON_STEP, N_LON = 0.0, 0.25, 1440

G0 = 9.80665          # 重力加速度
BOX_HALF_DEG = 10.0   # 计算环境场时取台风中心周围 ±10° 的箱体平均

# 变量目录名
VAR_DIRS = {
    "geopotential": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "sst": "sea_surface_temperature",
    "msl": "mean_sea_level_pressure",
}


# ---------------------------------------------------------------------------
# 时间索引
# ---------------------------------------------------------------------------
def time_to_index(when: dt.datetime) -> int:
    """真实时间 -> time_idx（向下对齐到 6 小时格点）。"""
    when = when.replace(tzinfo=None)
    delta = when - TIME_ORIGIN
    return int(delta.total_seconds() // (STEP_HOURS * 3600))


def index_to_time(idx: int) -> dt.datetime:
    return TIME_ORIGIN + dt.timedelta(hours=int(idx) * STEP_HOURS)


# ---------------------------------------------------------------------------
# 网格索引
# ---------------------------------------------------------------------------
def latlon_to_rc(lat: float, lon: float) -> tuple[int, int]:
    """经纬度 -> 网格行列。超出范围时夹取到边界。"""
    row = int(round((LAT_START - lat) / abs(LAT_STEP)))
    col = int(round(((lon % 360.0) - LON_START) / LON_STEP))
    row = min(max(row, 0), N_LAT - 1)
    col = col % N_LON
    return row, col


def _box_slice(lat: float, lon: float, half_deg: float = BOX_HALF_DEG):
    """返回以 (lat, lon) 为中心的箱体切片。经度跨 180° 时做环绕处理。"""
    half_rows = max(1, int(half_deg / abs(LAT_STEP)))
    half_cols = max(1, int(half_deg / LON_STEP))
    r0, c0 = latlon_to_rc(lat, lon)
    rs = slice(max(0, r0 - half_rows), min(N_LAT, r0 + half_rows + 1))
    # 经度需要环绕：用一个索引数组处理跨界
    cols = (np.arange(c0 - half_cols, c0 + half_cols + 1) % N_LON)
    return rs, cols, r0, c0


# ---------------------------------------------------------------------------
# 分块读取
# ---------------------------------------------------------------------------
_DECODER = None


def _decoder():
    """blosc 解码器（延迟初始化，避免无依赖时导入失败）。"""
    global _DECODER
    if _DECODER is None:
        from numcodecs import Blosc
        _DECODER = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE, blocksize=0)
    return _DECODER


def read_chunk(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """读取并解压一个 zarr 分块文件。"""
    raw = path.read_bytes()
    data = _decoder().decode(raw)
    expected = int(np.prod(shape)) * 4      # float32
    if len(data) != expected:
        raise ValueError(f"{path.name}: 解压后 {len(data)} 字节，期望 {expected}（形状 {shape}）")
    return np.frombuffer(data, dtype="<f4").reshape(shape)


@dataclass
class ARCOPaths:
    """ARCO 下载目录布局。"""
    root: Path

    def chunk(self, var: str, idx: int) -> Path:
        # 多层变量分块 4 段 <idx>.0.0.0；单层变量 3 段 <idx>.0.0
        multi_level = var in ("geopotential", "u", "v")
        name = f"{idx}.0.0.0" if multi_level else f"{idx}.0.0"
        return self.root / VAR_DIRS[var] / name

    def has(self, var: str, idx: int) -> bool:
        return self.chunk(var, idx).exists()

    def available_indices(self, var: str = "geopotential") -> set[int]:
        d = self.root / VAR_DIRS[var]
        if not d.exists():
            return set()
        out = set()
        for p in d.iterdir():
            parts = p.name.split(".")
            if len(parts) >= 1 and parts[0].isdigit():
                out.add(int(parts[0]))
        return out


# ---------------------------------------------------------------------------
# 单时次环境特征
# ---------------------------------------------------------------------------
def compute_env_vector(paths: ARCOPaths, idx: int, lat: float, lon: float,
                       half_deg: float = BOX_HALF_DEG) -> np.ndarray:
    """计算单个时次、单个位置的环境特征向量（顺序同 ENV_FEATURES）。

    缺失的变量填 NaN，由上层统一处理。
    """
    feats = dict.fromkeys(ENV_FEATURES, np.nan)

    # 位势高度（500hPa）
    p = paths.chunk("geopotential", idx)
    if p.exists():
        z = read_chunk(p, (len(LEVELS), N_LAT, N_LON))
        h500 = z[IDX_500] / G0                     # m²/s² -> 位势米
        rs, cols, r0, c0 = _box_slice(lat, lon, half_deg)
        box = h500[rs][:, cols]
        feats["h500_mean"] = float(np.nanmean(box))
        # 经向梯度：箱体南北边界的平均高度差 / 纬度跨度
        north_row = max(0, r0 - max(1, int(half_deg / abs(LAT_STEP))))
        south_row = min(N_LAT - 1, r0 + max(1, int(half_deg / abs(LAT_STEP))))
        h_n = float(np.nanmean(h500[north_row][cols]))
        h_s = float(np.nanmean(h500[south_row][cols]))
        span_deg = abs(north_row - south_row) * abs(LAT_STEP)
        feats["h500_grad_lat"] = (h_n - h_s) / span_deg if span_deg > 0 else np.nan
        # 到副高脊线（箱体内高度最大所在纬度）的距离
        prof = np.nanmean(h500[rs][:, cols], axis=1)
        if np.isfinite(prof).any():
            ridge_row = rs.start + int(np.nanargmax(prof))
            feats["bst_dist"] = abs(ridge_row - r0) * abs(LAT_STEP)

    # 500hPa 风
    up = paths.chunk("u", idx)
    vp = paths.chunk("v", idx)
    if up.exists() and vp.exists():
        cu = read_chunk(up, (len(LEVELS), N_LAT, N_LON))
        cv = read_chunk(vp, (len(LEVELS), N_LAT, N_LON))
        rs, cols, _, _ = _box_slice(lat, lon, half_deg)
        u = float(np.nanmean(cu[IDX_500][rs][:, cols]))
        v = float(np.nanmean(cv[IDX_500][rs][:, cols]))
        feats["u500"], feats["v500"] = u, v
        feats["speed500"] = float(np.hypot(u, v))
        # 垂直风切变：200hPa 与 850hPa 的风矢量差
        u2 = float(np.nanmean(cu[IDX_200][rs][:, cols]))
        v2 = float(np.nanmean(cv[IDX_200][rs][:, cols]))
        u8 = float(np.nanmean(cu[IDX_850][rs][:, cols]))
        v8 = float(np.nanmean(cv[IDX_850][rs][:, cols]))
        feats["shear"] = float(np.hypot(u2 - u8, v2 - v8))

    # 海表温度（开尔文 -> 摄氏度）
    sp = paths.chunk("sst", idx)
    if sp.exists():
        sst = read_chunk(sp, (N_LAT, N_LON))
        rs, cols, _, _ = _box_slice(lat, lon, half_deg)
        feats["sst"] = float(np.nanmean(sst[rs][:, cols])) - 273.15

    # 海平面气压（帕 -> 百帕）
    mp = paths.chunk("msl", idx)
    if mp.exists():
        msl = read_chunk(mp, (N_LAT, N_LON))
        rs, cols, _, _ = _box_slice(lat, lon, half_deg)
        feats["msl"] = float(np.nanmean(msl[rs][:, cols])) / 100.0

    return np.array([feats[k] for k in ENV_FEATURES], dtype=np.float32)


def build_env_table(paths: ARCOPaths, tracks, verbose: bool = True):
    """对轨迹表逐点提取环境特征，返回与轨迹对齐的长表。

    tracks 需含列: sid, time, lat, lon
    只有存在对应时次分块的轨迹点才会被计算；其余跳过。
    """
    import pandas as pd

    tracks = tracks.copy()
    tracks["time"] = pd.to_datetime(tracks["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    available = paths.available_indices("geopotential")
    if not available:
        raise FileNotFoundError(
            f"{paths.root / VAR_DIRS['geopotential']} 下没有分块文件，"
            "请确认 ERA5 分块已按 ARCO-ERA5 的命名规则放到该目录")
    if verbose:
        lo, hi = min(available), max(available)
        print(f"[arco] 可用时次 {len(available)} 个: "
              f"{index_to_time(lo):%Y-%m-%d %H:%M} ~ {index_to_time(hi):%Y-%m-%d %H:%M}")

    # 只处理有数据的时次
    tracks["_idx"] = tracks["time"].map(lambda t: time_to_index(t.to_pydatetime()))
    before = len(tracks)
    tracks = tracks[tracks["_idx"].isin(available)].reset_index(drop=True)
    if verbose:
        print(f"[arco] 轨迹点 {before} -> {len(tracks)}（有环境数据的）")
    if tracks.empty:
        raise ValueError("没有轨迹点能匹配到已下载的 ERA5 时次，检查时间范围是否一致")

    # 同一时次内按经纬度去重，避免重复解压大文件（这是性能关键）
    cache_key = list(zip(tracks["_idx"], tracks["lat"].round(0), tracks["lon"].round(0)))
    tracks["_key"] = cache_key
    unique_keys = tracks["_key"].drop_duplicates()
    if verbose:
        print(f"[arco] 需计算 {len(unique_keys)} 个 (时次, 位置) 组合")

    vec_cache: dict = {}
    for i, (idx, lat, lon) in enumerate(unique_keys, 1):
        try:
            vec_cache[(idx, lat, lon)] = compute_env_vector(paths, int(idx), float(lat), float(lon))
        except Exception as exc:
            if verbose and i <= 5:
                print(f"[arco] 跳过 idx={idx} ({lat},{lon}): {type(exc).__name__}")
            vec_cache[(idx, lat, lon)] = np.full(len(ENV_FEATURES), np.nan, np.float32)
        if verbose and i % 200 == 0:
            print(f"[arco] {i}/{len(unique_keys)}")

    rows = np.stack([vec_cache[k] for k in tracks["_key"]])
    out = pd.DataFrame(rows, columns=list(ENV_FEATURES))
    out.insert(0, "time", tracks["time"].values)
    out.insert(0, "sid", tracks["sid"].values)
    out["lat"] = tracks["lat"].values
    out["lon"] = tracks["lon"].values

    if verbose:
        valid = np.isfinite(rows).all(axis=1).mean()
        print(f"[arco] 完成 {len(out)} 行，完整特征比例 {valid:.1%}")
    return out
