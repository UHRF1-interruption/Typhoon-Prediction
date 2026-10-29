"""环境场数据：ERA5 下载、缓存、以及无凭据时的物理模拟。

为什么需要环境场
----------------
纯轨迹模型 24h 路径误差约 208km，业务水平约 55km。差距的物理根源是：
**台风转向由大尺度引导气流决定，而引导气流无法从台风自身的历史轨迹推出。**

关键环境要素（业务模式都在用）:
  * 500hPa 位势高度场 —— 副高强度与位置，决定台风能否北上
  * 500hPa 纬向/经向风 (u, v) —— 引导气流，深对流层平均引导最有效
  * 垂直风切变 (200-850hPa) —— 切变大则台风被剪切、路径也易偏
  * 海表温度 SST —— 影响强度；异常暖水区会改变路径
  * 海平面气压 —— 副高边界

两条数据路径
------------
1. `fetch_era5()`: 用 cdsapi 下载真实 ERA5。需要 ~/.cdsapirc 里的 API key。
   首次运行前请到 https://cds.climate.copernicus.eu 注册并同意 ERA5 许可。
2. `synthesize_environment()`: 无凭据时生成**物理合理**的模拟环境场，
   用于打通整条流水线、验证模型结构。注意：用模拟场得到的误差数字
   **不能当作科学结论**，它只证明"接了环境场之后管线能跑、指标会动"。

两者输出的数据结构完全一致，因此模型代码不需要任何改动就能切换。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# 环境特征（每个时空点）
ENV_FEATURES = [
    "h500_mean",        # 500hPa 高度场平均（副高强度指标）
    "h500_grad_lat",    # 高度场经向梯度（引导气流强度代理）
    "u500",             # 500hPa 纬向风
    "v500",             # 500hPa 经向风
    "speed500",         # 引导气流速度
    "shear",            # 垂直风切变（200-850hPa）
    "sst",              # 海表温度
    "msl",              # 海平面气压
    "bst_dist",         # 到副高脊线的距离（转向关键指标）
]

# 计算环境场时，以台风中心为中心取多大范围的平均
BOX_HALF_DEG = 10.0


@dataclass
class ERA5Request:
    """一次 ERA5 下载请求的配置。"""
    out_dir: Path
    years: list[int] = field(default_factory=list)
    months: list[int] = field(default_factory=list)
    # 单层变量
    single_vars: tuple[str, ...] = ("mean_sea_level_pressure", "sea_surface_temperature")
    # 气压层变量
    pressure_vars: tuple[str, ...] = ("geopotential", "u_component_of_wind",
                                      "v_component_of_wind")
    pressure_levels: tuple[int, ...] = (500, 200, 850)
    bbox: tuple[float, float, float, float] = (90.0, -5.0, 200.0, 50.0)  # N,W,S,E
    time_step_h: int = 6


def fetch_era5(req: ERA5Request, verbose: bool = True) -> list[Path]:
    """用 cdsapi 下载 ERA5 数据，按月存成 netcdf。

    需要先安装 cdsapi 并配置 ~/.cdsapirc:
        url: https://cds.climate.copernicus.eu/api
        key: <你的 API key>
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "缺少 cdsapi。安装: .venv/bin/python -m pip install "
            "-i https://mirrors.cernet.edu.cn/pypi/web/simple cdsapi"
        ) from exc

    req.out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    north, west, south, east = req.bbox[3], req.bbox[1], req.bbox[2], req.bbox[0]
    area = [north, west, south, east]
    hours = [f"{h:02d}:00" for h in range(0, 24, req.time_step_h)]
    outputs = []

    for year in req.years:
        for month in req.months:
            tag = f"era5_{year}{month:02d}.nc"
            target = req.out_dir / tag
            if target.exists():
                if verbose:
                    print(f"[era5] 已存在，跳过 {tag}")
                outputs.append(target)
                continue

            # 气压层数据
            pl_target = req.out_dir / f"era5_pl_{year}{month:02d}.nc"
            if not pl_target.exists():
                if verbose:
                    print(f"[era5] 下载气压层 {year}-{month:02d} ...")
                client.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": list(req.pressure_vars),
                        "pressure_level": [str(p) for p in req.pressure_levels],
                        "year": str(year), "month": f"{month:02d}",
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": hours,
                        "area": area,
                        "format": "netcdf",
                    },
                    str(pl_target),
                )
            outputs.append(pl_target)

            # 单层数据
            sl_target = req.out_dir / f"era5_sl_{year}{month:02d}.nc"
            if not sl_target.exists():
                if verbose:
                    print(f"[era5] 下载单层 {year}-{month:02d} ...")
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": list(req.single_vars),
                        "year": str(year), "month": f"{month:02d}",
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": hours,
                        "area": area,
                        "format": "netcdf",
                    },
                    str(sl_target),
                )
            outputs.append(sl_target)

    if verbose:
        print(f"[era5] 完成，共 {len(outputs)} 个文件 -> {req.out_dir}")
    return outputs


def extract_point_env(ds_pl, ds_sl, time, lat, lon,
                      box_half_deg: float = BOX_HALF_DEG) -> np.ndarray:
    """从 ERA5 数据集里取出以 (lat, lon) 为中心、边长 2*box_half_deg 的箱体平均环境量。

    返回长度 len(ENV_FEATURES) 的向量。
    """
    import xarray as xr  # noqa: F401

    def box_mean(da) -> float:
        lat_name = "latitude" if "latitude" in da.dims else "lat"
        lon_name = "longitude" if "longitude" in da.dims else "lon"
        sub = da.sel({lat_name: slice(lat + box_half_deg, lat - box_half_deg),
                      lon_name: slice(lon - box_half_deg, lon + box_half_deg)})
        if sub.size == 0:
            sub = da
        val = sub.mean(skipna=True).values
        return float(np.nanmean(val))

    def at(x, y):
        lat_name = "latitude" if "latitude" in x.dims else "lat"
        lon_name = "longitude" if "longitude" in x.dims else "lon"
        return x.sel({lat_name: y, lon_name: x}, method="nearest").values

    # 位势高度 -> 高度（米）
    h500 = ds_pl["z"].sel(pressure_level=500, method="nearest") / 9.80665
    h_lat = "latitude" if "latitude" in h500.dims else "lat"
    h_mean = box_mean(h500)
    # 经向梯度：箱体南北边界的高度差
    h_n = float(np.nanmean(h500.sel({h_lat: lat + box_half_deg}, method="nearest").values))
    h_s = float(np.nanmean(h500.sel({h_lat: lat - box_half_deg}, method="nearest").values))
    h_grad = (h_n - h_s) / (2 * box_half_deg)

    u500 = box_mean(ds_pl["u"].sel(pressure_level=500, method="nearest"))
    v500 = box_mean(ds_pl["v"].sel(pressure_level=500, method="nearest"))
    shear = _shear(ds_pl)

    sst = box_mean(ds_sl["sst"]) - 273.15 if "sst" in ds_sl else np.nan
    msl = box_mean(ds_sl["msl"]) / 100.0 if "msl" in ds_sl else np.nan

    # 到副高脊线距离：粗略用 500hPa 高度最大处的纬度差
    hbox = h500.sel({h_lat: slice(lat + box_half_deg, lat - box_half_deg)})
    try:
        ridge_lat = float(hbox.mean(dim=[d for d in hbox.dims if d != h_lat])[h_lat].values[
            int(np.nanargmax(hbox.mean(dim=[d for d in hbox.dims if d != h_lat]).values))])
        bst_dist = abs(lat - ridge_lat)
    except Exception:
        bst_dist = np.nan

    return np.array([h_mean, h_grad, u500, v500, np.hypot(u500, v500),
                     shear, sst, msl, bst_dist], dtype=np.float32)


def _shear(ds_pl) -> float:
    """200hPa 与 850hPa 风速差（垂直风切变）。"""
    try:
        u200 = ds_pl["u"].sel(pressure_level=200, method="nearest").mean(skipna=True).values
        v200 = ds_pl["v"].sel(pressure_level=200, method="nearest").mean(skipna=True).values
        u850 = ds_pl["u"].sel(pressure_level=850, method="nearest").mean(skipna=True).values
        v850 = ds_pl["v"].sel(pressure_level=850, method="nearest").mean(skipna=True).values
        return float(np.hypot(u200 - u850, v200 - v850))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# 无凭据时的物理模拟环境场
# ---------------------------------------------------------------------------
def synthesize_environment(track: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """基于台风动力学规则生成物理合理的模拟环境场。

    建模思路（简化但符合定性规律）:
      * 5800m 的 500hPa 高度线大致对应副高边缘；台风越靠近这条线越容易转向；
      * 低纬（<20°N）引导气流偏东风（副高南侧），高纬转为偏西风；
      * 海温随纬度下降，暖水区（西北太平洋 20-30°N, 130-150°E）偏高；
      * 切变在副高边缘和低纬较小，季风槽区域较大。

    注意: 这是**规则生成的合成场**，用于验证管线。真实结论必须用 ERA5。
    """
    rng = np.random.default_rng(seed)
    out = track.copy()
    lat = out["lat"].to_numpy(float)
    lon = out["lon"].to_numpy(float)

    # 副高：中心约在 25N / 145E，越远离中心高度越低
    ridge_lat = 25.0 + 3.0 * np.sin(np.radians(lon * 2))
    h500 = 5880.0 - 18.0 * np.abs(lat - ridge_lat) + rng.normal(0, 8, len(lat))
    h_grad = np.where(lat < 20, 6.0, np.where(lat < 30, 10.0, 4.0)) + rng.normal(0, 1.5, len(lat))

    # 引导气流：低纬东风、高纬西风
    u500 = np.where(lat < 18, -6.5, np.where(lat < 28, -3.0, 2.5)) + rng.normal(0, 1.2, len(lat))
    v500 = 0.8 * (lat - 20.0) / 10.0 + rng.normal(0, 0.8, len(lat))

    # 切变：副高边缘与季风槽较大
    shear = 4.0 + 3.0 * np.abs(lat - 22.0) / 10.0 + rng.normal(0, 1.0, len(lat))

    # 海温：随纬度递减，暖池偏高
    sst = 30.0 - 0.22 * np.maximum(lat - 15.0, 0) + rng.normal(0, 0.4, len(lat))
    warm_pool = np.exp(-((lat - 25.0) / 10.0) ** 2 - ((lon - 140.0) / 18.0) ** 2)
    sst += 1.2 * warm_pool

    msl = 1010.0 - 4.0 * np.abs(lat - ridge_lat) / 10.0 + rng.normal(0, 1.0, len(lat))
    bst_dist = np.abs(lat - ridge_lat)

    env = pd.DataFrame({
        "time": out["time"].values,
        "lat": lat, "lon": lon,
        "h500_mean": h500, "h500_grad_lat": h_grad,
        "u500": u500, "v500": v500, "speed500": np.hypot(u500, v500),
        "shear": shear, "sst": sst, "msl": msl, "bst_dist": bst_dist,
    })
    return env


def attach_env_to_tracks(tracks: pd.DataFrame, env: pd.DataFrame,
                         time_tol_h: float = 3.0) -> pd.DataFrame:
    """把环境场的逐时次数值按 (sid, time) 对齐到轨迹上。

    轨迹是每 3h/6h 一条，环境场可能采样更密，用最近邻 + 时间容差匹配。
    """
    out = tracks.copy()
    env = env.copy()
    env["time"] = pd.to_datetime(env["time"])
    out["time"] = pd.to_datetime(out["time"])

    merged = []
    for sid, group in out.groupby("sid"):
        sub_env = env
        if "sid" in env.columns:
            sub_env = env[env["sid"] == sid]
        if sub_env.empty:
            merged.append(group)
            continue
        g = group.sort_values("time")
        e = sub_env.sort_values("time")
        # 最近邻时间匹配（环境场是逐点生成的，这里每行都有精确对应时刻）
        g = pd.merge_asof(g, e.drop(columns=["sid"], errors="ignore"),
                          on="time", direction="nearest",
                          tolerance=pd.Timedelta(hours=time_tol_h))
        merged.append(g)
    return pd.concat(merged, ignore_index=True)
