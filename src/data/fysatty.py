"""风云卫星（FYSATTY）台风数据集读取。

数据由上海台风所（STI/CMA）制作，每个 NetCDF 文件包含一个时次的:
  - 台风中心位置 CentLat / CentLon
  - 最低海平面气压 CentPrs
  - 最大风速 WindSpd
  - 751×751 网格（4km 分辨率）的 4 个卫星通道:
      NOMChannelVIS  0.65 μm 可见光
      NOMChannelIR3  6.95 μm 水汽
      NOMChannelIR1  10.8 μm 红外窗区
      NOMChannelIR2  12.0 μm 红外分裂窗

文件命名: <TCID>.<NAME>.<YYYY>.<MM>.<DD>.<HHMM>.<分钟>.<卫星>.<版本>.<...>.nc
  TCID 形如 2023267N15111 = 2023 年第 267 天、中心位于北纬 15.1 度东经 111 度附近。

用途：强度预测。云图的对称性、眼区清晰度、暖心结构对强度变化有信息量，
      这是纯轨迹序列学不到的部分。

内存说明: 单个文件 751×751×4 float32 ≈ 8.6MB，9574 个文件不能全部载入。
          因此这里只把**中心裁剪后的低分辨率 patch** 与标量字段整理成表，
          完整云图按需从 zip 中读取。
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 云图通道名（按物理意义排序）
# 注意: NOMChannelVIS（0.65μm 可见光）在本数据集中绝大多数时次无有效值
#       （夜间无可见光，且部分卫星文件该通道为空），因此默认不用；
#       红外三通道（6.95/10.8/12.0μm）始终有效，能反映云顶温度与对流结构。
CHANNELS = ["NOMChannelIR3", "NOMChannelIR1", "NOMChannelIR2"]
ALL_CHANNELS = ["NOMChannelVIS"] + CHANNELS
CHANNEL_LABELS = {
    "NOMChannelVIS": "0.65um visible",
    "NOMChannelIR3": "6.95um water vapor",
    "NOMChannelIR1": "10.8um IR window",
    "NOMChannelIR2": "12.0um IR split",
}

# 文件名分段位置
IDX_TCFILE, IDX_NAME, IDX_YEAR, IDX_MONTH, IDX_DAY = 0, 1, 2, 3, 4
IDX_HHMM, IDX_MINUTE, IDX_SAT = 5, 6, 7


@dataclass
class SatFileInfo:
    """压缩包中一个 NetCDF 文件的元信息（不读内容）。"""
    zip_path: str
    member: str
    tcid: str
    name: str
    time: pd.Timestamp
    satellite: str
    minute: int


def list_satellite_files(zip_paths: list[str | Path]) -> pd.DataFrame:
    """扫描所有 zip，返回每个 NetCDF 文件的元信息表（只读文件名，不解压）。"""
    rows = []
    for zp in zip_paths:
        try:
            with zipfile.ZipFile(zp) as zf:
                for full in zf.namelist():
                    if not full.endswith(".nc"):
                        continue
                    parts = full.split("/")[-1].split(".")
                    if len(parts) < 8:
                        continue
                    try:
                        year, month, day = int(parts[IDX_YEAR]), int(parts[IDX_MONTH]), int(parts[IDX_DAY])
                        hhmm = parts[IDX_HHMM]
                        hour, minute = int(hhmm[:2]), int(hhmm[2:]) if len(hhmm) == 4 else int(hhmm[2:4])
                        ts = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
                    except (ValueError, IndexError):
                        continue
                    rows.append({
                        "zip_path": str(zp),
                        "member": full,
                        "tcid": parts[IDX_TCFILE],
                        "name": parts[IDX_NAME],
                        "time": ts,
                        "satellite": parts[IDX_SAT],
                        "minute": int(parts[IDX_MINUTE]) if parts[IDX_MINUTE].isdigit() else 0,
                    })
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"[sat] 跳过损坏/空压缩包 {Path(zp).name}: {type(exc).__name__}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("没有找到任何 NetCDF 文件")
    df = df.sort_values(["name", "time", "satellite"]).reset_index(drop=True)
    years = ", ".join(str(y) for y in sorted(df["time"].dt.year.unique()))
    print(f"[sat] 共 {len(df)} 个文件, {df['name'].nunique()} 个台风, "
          f"卫星 {sorted(df['satellite'].unique())}, 年份 {years}")
    return df


def read_netcdf_member(zip_path: str | Path, member: str):
    """从 zip 直接读取一个 NetCDF 文件（不落盘）。"""
    import xarray as xr

    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(member)
    return xr.open_dataset(io.BytesIO(raw), engine="h5netcdf")


def extract_scalars(zip_path: str | Path, member: str) -> dict:
    """只读取标量字段（中心位置/气压/风速），速度远快于读整张云图。"""
    ds = read_netcdf_member(zip_path, member)
    try:
        out = {
            "cent_lat": float(ds["CentLat"].values),
            "cent_lon": float(ds["CentLon"].values),
            "cent_prs": float(ds["CentPrs"].values),
            "wind_spd": float(ds["WindSpd"].values),
            "vza": float(ds["VZA"].values) if "VZA" in ds else np.nan,
        }
        attrs = {k: str(v) for k, v in ds.attrs.items()}
    finally:
        ds.close()
    out["attrs"] = attrs
    return out


def extract_patch(zip_path: str | Path, member: str, half_size: int = 32,
                  out_size: int | None = 32) -> dict:
    """以台风中心为中心裁剪云图 patch，并可选下采样到 out_size。

    返回 patch: (4, out_size, out_size) float32，NaN 已填为通道均值。
    half_size=32 对应 32×4km ≈ 128km 的半边长，覆盖台风核心区。
    """
    ds = read_netcdf_member(zip_path, member)
    try:
        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
        clat = float(ds["CentLat"].values)
        clon = float(ds["CentLon"].values)
        scalars = {
            "cent_lat": clat, "cent_lon": clon,
            "cent_prs": float(ds["CentPrs"].values),
            "wind_spd": float(ds["WindSpd"].values),
        }

        iy = int(np.argmin(np.abs(lat - clat)))
        ix = int(np.argmin(np.abs(lon - clon)))
        y0, y1 = max(0, iy - half_size), min(len(lat), iy + half_size)
        x0, x1 = max(0, ix - half_size), min(len(lon), ix + half_size)

        chans = []
        for ch in CHANNELS:
            if ch not in ds:
                chans.append(np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32))
                continue
            block = ds[ch].values[y0:y1, x0:x1].astype(np.float32)
            block[block >= 65535] = np.nan          # missing_value
            block[block < 100] = np.nan             # valid_range 下限
            chans.append(block)
        patch = np.stack(chans, axis=0) if chans else np.zeros((len(CHANNELS), 1, 1), np.float32)
        # 记录有效像素比例：全零/几乎无有效像素的帧不应进入训练
        valid_ratio = float(np.isfinite(patch).mean())

        # 统一尺寸 + 均值填充
        if out_size and (patch.shape[1] != out_size or patch.shape[2] != out_size):
            patch = _resize_nearest(patch, out_size)
        for c in range(patch.shape[0]):
            m = np.nanmean(patch[c])
            if not np.isfinite(m):
                m = 0.0
            np.nan_to_num(patch[c], copy=False, nan=m)
    finally:
        ds.close()
    return {"patch": patch, "valid_ratio": valid_ratio, **scalars}


def _resize_nearest(arr: np.ndarray, size: int) -> np.ndarray:
    """最近邻重采样 (C,H,W) -> (C,size,size)。"""
    c, h, w = arr.shape
    yi = (np.arange(size) * (h / size)).astype(int).clip(0, h - 1)
    xi = (np.arange(size) * (w / size)).astype(int).clip(0, w - 1)
    return arr[:, yi][:, :, xi]


def build_scalar_table(files: pd.DataFrame, sample_every: int = 1) -> pd.DataFrame:
    """遍历文件读取标量字段，生成逐时次强度表。

    同一时次可能有多颗卫星的观测，这里保留全部并标记 satellite，
    后续可按时间对齐取均值或选择主卫星。
    """
    rows = []
    subset = files.iloc[::max(1, sample_every)]
    total = len(subset)
    for i, row in enumerate(subset.itertuples(index=False), 1):
        try:
            sc = extract_scalars(row.zip_path, row.member)
        except Exception as exc:
            print(f"[sat] 读取失败 {row.member.split('/')[-1][:40]}: {type(exc).__name__}")
            continue
        rows.append({
            "tcid": row.tcid, "name": row.name, "time": row.time,
            "satellite": row.satellite, "minute": row.minute,
            "cent_lat": sc["cent_lat"], "cent_lon": sc["cent_lon"],
            "cent_prs": sc["cent_prs"], "wind_spd": sc["wind_spd"],
            "vza": sc["vza"],
        })
        if i % 500 == 0:
            print(f"[sat] 已读取 {i}/{total}")
    df = pd.DataFrame(rows)
    print(f"[sat] 标量表 {len(df)} 行, {df['name'].nunique()} 个台风")
    return df
