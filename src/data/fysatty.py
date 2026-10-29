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
    df = _attach_storm_key(df)
    df = df.sort_values(["storm_key", "time", "satellite"]).reset_index(drop=True)
    years = ", ".join(str(y) for y in sorted(df["time"].dt.year.unique()))
    print(f"[sat] 共 {len(df)} 个文件, {df['storm_key'].nunique()} 个台风, "
          f"卫星 {sorted(df['satellite'].unique())}, 年份 {years}")
    return df


def _attach_storm_key(df: pd.DataFrame) -> pd.DataFrame:
    """为每个文件生成稳定的台风标识 `storm_key`。

    为什么不能直接用现有字段
    ------------------------
    * `tcid`（如 2016278N19130）**每次观测都会变**——它编码的是
      "年份 + 儒略日 + 当前中心纬经度"，不是台风身份。实测同一台风的
      2005111N09142（04-21）→ 2005112N11137（04-22），儒略日每天递增。
      53043 个文件对应 11128 个不同 tcid。
    * `name` 会**跨年份重名**（如 BOLAVEN 出现在 2017/2023/2024），
      按 name 聚合会把不同台风混在一起（45 个名字有重名）。
    * `name` 还可能是 `nameless`（占 5%），一个名字对应一年内多个台风。

    做法：`storm_key = <NAME 大写>-<年份>`，`nameless` 额外按 48h 时间
    间隙切分（`NAMELESS-2018-3`）。实测得到 306 个台风，跨度中位 168h，
    符合台风生命史。
    """
    df = df.copy().sort_values(["name", "time"]).reset_index(drop=True)
    key = df["name"].astype(str).str.upper() + "-" + df["time"].dt.year.astype(str)

    GAP_H = 48.0
    is_nameless = df["name"].astype(str).str.lower().eq("nameless").to_numpy()
    gap = df["time"].diff().dt.total_seconds().div(3600).fillna(0).to_numpy()

    suffix = np.zeros(len(df), dtype=int)
    run = 0
    for i in range(len(df)):
        if i > 0 and is_nameless[i] and is_nameless[i - 1] and key.iloc[i] == key.iloc[i - 1]:
            if gap[i] > GAP_H:
                run += 1
        elif not is_nameless[i]:
            run = 0
        suffix[i] = run
    key = key + np.where(is_nameless & (suffix > 0), "-" + suffix.astype(str), "")
    df["storm_key"] = key
    return df.sort_values(["time"]).reset_index(drop=True)


class CorruptFileError(ValueError):
    """文件内容损坏（HDF5 对象头/校验和错误），无法恢复有用数据。"""


# 物理合理范围：FY 卫星亮温约 150-350 K
TB_MIN, TB_MAX = 100.0, 400.0


def validate_channel(block: np.ndarray, channel: str, member: str = "") -> None:
    """校验云图通道数据是否合理，不合理则抛 CorruptFileError。

    为什么需要显式校验
    ------------------
    部分文件的 HDF5 元数据损坏（对象头校验和错误）。此时 h5py 仍能返回
    **未初始化的内存**，表现为 1e38 量级的乱码数值。若不校验，这些垃圾
    会被当成真实亮温写进 patch 缓存，污染训练数据。

    实测损坏样例（2005302N13113.Kai-tak.2005.10.29.0300.FY2-C）：
      NOMChannelIR1 范围 [-3.4e38, +3.4e38]，而正常应为 [150, 400]
    """
    arr = np.asarray(block, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise CorruptFileError(f"{channel} 全部为 NaN/inf ({member})")
    # 用分位数而非极值判断，避免个别坏点误杀；但乱码会整体偏移
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi > TB_MAX * 10 or lo < -TB_MAX:
        raise CorruptFileError(
            f"{channel} 数值异常（0.5%-99.5% 分位 {lo:.3g}~{hi:.3g}，"
            f"合理范围 {TB_MIN:.0f}-{TB_MAX:.0f}）: {member}")
    # 有效像素（在物理范围内的）不能太少
    good = float(np.mean((finite >= TB_MIN) & (finite <= TB_MAX)))
    if good < 0.05:
        raise CorruptFileError(
            f"{channel} 仅 {good:.1%} 像素在合理范围内: {member}")


def read_netcdf_member(zip_path: str | Path, member: str):
    """从 zip 直接读取一个 NetCDF 文件（不落盘）。

    优先用 xarray + h5netcdf；若失败则回退到 h5py 原生读取。

    为什么要回退
    ------------
    部分文件（如 2005302N13113.Kai-tak.2005.10.29.0300.FY2-C）的 HDF5
    元数据校验和损坏。`h5netcdf` 在解析维度标签（dimension scales）时会
    `RuntimeError: incorrect metadata checksum`，但 **h5py 仍能正常读取数据**。
    实测该文件的 CentLat/CentLon/CentPrs/WindSpd 全部可读。

    回退路径返回的对象只支持 `.variables` 风格的 `varname[...]` 访问，
    与 extract_scalars / extract_patch 的用法兼容。
    """
    import xarray as xr

    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(member)
    try:
        return xr.open_dataset(io.BytesIO(raw), engine="h5netcdf")
    except Exception:
        return _H5pyShim(raw)


class _H5pyShim:
    """h5netcdf 失败时的最小替代：按变量名直读数据。

    只实现本模块用到的接口：`ds[name].values`、`"name" in ds`、`ds.close()`。
    """

    def __init__(self, raw: bytes):
        import h5py
        self._f = h5py.File(io.BytesIO(raw), "r")

    def __contains__(self, name: str) -> bool:
        return name in self._f

    def __getitem__(self, name: str):
        try:
            return _H5Var(self._f[name])
        except (KeyError, RuntimeError, OSError) as exc:
            # h5py 对损坏的对象头会抛 KeyError("bad object header...")，
            # 统一转成语义明确的 CorruptFileError，便于上层区分"文件坏了"
            raise CorruptFileError(f"无法读取变量 {name}: {type(exc).__name__}") from exc

    @property
    def variables(self):
        return self

    def keys(self):
        return list(self._f.keys())

    def close(self):
        self._f.close()

    @property
    def attrs(self) -> dict:
        try:
            return {k: self._f.attrs[k] for k in self._f.attrs}
        except Exception:
            # 属性区可能也受校验和损坏影响，读不到就返回空
            return {}


class _H5Var:
    def __init__(self, node):
        self._n = node

    @property
    def values(self):
        import numpy as np
        try:
            return np.asarray(self._n[()])
        except (KeyError, RuntimeError, OSError) as exc:
            raise CorruptFileError(f"数据块读取失败: {type(exc).__name__}") from exc

    @property
    def shape(self):
        return self._n.shape


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
            validate_channel(block, ch, member)     # 损坏文件在此抛出
            block[block >= 65535] = np.nan          # missing_value
            block[block < TB_MIN] = np.nan          # valid_range 下限
            block[block > TB_MAX] = np.nan          # 上限（防乱码）
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


def patch_cache_name(cfg) -> str:
    """按 patch 边长生成缓存文件名，避免换尺寸后覆盖旧缓存。

    `satellite.patch_cache` 可以是 "sat_patches.npz"，此时自动插入尺寸：
        sat_patches.npz  ->  sat_patches64.npz

    文件名里已经带数字（用户自定义）时原样返回，不做二次插入。
    """
    base = str(cfg.satellite.patch_cache)
    size = int(cfg.satellite.patch_size)
    stem, dot, ext = base.rpartition(".")
    if not dot:                         # 没有扩展名
        stem, ext = base, ""
    if re.search(r"\d", stem):          # 已含尺寸信息
        return base
    return f"{stem}{size}" + (f".{ext}" if ext else "")
