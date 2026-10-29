"""IBTrACS 下载与读取。

IBTrACS (International Best Track Archive for Climate Stewardship) 是 NOAA 官方
全球台风最佳路径数据集。CSV 版本前两行是文件级元数据（版本、引用），
第三行才是表头，所以读取时 skiprows=2。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# IBTrACS CSV 的行结构:
#   第 1 行: 列名
#   第 2 行: 单位
#   第 3 行起: 数据
# 注意不同版本可能多一行文件元数据，因此读取时按内容探测真正的列名行。
HEADER_ROW = 0

# 必需列：缺失则无法解析
REQUIRED_COLUMNS = ["SID", "ISO_TIME", "LAT", "LON"]
# 可选列：有就用，没有就留空（IBTrACS 里没有 STORM_SPEED/STORM_DIR）
OPTIONAL_COLUMNS = ["SEASON", "NUMBER", "BASIN", "SUBBASIN", "NAME", "NATURE",
                    "WMO_WIND", "WMO_PRES", "USA_WIND", "USA_PRES", "USA_ATCF_ID",
                    "TRACK_TYPE", "WM0_WIND"]


class DownloadInterrupted(RuntimeError):
    """下载中途失败。已经写入的部分数据仍然有效，下次调用会续传。"""


# 当请求的子集文件不存在时，按优先级回退到这些本地已有的文件。
# ALL（全球）与 since1980 都是完整 IBTrACS，可直接筛选出所需盆地。
LOCAL_FALLBACKS = {
    "WP": ["ibtracs.WP.list.v04r01.csv", "ibtracs.since1980.list.v04r01.csv",
           "ibtracs.ALL.list.v04r01.csv", "ibtracs.last3years.list.v04r01.csv"],
    "NA": ["ibtracs.NA.list.v04r01.csv", "ibtracs.since1980.list.v04r01.csv",
           "ibtracs.ALL.list.v04r01.csv", "ibtracs.last3years.list.v04r01.csv"],
    "EP": ["ibtracs.EP.list.v04r01.csv", "ibtracs.since1980.list.v04r01.csv",
           "ibtracs.ALL.list.v04r01.csv", "ibtracs.last3years.list.v04r01.csv"],
}


def find_local_ibtracs(out_dir: str | Path, suffix: str) -> Path | None:
    """在 data/raw 里查找可用的 IBTrACS 文件，找不到返回 None。"""
    out_dir = Path(out_dir)
    candidates = [f"ibtracs.{suffix}.list.v04r01.csv"]
    candidates += LOCAL_FALLBACKS.get(suffix.upper(), [])
    candidates += ["ibtracs.since1980.list.v04r01.csv", "ibtracs.ALL.list.v04r01.csv"]
    for name in candidates:
        path = out_dir / name
        if path.exists() and path.stat().st_size > 1_000_000:
            return path
    # 最后兜底：目录里任何 ibtracs.*.csv
    for path in sorted(out_dir.glob("ibtracs.*.csv")):
        if path.stat().st_size > 1_000_000:
            return path
    return None


def download_ibtracs(
    out_dir: str | Path,
    version: str = "v04r01",
    base_url: str = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/",
    suffix: str = "WP",
    workers: int = 24,
    time_budget_s: float | None = None,
    round_retries: int = 3,
) -> Path:
    """返回可用的 IBTrACS CSV 路径。

    优先级：
      1. 本地已有完整文件（含用户手动放入的）→ 直接使用，不联网；
      2. 否则下载（多线程分块 + 断点续传）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local = find_local_ibtracs(out_dir, suffix)
    if local is not None:
        size_mb = local.stat().st_size / 1e6
        print(f"[download] 发现本地 IBTrACS 文件，跳过下载: {local.name} ({size_mb:.1f}MB)")
        if f".{suffix}." not in local.name and suffix.upper() != local.name.split(".")[1].upper():
            print(f"[download] 注意: 该文件不是纯 {suffix} 子集，将在解析时按 basin={suffix} 过滤")
        return local

    filename = f"ibtracs.{suffix}.list.{version}.csv"
    target = out_dir / filename
    url = base_url.rstrip("/") + "/" + filename

    from ..utils.fetch import fetch

    stale = target.with_suffix(".csv.part")
    if stale.exists():
        stale.unlink()
        print("[download] 删除旧版单连接残留文件")

    last_exc: Exception | None = None
    for attempt in range(1, round_retries + 1):
        try:
            print(f"[download] 获取 {url}"
                  + (f"（第 {attempt}/{round_retries} 轮）" if attempt > 1 else ""))
            return fetch(url, target, workers=workers,
                         timeout=20, retries=8, time_budget_s=time_budget_s)
        except Exception as exc:
            last_exc = exc
            print(f"[download] 本轮失败: {type(exc).__name__}: {str(exc)[:120]}")
    raise DownloadInterrupted(
        f"下载未能完成，已下载的分块保存在 {target.name}.tmp.parts/，重新运行即可续传。"
        f"最后错误: {last_exc}"
    ) from last_exc


def _detect_header(path: Path, max_scan: int = 6) -> tuple[list[str], int]:
    """探测列名所在行。

    返回 (列名列表, 列名行号(0-based))。列名行的特征是包含 SID 与 ISO_TIME。
    兼容带/不带文件级元数据行的不同 IBTrACS 版本。
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for idx in range(max_scan):
            line = fh.readline()
            if not line:
                break
            cols = [c.strip() for c in line.rstrip("\r\n").split(",")]
            upper = {c.upper() for c in cols}
            if "SID" in upper and "ISO_TIME" in upper:
                return cols, idx
    raise ValueError(f"{path.name} 前 {max_scan} 行未找到列名行（应含 SID 与 ISO_TIME）")


def _merge_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """按给定列顺序取第一个非空值，转成数值。列不存在则返回全 NaN。"""
    merged = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        merged = merged.fillna(values)
    return merged


def load_ibtracs(path: str | Path, basin: str | None = None,
                 min_wind_knots: float = 34.0) -> pd.DataFrame:
    """读取 IBTrACS CSV，清洗成统一列名的长表。

    返回列: sid, season, time(datetime), lat, lon(0-360), wind(kt), pres(hpa)
    """
    path = Path(path)
    print(f"[load] 读取 {path.name}（大文件约需 1-2 分钟）")
    # IBTrACS CSV 前两行是文件元数据，第 3 行是列名，第 4 行是单位。
    # 只读需要的列以控制内存；单位行全部按字符串读，之后再转数值。
    names, header_line = _detect_header(path)
    wanted = [c for c in names
              if c.upper() in {k.upper() for k in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}]
    df = pd.read_csv(
        path,
        skiprows=header_line + 1,     # 跳过列名行，让单位行被 names 覆盖掉
        header=0,
        names=wanted,
        usecols=[names.index(c) for c in wanted],
        dtype=str,
        low_memory=False,
        keep_default_na=False,
        na_values=["", " "],
    )
    df.columns = [c.strip().upper() for c in df.columns]

    # IBTrACS 列名在不同版本略有差异，做兼容映射
    def pick(*names: str) -> str | None:
        for name in names:
            if name in df.columns:
                return name
        return None

    col_sid = pick("SID", "USA_ATCF_ID")
    col_time = pick("ISO_TIME")
    col_lat, col_lon = pick("LAT"), pick("LON")
    # 强度优先用 WMO_WIND（国际统一口径），缺失时回退到 USA_WIND；
    # 实测 WMO_WIND 在很多年代缺失率高达 74%，只用它会丢掉大部分样本。
    col_wind = pick("WMO_WIND", "USA_WIND")
    col_pres = pick("WMO_PRES", "USA_PRES")
    col_basin = pick("BASIN")
    col_season = pick("SEASON")

    missing = [n for n, c in {
        "SID": col_sid, "ISO_TIME": col_time, "LAT": col_lat, "LON": col_lon,
    }.items() if c is None]
    if missing:
        raise ValueError(f"IBTrACS 文件缺少必要列: {missing}，实际列={list(df.columns)}")

    out = pd.DataFrame({
        "sid": df[col_sid].astype(str),
        "time": pd.to_datetime(df[col_time], errors="coerce", utc=True),
        "lat": pd.to_numeric(df[col_lat], errors="coerce"),
        "lon": pd.to_numeric(df[col_lon], errors="coerce"),
        "wind": _merge_numeric(df, ["WMO_WIND", "USA_WIND"]),
        "pres": _merge_numeric(df, ["WMO_PRES", "USA_PRES"]),
        "basin": df[col_basin].astype(str) if col_basin else "",
        "season": pd.to_numeric(df[col_season], errors="coerce") if col_season else np.nan,
        "storm_speed": pd.to_numeric(df["STORM_SPEED"], errors="coerce")
            if "STORM_SPEED" in df.columns else np.nan,
        "storm_dir": pd.to_numeric(df["STORM_DIR"], errors="coerce")
            if "STORM_DIR" in df.columns else np.nan,
    })

    out = out.dropna(subset=["time", "lat", "lon"])

    if basin:
        before = out["sid"].nunique()
        out = out[out["basin"].str.upper().str.contains(basin.upper(), na=False)]
        print(f"[load] 盆地过滤 {basin}: {before} -> {out['sid'].nunique()} 个风暴")

    # 强度过滤：保留达到阈值的行（缺失强度默认保留，便于连续轨迹）
    if min_wind_knots is not None and out["wind"].notna().any():
        keep = out["wind"] >= min_wind_knots
        keep |= out["wind"].isna()
        out = out[keep]

    # 经度统一转到 0-360，避免 -180~180 与 0~360 混用
    out["lon"] = out["lon"] % 360.0

    out = out.sort_values(["sid", "time"]).reset_index(drop=True)
    print(f"[load] 有效记录 {len(out)} 条，{out['sid'].nunique()} 个风暴，"
          f"年份 {int(out['time'].dt.year.min())}-{int(out['time'].dt.year.max())}")
    return out


def summarize(df: pd.DataFrame) -> str:
    """简短的数据概览，便于下载后确认数据质量。"""
    per_storm = df.groupby("sid").size()
    return (
        f"记录数={len(df)}, 风暴数={df['sid'].nunique()}, "
        f"每风暴平均点数={per_storm.mean():.1f} (min={per_storm.min()}, max={per_storm.max()}), "
        f"强度缺失率={df['wind'].isna().mean():.1%}"
    )
