"""CMA 年度台风预报误差数据集读取与分析。

本工程附带的数据文件（``2024_tc_tra_err.csv`` 等）记录了 2024、2025 年
西北太平洋每个台风、每个预报时效的**官方/模式预报位置**与**预报误差**，
覆盖 21 个预报方法，包括:

  - 数值模式: ecmf(EC)、avno(NCEP)、ggfs、japn、egrr、babj(中央气象台)、
    hwrf、pgtw、vhhh、shtm、gztm、gtym、rksl、jawt
  - AI 模型:  pangu(盘古)、aifs、fengwu(风乌)、fengqing(风清)、
    fuxi(伏羲)、graphcast、fourcastnet

用途：
  1. 建立**业务化基线**——自研模型必须打败"持续法"，并对标 EC 与 AI 模式；
  2. 提供 2024/2025 年的真实历史路径，可作为模型的外部测试集；
  3. 分析误差随时效增长规律、大误差(bust)样本特征。

列含义（tra_err）:
  fcst_method         预报方法代号
  TC_ID               台风编号（YYNN，如 2401 = 2024 年第 1 号）
  int_time            起报时间 YYYYMMDDHH（UTC）
  fcst_lead_time      预报时效（小时）
  fcst_lon/fcst_lat   预报位置（度）
  tra_err             路径误差（km）
  bust                是否大误差案例 (0/1)

列含义（int_err）:
  fcst_vmax           预报最大风速（m/s）
  int_err             强度误差（预报 - 实况，m/s，有符号）
  int_err_abs         强度误差绝对值（m/s）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRA_COLUMNS = ["method", "tc_id", "int_time", "lead", "fcst_lon", "fcst_lat", "tra_err", "bust"]
INT_COLUMNS = ["method", "tc_id", "int_time", "lead", "fcst_vmax", "int_err", "int_err_abs", "bust"]

# 方法代号 -> 便于阅读的中文/官方名称
METHOD_LABELS = {
    "ecmf": "ECMWF 高分辨(EC)",
    "avno": "NCEP GFS(avno)",
    "ggfs": "NCEP GFS(ggfs)",
    "japn": "JMA 日本",
    "egrr": "UKMO 英国",
    "babj": "CMA 中央气象台",
    "hwrf": "HWRF 飓风模式",
    "pgtw": "JTWC",
    "vhhh": "香港天文台",
    "shtm": "上海台风所",
    "gztm": "广州热带海洋气象所",
    "gtym": "CMA 台风模式",
    "rksl": "韩国模式",
    "jawt": "日本 AI 台风",
    "pangu": "Pangu-Weather(盘古)",
    "aifs": "ECMWF AIFS",
    "fengwu": "风乌 FengWu",
    "fengqing": "风清 FengQing",
    "fuxi": "伏羲 FuXi",
    "graphcast": "GraphCast",
    "fourcastnet": "FourCastNet",
}


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().replace("\n", " ") for c in df.columns]
    if len(df.columns) != len(columns):
        raise ValueError(f"{path.name} 列数不符: 期望 {len(columns)}，实际 {len(df.columns)}: {list(df.columns)}")
    df.columns = columns
    for col in columns:
        if col in ("method", "tc_id", "int_time"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["int_time"] = pd.to_datetime(df["int_time"], format="%Y%m%d%H", errors="coerce", utc=True)
    df["year"] = df["int_time"].dt.year
    return df


def load_forecast_errors(data_dir: str | Path,
                         years: tuple[int, ...] = (2024, 2025)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取路径误差与强度误差表，返回 (tra, int_)。"""
    data_dir = Path(data_dir)
    tra_frames, int_frames = [], []
    for year in years:
        p = data_dir / f"{year}_tc_tra_err.csv"
        if p.exists():
            tra_frames.append(_read_csv(p, TRA_COLUMNS))
        q = data_dir / f"{year}_tc_int_err.csv"
        if q.exists():
            int_frames.append(_read_csv(q, INT_COLUMNS))
    if not tra_frames:
        raise FileNotFoundError(f"{data_dir} 下没有找到 *_tc_tra_err.csv")
    tra = pd.concat(tra_frames, ignore_index=True)
    int_ = pd.concat(int_frames, ignore_index=True) if int_frames else pd.DataFrame()
    years = ", ".join(str(int(y)) for y in sorted(tra["year"].dropna().unique()))
    print(f"[fcst] 路径误差 {len(tra)} 行 / {tra['method'].nunique()} 个方法 / "
          f"{tra['tc_id'].nunique()} 个台风, 年份 {years}")
    if len(int_):
        print(f"[fcst] 强度误差 {len(int_)} 行 / {int_['method'].nunique()} 个方法")
    return tra, int_


def clean_outliers(df: pd.DataFrame, col: str = "tra_err", threshold_km: float = 3000.0) -> pd.DataFrame:
    """剔除明显异常值。

    数据里存在少量物理上不可能的记录（如 3 小时时效误差 3542 km，多半是
    预报位置写成 0 或跨半球跳变），统计时应该剔除，否则均值会被严重带偏。
    """
    before = len(df)
    out = df[(df[col] > 0) & (df[col] < threshold_km)].copy()
    print(f"[fcst] 剔除 {col} 异常值 {before - len(out)} 行 "
          f"(保留 {len(out)}, 阈值 <{threshold_km} km)")
    return out


def error_by_horizon(df: pd.DataFrame, value_col: str = "tra_err",
                     horizons: list[int] | None = None,
                     methods: list[str] | None = None) -> pd.DataFrame:
    """计算各方法在各时效的平均误差。"""
    if horizons is not None:
        df = df[df["lead"].isin(horizons)]
    if methods is not None:
        df = df[df["method"].isin(methods)]
    pivot = df.pivot_table(index="method", columns="lead", values=value_col, aggfunc="mean")
    return pivot.sort_index(axis=1)


def summarize_baselines(tra: pd.DataFrame, int_: pd.DataFrame,
                        horizons: list[int] | None = None) -> pd.DataFrame:
    """生成业务基线汇总表：各方法在 24/48/72/120h 的路径与强度误差。"""
    horizons = horizons or [24, 48, 72, 120]
    rows = []
    for method in sorted(tra["method"].unique()):
        sub = tra[tra["method"] == method]
        row = {"method": method, "label": METHOD_LABELS.get(method, method),
               "n": len(sub)}
        for h in horizons:
            v = sub.loc[sub["lead"] == h, "tra_err"]
            row[f"tra_{h}h_km"] = float(v.mean()) if len(v) else np.nan
        if len(int_):
            si = int_[int_["method"] == method]
            for h in horizons:
                v = si.loc[si["lead"] == h, "int_err_abs"]
                row[f"int_{h}h_ms"] = float(v.mean()) if len(v) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("tra_24h_km", na_position="last").reset_index(drop=True)
