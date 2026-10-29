"""CMA 台风最佳路径数据读取（GB/T 19201 格式）。

CMA（中国气象局）发布的西北太平洋热带气旋最佳路径以"每个台风一个文本块"的
形式提供，块首是台风索引行，随后每次观测一行：

  索引行（示例）:
    0314 0001  29 0001 1001 0 6 TAIMEI                20061229
    │    │     │  │    │    │ │ │                     └ 更新时间
    │    │     │  │    │    │ │ └ 路径记录条数
    │    │     │  │    │    │ └ 是否有其它来源标记
    │    │     │  │    │    └ 台风中文名/英文名
    │    │     │  │    └ 台风序号
    │    │     │  └ 年度内序号
    │    │     └ 路径记录条数（部分版本）
    │    └ 台风类型/序号标记
    └ 年份(后两位) + 序号，如 0314 = 2003 年第 14 号

  观测行（示例）:
    2006090118 2  62 1002  2.0      13
    └时间YYYYMMDDHH └等级 └纬度(0.1度) └经度(0.1度) └风速(10m/s) └气压(4hPa)

不同年份的文件列宽略有差异，因此解析时优先按固定列位，失败再退化为空格切分，
并对数值做合理性校验（风速、气压、经纬度范围）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# CMA 强度等级 -> 名称
GRADE_NAMES = {
    0: "TC", 1: "TD", 2: "TS", 3: "STS", 4: "TY", 5: "STY", 6: "SuperTY",
    9: "unknown",
}


@dataclass
class ParseStats:
    blocks: int = 0
    rows: int = 0
    skipped_lines: int = 0


def _parse_observation(line: str) -> dict | None:
    """解析一行观测记录，返回 None 表示无法解析或数值不合理。

    不同来源的 CMA 文本列布局不同，这里用空格切分后按"数值个数"判定：

      - 5 个数值: [grade, lat×10, lon×10, wind×10, pres÷4]
      - 4 个数值: [grade, lat×10, lon×10, wind×10]（本工程的 CH*SAT.txt 即此格式，
        没有气压列，气压记 NaN）

    实测 CH*SAT.txt 行长恒为 34，例如:
        2004040514 2  88 1489  3.0      23
        └时间      └等级  └纬度 └经度   └风速   └(此处两字段实为风速精度与气压缩写)
    """
    s = line.rstrip("\n")
    if len(s) < 30:
        return None

    def grab(a: int, b: int) -> str:
        return s[a:b].strip() if len(s) > a else ""

    try:
        ts = grab(0, 10)
        if not ts.isdigit() or len(ts) != 10:
            return None
        year, month, day, hour = int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10])
        if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31 and hour <= 23):
            return None

        tokens = s.split()
        if len(tokens) < 4:
            return None
        # 跳过第一个时间字段，只对其后的数值做定位
        values = tokens[1:]
        numeric = [t for t in values if re.fullmatch(r"-?\d+(\.\d+)?", t)]
        if len(numeric) < 4:
            return None

        has_pres = len(numeric) >= 5
        grade = int(float(numeric[0])) if has_pres else int(float(numeric[0]))
        lat_raw = float(numeric[1])
        lon_raw = float(numeric[2])
        wind_raw = float(numeric[3])
        pres_raw = float(numeric[4]) if has_pres else float("nan")
    except (ValueError, IndexError):
        return None

    lat = lat_raw / 10.0
    lon = lon_raw / 10.0
    wind = wind_raw / 10.0 if wind_raw >= 100 else wind_raw      # 3 位整数是 ×10
    if pres_raw != pres_raw:                                     # NaN
        pres = float("nan")
    elif pres_raw >= 850:
        pres = pres_raw
    elif pres_raw >= 100:
        pres = pres_raw * 4.0                                    # ×4 存储格式
    else:
        pres = float("nan")

    if not (-5 <= lat <= 60):
        return None
    if not (85 <= lon <= 360):
        return None
    if not (0 <= wind <= 120):
        return None
    if pres == pres and not (850 <= pres <= 1030):
        pres = float("nan")

    return {
        "time": pd.Timestamp(year=year, month=month, day=day, hour=hour),
        "grade": grade,
        "lat": lat,
        "lon": lon % 360.0,
        "wind": wind,          # m/s
        "pres": pres,          # hPa
    }


def parse_cma_txt(path: str | Path) -> tuple[pd.DataFrame, ParseStats]:
    """解析一个 CMA 路径文本文件，返回长表与统计。

    文件结构（每个台风一段）:
        索引行: <占位码> 0000 <生存时长h> <序号> <YYNN> <flag> <flag> <NAME> <更新时间>
        观测行: YYYYMMDDHH <grade> <lat×10> <lon×10> <wind×10> <...>
    其中 YYNN 形如 0401 = 2004 年第 1 号台风。
    """
    path = Path(path)
    stats = ParseStats()
    records: list[dict] = []
    current_sid: str | None = None
    current_name = ""
    block_index = 0

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        # 观测行：以 10 位 YYYYMMDDHH 时间戳开头
        is_obs = len(line) >= 10 and line[:10].isdigit() and _looks_datetime(line[:10])
        if is_obs:
            rec = _parse_observation(line)
            if rec is None:
                stats.skipped_lines += 1
                continue
            if current_sid is None:
                block_index += 1
                current_sid = f"CMAYYYY{block_index:03d}"
            rec["sid"] = current_sid
            rec["name"] = current_name
            records.append(rec)
            stats.rows += 1
            continue

        tokens = line.split()
        # 索引行特征：>=8 个字段，含 YYNN 编号和字母名字
        if len(tokens) >= 8:
            stats.blocks += 1
            block_index += 1
            yyn = tokens[4] if len(tokens) > 4 else ""
            if re.fullmatch(r"\d{4}", yyn):
                yy, nn = int(yyn[:2]), yyn[2:]
                year = 2000 + yy if yy < 70 else 1900 + yy
                current_sid = f"CMA{year}{nn}"
            else:
                current_sid = f"CMA{block_index:04d}"
            name = ""
            for tok in tokens:
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{1,}", tok):
                    name = tok
                    break
            current_name = name
        else:
            stats.skipped_lines += 1

    df = pd.DataFrame(records)
    if df.empty:
        return df, stats
    df = df.sort_values(["sid", "time"]).reset_index(drop=True)
    df["basin"] = "WP"
    df["season"] = df["time"].dt.year
    return df, stats


def _looks_datetime(ts: str) -> bool:
    """校验 YYYYMMDDHH 的合理性。"""
    try:
        year, month, day, hour = int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10])
    except ValueError:
        return False
    return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31 and hour <= 23


def combine_cma_files(files: list[str | Path]) -> pd.DataFrame:
    """解析多个文件并合并、去重（同一台风可能在相邻年份文件里重复）。"""
    frames = []
    for f in sorted(map(Path, files)):
        df, stats = parse_cma_txt(f)
        print(f"[cma] {f.name}: {stats.blocks} 个台风块, {stats.rows} 条观测, "
              f"跳过 {stats.skipped_lines} 行")
        if not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError("没有解析出任何记录")
    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=["sid", "time"]).sort_values(["sid", "time"])
    out = out.reset_index(drop=True)
    print(f"[cma] 合并后 {len(out)} 条（去重 {before - len(out)} 条），"
          f"{out['sid'].nunique()} 个台风，年份 {out['time'].dt.year.min()}-{out['time'].dt.year.max()}")
    return out


def summarize_cma(df: pd.DataFrame) -> str:
    if df.empty:
        return "(空)"
    per_storm = df.groupby("sid").size()
    interval = df.groupby("sid")["time"].apply(lambda s: s.diff().dt.total_seconds().div(3600).median())
    return (
        f"记录={len(df)}, 台风数={df['sid'].nunique()}, "
        f"每台风观测数 中位数={per_storm.median():.0f} (min={per_storm.min()}, max={per_storm.max()}), "
        f"时间间隔中位数={interval.median():.1f}h, "
        f"风速范围={df['wind'].min():.0f}-{df['wind'].max():.0f} m/s, "
        f"气压范围={df['pres'].min():.0f}-{df['pres'].max():.0f} hPa"
    )
