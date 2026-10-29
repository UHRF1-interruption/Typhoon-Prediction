"""按地形状态分段建模强度预测。

物理依据（实测，2021-2024 卫星数据 5565 个时次）
----------------------------------------------
不同地形状态下的 12h 强度变化方向**相反**：

| 状态     | 样本量 | 12h 平均Δwind | 说明 |
|----------|--------|---------------|------|
| 纯海洋   | 1565   | **+1.19 m/s** | 暖水供能，总体增强 |
| 近岸     | 1434   | +0.62 m/s     | 增强受抑制 |
| 陆地     | 2016   | **−1.79 m/s** | 摩擦+暖心破坏，总体减弱 |

因此把「海洋 / 近岸 / 陆地」混在一个模型里训练，模型需要同时学习两种
相反的动态，可能互相干扰。分段建模让每个模型专注一种物理过程。

分段定义
--------
按台风所在位置的多半径圈陆地占比 land_frac 划分：

  ocean    : land_frac <= 0.01        纯海洋
  coastal  : 0.01 < land_frac <= 0.10  近岸（含陆缘海）
  land     : land_frac > 0.10          已登陆/深入内陆

样本量（2021-2024）: ocean 1652、coastal 1490、land 2233 时次，
各自覆盖 57-68 个台风，足够独立训练。

注意事项
--------
* 分段后每段样本变少（约 1/3），模型容量需相应减小，否则过拟合
* 边界处的样本可能被分到任一段，评估时必须用同一套分段规则
* 训练时每段独立做标准化，推理时按同一规则选择模型
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = ["ocean", "coastal", "land"]

# 分段阈值（land_frac）
OCEAN_MAX = 0.01
COASTAL_MAX = 0.10


def assign_segment(land_frac) -> np.ndarray:
    """根据陆地占比返回段标签数组（dtype=object，元素为 'ocean'/'coastal'/'land'）。"""
    lf = np.asarray(land_frac, dtype=float)
    out = np.full(lf.shape, "ocean", dtype=object)
    out[(lf > OCEAN_MAX) & (lf <= COASTAL_MAX)] = "coastal"
    out[lf > COASTAL_MAX] = "land"
    # 缺失值归入 ocean（保守：不做特殊处理）
    out[~np.isfinite(lf)] = "ocean"
    return out


def segment_samples(bundle: dict, seg_labels: np.ndarray,
                    min_samples: int = 100) -> dict[str, np.ndarray]:
    """把 sample 索引按段拆开。

    bundle: build_intensity_samples 的输出（用于取 sid 以便按台风划分）
    seg_labels: 每个样本的段标签，长度 == 样本数
    返回: {segment: 样本索引数组}
    """
    out = {}
    for seg in SEGMENTS:
        idx = np.where(seg_labels == seg)[0]
        if len(idx) >= min_samples:
            out[seg] = idx
    return out


def sample_segment_labels(bundle: dict, aligned_scalars: pd.DataFrame) -> np.ndarray:
    """为每个样本确定所属段。

    做法：取输入窗口**最后一步**（预报时刻）的 land_frac，
    因为它最接近「当前状态」。
    注意 build_intensity_samples 返回的顺序与 name/end_time 对齐，
    这里用 (name, end_time) 回查 land_frac。
    """
    # 每个 (name, time) 的 land_frac
    lf_map = (aligned_scalars.groupby(["name", "time"])["land_frac"]
              .mean().to_dict())
    labels = np.empty(len(bundle["name"]), dtype=object)
    for i, (nm, et) in enumerate(zip(bundle["name"], bundle["end_time"])):
        key = (nm, pd.Timestamp(et))
        lf = lf_map.get(key, np.nan)
        labels[i] = assign_segment([lf])[0]
    return labels


def summarize_segments(labels: np.ndarray, verbose: bool = True) -> dict:
    """打印/返回各段样本数。"""
    stats = {}
    for seg in SEGMENTS:
        n = int((labels == seg).sum())
        stats[seg] = n
    if verbose:
        total = max(1, sum(stats.values()))
        for seg, n in stats.items():
            print(f"[segment] {seg:8s} {n:5d} 样本 ({n / total * 100:.1f}%)")
    return stats
