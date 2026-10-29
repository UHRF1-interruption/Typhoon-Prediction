"""从卫星红外云图反演台风暖心结构特征。

物理依据
--------
台风的强度本质上由**暖心强度**决定：
  * 暖心 = 高层中心温度高于周围环境的正异常
  * 暖心越强，中心气压越低（满足静力平衡），风速越大
  * 暖心被地形破坏时强度骤降

为什么红外云图能反演
--------------------
红外亮温近似反映云顶高度：
  * 亮温低（<200K）= 深对流云顶高 = 眼墙强对流
  * 亮温高（>270K）= 晴空或低云 = 眼区清空

关键结构指标（业务上常用）：

| 特征 | 定义 | 与强度的关系 |
|---|---|---|
| **中心亮温差** (Tb_center - Tb_ring) | 眼区比眼墙暖多少 | **正相关**（眼越清空越强）|
| **眼墙最低亮温** | 眼墙内最低温 | **负相关**（对流越深越强）|
| **CDO 面积** | 中心密集云区（<220K）占比 | 正相关 |
| **亮温梯度** | 眼墙向外侧的径向梯度 | 正相关（结构越紧实）|
| **冷云占比** | <200K 像素比例 | 正相关 |
| **对称性** | 四象限亮温标准差 | 负相关（越对称越强）|
| **同心度** | 中心到最冷环的距离 | 反映眼壁位置 |

实现说明
--------
patch 是 (3, S, S) 的 32×32 网格，中心在 patch 中央（S//2）。
我们不知道每个 patch 覆盖的真实地理尺度（取决于 FY 卫星分辨率），
但**径向距离的相对关系**仍然有效，因此特征以"像素"为单位计算。
"""
from __future__ import annotations

import numpy as np

# 暖心结构特征（顺序固定）
WARMCORE_FEATURES = [
    "wc_center_tb",      # 中心（3x3）平均亮温 K
    "wc_ring_tb",        # 眼墙环（半径 4-7 像素）最低亮温
    "wc_eye_diff",       # 中心 - 眼墙环（暖心信号，越大越强）
    "wc_min_tb",         # 全 patch 最低亮温
    "wc_cold_frac",      # <200K 像素占比（深对流）
    "wc_cdo_frac",       # <220K 像素占比（中心密集云区）
    "wc_grad",           # 径向亮温梯度（中心到 8 像素）
    "wc_asym",           # 四象限亮温标准差（对称性）
    "wc_center_r",       # 最冷环所在半径（眼壁位置）
    "wc_contrast",       # patch 内亮温动态范围
]


def _radius_map(size: int) -> np.ndarray:
    """返回每个像素到中心的半径（像素单位）。"""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    return np.hypot(yy - c, xx - c)


def compute_warmcore_features(patch: np.ndarray) -> np.ndarray:
    """从单个 patch 提取暖心结构特征。

    patch: (C, S, S) 或多通道；若是 3 通道，取通道平均亮温
           若只有单通道，直接用
    返回: (len(WARMCORE_FEATURES),) float32
    """
    p = np.asarray(patch, dtype=np.float32)
    if p.ndim == 3:
        tb = np.nanmean(p, axis=0)
    else:
        tb = p.copy()
    # 掩掉无效值（0 通常是无数据）
    valid = np.isfinite(tb) & (tb > 50.0)
    if valid.sum() < 10:
        return np.full(len(WARMCORE_FEATURES), np.nan, np.float32)
    tb = np.where(valid, tb, np.nan)

    S = tb.shape[0]
    rmap = _radius_map(S)
    c = S // 2

    # 中心（3x3）
    center_tb = float(np.nanmean(tb[max(0, c - 1):c + 2, max(0, c - 1):c + 2]))

    # 眼墙环：半径 4-7 像素
    ring = (rmap >= 4.0) & (rmap <= 7.0) & valid
    ring_tb = float(np.nanmin(tb[ring])) if ring.sum() > 3 else np.nan

    # 全图最低
    min_tb = float(np.nanmin(tb))

    # 冷云占比
    cold_frac = float(np.nansum(tb < 200.0) / max(1, valid.sum()))
    cdo_frac = float(np.nansum(tb < 220.0) / max(1, valid.sum()))

    # 径向梯度：中心到半径 8 的平均亮温变化率
    far = (rmap >= 7.5) & (rmap <= 9.0) & valid
    far_tb = float(np.nanmean(tb[far])) if far.sum() > 3 else np.nan
    grad = (far_tb - center_tb) / 8.0 if np.isfinite(far_tb) else np.nan

    # 四象限对称性：相对于中心的四个象限平均亮温的标准差
    qvals = []
    for sy, sx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        sel = np.zeros_like(valid)
        ys = slice(0, c) if sy < 0 else slice(c + 1, S)
        xs = slice(0, c) if sx < 0 else slice(c + 1, S)
        sel[ys, xs] = True
        sel &= valid
        if sel.sum() > 5:
            qvals.append(float(np.nanmean(tb[sel])))
    asym = float(np.std(qvals)) if len(qvals) >= 3 else np.nan

    # 最冷环所在半径（沿径向剖面找最低平均亮温）
    prof = []
    for r0 in range(0, int(S / 2)):
        m = (rmap >= r0) & (rmap < r0 + 1) & valid
        if m.sum() >= 4:
            prof.append((r0, float(np.nanmean(tb[m]))))
    center_r = float(min(prof, key=lambda x: x[1])[0]) if prof else np.nan

    # 动态范围
    contrast = float(np.nanmax(tb) - min_tb)

    feats = [
        center_tb,
        ring_tb,
        (center_tb - ring_tb) if np.isfinite(ring_tb) else np.nan,
        min_tb,
        cold_frac,
        cdo_frac,
        grad,
        asym,
        center_r,
        contrast,
    ]
    return np.array(feats, dtype=np.float32)


def build_warmcore_table(patches: dict, keys) -> np.ndarray:
    """批量提取暖心特征。patches: {(name,time)->(C,S,S)}, keys: 对应的键序列。"""
    rows = []
    for k in keys:
        p = patches[k]
        rows.append(compute_warmcore_features(p))
    return np.stack(rows) if rows else np.zeros((0, len(WARMCORE_FEATURES)), np.float32)
