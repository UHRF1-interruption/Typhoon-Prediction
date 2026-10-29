"""云图手工特征：用少量物理可解释的统计量替代 CNN。

为什么这样做
------------
之前的融合模型把每帧 32×32×3 的云图交给三层 CNN（约 30 万参数），
在只有几十个台风的小样本上严重过拟合：
    验证 MAE 在 epoch 3 达到最低 1.70，到 epoch 18 涨到 2.08（+23%）。

这里改用**台风结构的物理描述子**——每个时次只有约 15 维，几乎不增加参数，
但保留了强度预报最相关的信息：

  * 眼区温度    中心 ±3 像素（约 12km）的平均亮温：眼越暖、越清晰，台风越强
  * 眼区对比度  中心 vs 环境环带（30–120km）的亮温差：暖心结构的直接体现
  * 冷云面积    低于阈值的像素占比：强对流区（CDO）覆盖范围
  * 对称性      各方向环带亮温的方差：结构不对称通常对应快速变化期
  * 亮温分位数  不同百分位反映云顶高度的分布

这些量都是业务预报员看云图卫星图时实际关注的特征，因此物理上合理，
比让网络自己从 3072 个像素里找规律更不容易过拟合。
"""
from __future__ import annotations

import numpy as np

# 特征名（顺序固定，便于与模型对齐）
PATCH_FEATURE_NAMES = [
    "ir1_center",        # 中心 10.8μm 亮温
    "ir1_center_std",    # 中心亮温标准差
    "ir1_warmest",       # 中心区最暖像素（眼温）
    "ir1_env",           # 环境环带平均亮温
    "ir1_contrast",      # 中心 - 环境（负值越大说明云顶越高）
    "ir3_center",
    "ir3_center_std",
    "ir3_contrast",
    "ir2_center",
    "ir2_contrast",
    "cold_area",         # 亮温低于阈值的像素占比
    "very_cold_area",
    "radial_grad",       # 径向亮温梯度（中心到环境的变化率）
    "asymmetry",         # 方位角方向的亮温方差
    "bright_std",        # 整幅 patch 亮温标准差
]

_CH_INDEX = {"ir3": 0, "ir1": 1, "ir2": 2}   # 与 fysatty.CHANNELS 顺序一致
COLD_THRESHOLD = 220.0      # 约对应强对流云顶（K）
VERY_COLD_THRESHOLD = 200.0


def _rings(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """生成中心区与环带的距离掩码。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)     # 单位：像素（4km）
    center = dist <= 3.0
    env = (dist >= 8.0) & (dist <= 14.0)
    inner = dist <= 6.0
    outer = dist >= 10.0
    return center, env, inner, outer, dist


def patch_features(patch: np.ndarray) -> np.ndarray:
    """把单帧 patch (C,S,S) 转成特征向量 PATCH_FEATURE_NAMES。

    通道顺序: ir3(6.95), ir1(10.8), ir2(12.0)，与 fysatty.CHANNELS 一致。
    """
    p = np.asarray(patch, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"patch 形状应为 (C,S,S)，收到 {p.shape}")
    s = p.shape[-1]
    center, env, inner, outer, dist = _rings((s, s))

    ir3, ir1, ir2 = p[0], p[1], p[2]
    feats = []

    # 10.8μm：眼区与外界的对比最关键
    c = float(ir1[center].mean())
    c_std = float(ir1[center].std())
    warmest = float(ir1[center].max())
    e = float(ir1[env].mean()) if env.any() else c
    feats += [c, c_std, warmest, e, c - e]

    # 6.95μm 水汽通道：反映中高层水汽与对流
    c3 = float(ir3[center].mean())
    e3 = float(ir3[env].mean()) if env.any() else c3
    feats += [c3, float(ir3[center].std()), c3 - e3]

    # 12.0μm 分裂窗：与 10.8 配合可识别薄卷云
    c2 = float(ir2[center].mean())
    e2 = float(ir2[env].mean()) if env.any() else c2
    feats += [c2, c2 - e2]

    # 冷云面积（用 10.8μm）
    feats.append(float((ir1 < COLD_THRESHOLD).mean()))
    feats.append(float((ir1 < VERY_COLD_THRESHOLD).mean()))

    # 径向梯度：中心到环境环的平均变化速率（每像素）
    inner_mean = float(ir1[inner].mean()) if inner.any() else c
    outer_mean = float(ir1[outer].mean()) if outer.any() else e
    d_in = float(dist[inner].mean()) if inner.any() else 0.0
    d_out = float(dist[outer].mean()) if outer.any() else 1.0
    feats.append((outer_mean - inner_mean) / max(1e-3, d_out - d_in))

    # 方位角对称性：以中心为圆心的各方向平均亮温的离散程度
    yy, xx = np.mgrid[0:s, 0:s]
    cy, cx = (s - 1) / 2.0, (s - 1) / 2.0
    theta = np.arctan2(yy - cy, xx - cx)
    band = (dist >= 4.0) & (dist <= 12.0)
    sector_means = []
    n_bins = 8
    for k in range(n_bins):
        lo = -np.pi + 2 * np.pi * k / n_bins
        hi = lo + 2 * np.pi / n_bins
        m = band & (theta >= lo) & (theta < hi)
        if m.any():
            sector_means.append(float(ir1[m].mean()))
    feats.append(float(np.std(sector_means)) if len(sector_means) > 1 else 0.0)

    feats.append(float(ir1.std()))
    return np.asarray(feats, dtype=np.float32)


def stack_patch_features(patches: np.ndarray) -> np.ndarray:
    """批量转换 (..., C, S, S) -> (..., F)。"""
    lead = patches.shape[:-3]
    flat = patches.reshape(-1, *patches.shape[-3:])
    out = np.stack([patch_features(x) for x in flat])
    return out.reshape(*lead, out.shape[-1])
