"""云图手工特征：用少量物理可解释的统计量替代 CNN。

为什么这样做
------------
之前的融合模型把每帧 32×32×3 的云图交给三层 CNN（约 30 万参数），
在只有几十个台风的小样本上严重过拟合：
    验证 MAE 在 epoch 3 达到最低 1.70，到 epoch 18 涨到 2.08（+23%）。

这里改用**台风结构的物理描述子**——每个时次只有约 15 维，几乎不增加参数，
但保留了强度预报最相关的信息：

  * 眼区温度    中心约 ±12km 的平均亮温：眼越暖、越清晰，台风越强
  * 眼区对比度  中心 vs 环境环带的亮温差：暖心结构的直接体现
  * 冷云面积    低于阈值的像素占比：强对流区（CDO）覆盖范围
  * 对称性      各方向环带亮温的方差：结构不对称通常对应快速变化期
  * 圈层亮温    内核 / 雨带 / 外圈三个阶段，见下

尺度说明（重要）
----------------
早期版本把半径写死成 32×32 patch 的像素值（眼区 ≤3px 即 12km）。
patch 边长现在是可配置的（默认 64，即 ±128km），若仍用绝对像素，
换尺寸后眼区会变成 12km、环境环带跑出画面，特征全部失真。
因此所有半径一律**按 patch 边长比例**定义（R 表示到边缘的半径）：

    眼区   r <= 0.09R     （64px 时约 ±12km，与原来含义一致）
    内核   0.09R < r <= 0.25R
    雨带   0.25R < r <= 0.50R
    外圈   0.50R < r <= 1.00R

为什么要加圈层特征：实测（IR1 通道平均亮温 vs 当前强度，n=120）
    视野 ±64km r=-0.46 | ±128km r=-0.55 | ±256km r=-0.50 | ±760km r=-0.14
按半径拆开 32×32 patch：核心 r<3 r=-0.19，外圈 8<r<16 r=-0.44。
即信息量从中心向外递增，只看眼区会丢掉大部分信号。
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
    # 以下 8 维为圈层特征（扩容视野后新增，见模块文档）
    "ir1_inner",         # 内核环平均亮温
    "ir1_band",          # 雨带环平均亮温
    "ir1_outer",         # 外圈平均亮温
    "ir1_outer_minus_inner",
    "cold_area_band",    # 雨带冷云面积占比
    "cold_area_outer",   # 外圈冷云面积占比
    "very_cold_area_band",
    "asymmetry_outer",   # 外圈方位角方差（螺旋结构/对流不对称）
]

_CH_INDEX = {"ir3": 0, "ir1": 1, "ir2": 2}   # 与 fysatty.CHANNELS 顺序一致
COLD_THRESHOLD = 220.0      # 约对应强对流云顶（K）
VERY_COLD_THRESHOLD = 200.0
PIXEL_KM = 4.0              # 卫星网格分辨率（km/像素），用于把像素距离换算成物理距离


def _rings(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """按 patch 边长比例生成各圈层的距离掩码。

    返回 (center, env, inner, outer, band, out_ring, dist)。
    所有半径都是到中心距离 R=(s-1)/2 的比例，因此与 patch 尺寸无关：

        center   r <= 0.09R    眼区（64px 时约 ±12km）
        inner    r <= 0.25R    内核
        env      0.31R-0.55R   环境环带（与旧版 32px 下的 8-14px 一致）
        band     0.25R-0.50R   雨带
        out_ring 0.50R-1.00R   外圈
        outer    r >= 0.39R    径向梯度用的外围环
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)     # 单位：像素（4km）
    R = float(max(1.0, cx))                             # 边长一半（正方形 patch）
    r = dist / R
    center = r <= 0.09
    inner = r <= 0.25
    env = (r >= 0.31) & (r <= 0.55)
    band = (r > 0.25) & (r <= 0.50)
    out_ring = (r > 0.50) & (r <= 1.00)
    outer = r >= 0.39
    return center, env, inner, outer, band, out_ring, dist


def _sector_std(chan: np.ndarray, mask: np.ndarray, theta: np.ndarray,
                n_bins: int = 8) -> float:
    """mask 区域内按方位角分 n_bins 扇区的平均亮温离散程度。"""
    means = []
    for k in range(n_bins):
        lo = -np.pi + 2 * np.pi * k / n_bins
        hi = lo + 2 * np.pi / n_bins
        m = mask & (theta >= lo) & (theta < hi)
        if m.any():
            means.append(float(chan[m].mean()))
    return float(np.std(means)) if len(means) > 1 else 0.0


def patch_features(patch: np.ndarray) -> np.ndarray:
    """把单帧 patch (C,S,S) 转成特征向量 PATCH_FEATURE_NAMES。

    通道顺序: ir3(6.95), ir1(10.8), ir2(12.0)，与 fysatty.CHANNELS 一致。
    尺寸无关：所有半径按 patch 边长比例定义。
    """
    p = np.asarray(patch, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"patch 形状应为 (C,S,S)，收到 {p.shape}")
    s = p.shape[-1]
    center, env, inner, outer, band, out_ring, dist = _rings((s, s))
    R_px = (s - 1) / 2.0                                # patch 半径（像素）

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

    # 径向梯度：内核到外围的平均变化率，单位 K/km，与 patch 尺寸无关。
    # 分母必须用两个环带的**面积加权中心半径**；若用 dist[mask].mean()，
    # inner 是实心圆盘、outer 是延伸到边缘的大环带，两者均值距离之差会随
    # 边长等比放大，同一物理场在 32px 与 64px 下梯度会差 2 倍（已实测）。
    inner_mean = float(ir1[inner].mean()) if inner.any() else c
    outer_mean = float(ir1[outer].mean()) if outer.any() else e
    inner_c = 0.125 * R_px      # 圆盘 r<=0.25R 的面积加权中心 ≈ 2/3*0.25R
    outer_c = 0.695 * R_px      # 环带 0.39R-1.0R 的面积加权中心 ≈ 0.695R
    feats.append((outer_mean - inner_mean) / max(1e-3, (outer_c - inner_c) * PIXEL_KM))

    # 方位角对称性：以中心为圆心、半径 0.125R-0.375R 环带上各方向平均亮温的离散程度
    # （旧版 32px 下即 4-12px，比例等价）
    yy, xx = np.mgrid[0:s, 0:s]
    cy, cx = (s - 1) / 2.0, (s - 1) / 2.0
    theta = np.arctan2(yy - cy, xx - cx)
    r_norm = dist / max(1.0, cx)
    feats.append(_sector_std(ir1, (r_norm >= 0.125) & (r_norm <= 0.375), theta))

    feats.append(float(ir1.std()))

    # ---- 圈层特征（C 部分）：信息量从中心向外递增，必须显式喂给模型 ----
    def mean_in(ch, m):
        return float(ch[m].mean()) if m.any() else float("nan")

    i_m = mean_in(ir1, inner)
    b_m = mean_in(ir1, band)
    o_m = mean_in(ir1, out_ring)
    feats += [i_m, b_m, o_m, o_m - i_m]
    feats.append(mean_in((ir1 < COLD_THRESHOLD).astype(np.float32), band))
    feats.append(mean_in((ir1 < COLD_THRESHOLD).astype(np.float32), out_ring))
    feats.append(mean_in((ir1 < VERY_COLD_THRESHOLD).astype(np.float32), band))
    feats.append(_sector_std(ir1, out_ring, theta))
    return np.asarray(feats, dtype=np.float32)


def stack_patch_features(patches: np.ndarray) -> np.ndarray:
    """批量转换 (..., C, S, S) -> (..., F)。"""
    lead = patches.shape[:-3]
    flat = patches.reshape(-1, *patches.shape[-3:])
    out = np.stack([patch_features(x) for x in flat])
    return out.reshape(*lead, out.shape[-1])
