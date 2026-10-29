"""快速增强（RI）事件预测：任务定义与样本构造。

背景与动机
----------
前一轮 5 折交叉验证证明：预测「未来 12h 强度数值」时，纯时序 GRU（1.62±0.16 m/s）
明显优于任何带云图的模型。原因是 12h 内强度变化高度自相关，8 步历史序列已含
绝大部分可预测信号，云图提供的是「同一时刻的快照」，增量信息有限。

本模块改预测**快速增强事件（Rapid Intensification, RI）**，并把预测窗口延长到 24h：

  1. RI 是强度预报的核心难点，业务上关注的是「会不会爆发」而不是「数值准不准」；
  2. 云图的暖心结构、眼区清晰度、对流爆发是 RI 的重要前兆信号，
     在事件判别上应比数值回归更能体现价值；
  3. 24h 窗口让纯外推能力显著下降，给外部信息留出发挥空间。

RI 定义（业界常用）:
  未来 24 小时内最大风速增加 ≥ 阈值（默认 15 m/s ≈ 30 kt）。
  本项目数据里 15 m/s 阈值只有约 4.5% 正样本，样本偏少时可用 10 m/s（11.7%）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sat_intensity import SCALAR_FEATURES, add_intensity_features, regularize_intensity


@dataclass
class RIConfig:
    in_len: int = 8            # 过去 24h（8 × 3h）
    horizon: int = 8           # 未来 24h（8 × 3h）
    interval_h: float = 3.0
    ri_threshold_ms: float = 15.0   # RI 判定阈值（m/s）
    patch_size: int = 32
    use_patch: bool = True
    # 可选：只保留起始强度在某区间的样本（RI 主要发生在中等强度台风上）
    min_wind: float = 0.0
    max_wind: float = 999.0


def build_ri_samples(scalars: pd.DataFrame, cfg: RIConfig,
                     patches: dict | None = None) -> dict:
    """构造 RI 分类样本。

    返回 dict:
      s    (N, in_len, F)        标量特征序列
      p    (N, in_len, C,S,S)    云图 patch（use_patch=True）
      pfeat (N, in_len, F_img)   云图手工特征（use_patch=True）
      y    (N,)                  0/1 标签：未来 horizon 步是否达到 RI
      delta (N,)                 未来 24h 实际强度变化（用于分析与评估）
      w0   (N,)                  起始强度
      name (N,)                  台风名
      end_time (N,)              输入窗口最后一刻
    """
    g = (scalars.groupby(["name", "time"])
         .agg(wind=("wind_spd", "mean"), prs=("cent_prs", "mean"),
              lat=("cent_lat", "mean"), lon=("cent_lon", "mean"))
         .reset_index())

    S, P, Y, DELTA, W0, names, times = [], [], [], [], [], [], []
    skipped_patch = 0

    for name, seq in g.groupby("name"):
        reg = regularize_intensity(seq, cfg)
        if reg.empty or len(reg) < cfg.in_len + cfg.horizon:
            continue
        reg = add_intensity_features(reg, cfg.interval_h)
        feat = reg[SCALAR_FEATURES].to_numpy(np.float32)
        wind = reg["wind"].to_numpy(np.float32)
        n = len(reg)

        for start in range(0, n - cfg.in_len - cfg.horizon + 1):
            end_in = start + cfg.in_len
            end_out = end_in + cfg.horizon
            s_win = feat[start:end_in]
            w = wind[start:end_in]
            if not np.isfinite(s_win).all():
                continue
            w0 = float(wind[end_in - 1])
            if not (cfg.min_wind <= w0 <= cfg.max_wind):
                continue
            future = wind[end_in:end_out]
            if not np.isfinite(future).all():
                continue
            delta = float(future.max() - w0)      # 未来窗口内的最大增幅
            label = int(delta >= cfg.ri_threshold_ms)

            if cfg.use_patch:
                if patches is None:
                    raise ValueError("use_patch=True 时必须提供 patches")
                frame_times = reg["time"].iloc[start:end_in].tolist()
                got = [patches.get((name, ts)) for ts in frame_times]
                if any(gg is None for gg in got):
                    skipped_patch += 1
                    continue
                if any((np.asarray(gg) > 0).mean() < 0.5 for gg in got):
                    skipped_patch += 1
                    continue
                P.append(np.stack(got))

            S.append(s_win)
            Y.append(label)
            DELTA.append(delta)
            W0.append(w0)
            names.append(name)
            times.append(reg["time"].iloc[end_in - 1])

    if not Y:
        raise ValueError("没有构造出任何 RI 样本，检查窗口长度与阈值")

    out = {
        "s": np.stack(S).astype(np.float32),
        "y": np.asarray(Y, dtype=np.int64),
        "delta": np.asarray(DELTA, dtype=np.float32),
        "w0": np.asarray(W0, dtype=np.float32),
        "name": np.array(names),
        "end_time": np.array([str(t) for t in times]),
    }
    if cfg.use_patch:
        out["p"] = np.stack(P).astype(np.float32)
        from .patch_features import stack_patch_features
        out["pfeat"] = stack_patch_features(out["p"]).astype(np.float32)

    pos = int(out["y"].sum())
    print(f"[RI] 样本 {len(out['y'])}, 正样本 {pos} ({pos / len(out['y']) * 100:.1f}%), "
          f"阈值 {cfg.ri_threshold_ms} m/s, 窗口 {cfg.in_len}->{cfg.horizon} 步"
          + (f", 云图 {out['p'].shape}" if cfg.use_patch else "")
          + (f", 跳过(缺帧) {skipped_patch}" if skipped_patch else ""))
    return out


def class_weights(y: np.ndarray) -> tuple[float, float]:
    """返回 (负类权重, 正类权重)，用于处理 RI 样本稀少。"""
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - pos))
    w_pos = len(y) / (2.0 * pos)
    w_neg = len(y) / (2.0 * neg)
    return w_neg, w_pos
