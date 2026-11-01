"""无标签（label-free）的物理约束：作为软惩罚加入总损失。

和 `physics.py` 的关键区别
--------------------------
`physics.py` 里的 velocity / accel / turn / endpoint **全部引用 `true_delta`**，
本质是"与真值一致性的辅助损失"——它和主损失在优化同一件事，只是换了度量方式。
所以那个版本不能叫物理约束，其收益有限（§4.11.3 仅回收 1.1 km）也就可以理解。

本模块只用**预测输出 + 输入特征**，**不接触标签**。因此它编码的是
"解必须像物理系统"，而不是"解必须接近答案"。这类项才能提供主损失之外的
独立约束方向。

三项（路径任务）
----------------
1. **引导流一致（steering）**
   台风的逐时效移动速度应近似等于环境引导气流：
       v_t ≈ k · u_env
   `u_env` 取 500 hPa 风（`u500`/`v500`，经典引导层），`k` 是**可学标量**
   （引导系数，通常 <1）。不预设 k=0.8 之类的常数——**由训练确定并报告**，
   避免自创公式。
2. **轨迹平滑（smooth）**
   TC 中心位置对时间是 C¹ 的，速度不应逐时效大幅跳变：
       惩罚位置的二阶差分 ‖p_{t+1} − 2p_t + p_{t-1}‖
3. **曲率上界（curvature）**
   转向角有物理上界。实测 6h 相邻段夹角 p99.9 = 90°（见 physics.TURN_ABS_DEG），
   超过即惩罚：relu(θ_pred − 90°)。

全部按特征尺度归一化成 O(1)，再用 `GradNormBalancer` 或固定权重组合。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .physics import (KM_PER_DEG, SCALE_ACC_KM, SCALE_VEL_KM, TURN_ABS_DEG,
                      _cos_lat)


@dataclass
class LawCtx:
    """把"从标准化特征里还原物理量"所需的元信息打包。

    特征 x 已标准化，必须用 spec.mean/std 还原成物理单位（m/s、度）。
    找不到 u500/v500 时 u_idx/v_idx 为 None，引导流项自动跳过。
    """

    mean: torch.Tensor          # (F,)
    std: torch.Tensor           # (F,)
    u_idx: int | None = None
    v_idx: int | None = None
    interval_h: float = 6.0

    @property
    def has_steering(self) -> bool:
        return self.u_idx is not None and self.v_idx is not None


def build_ctx(names: list[str], mean, std, interval_h: float = 6.0) -> LawCtx:
    """从特征名与标准化统计量构造上下文。"""
    idx = {n: i for i, n in enumerate(names)}
    return LawCtx(
        mean=torch.as_tensor(mean, dtype=torch.float32),
        std=torch.as_tensor(std, dtype=torch.float32),
        u_idx=idx.get("u500"),
        v_idx=idx.get("v500"),
        interval_h=float(interval_h),
    )


def _steer_km(x: torch.Tensor, ctx: LawCtx) -> torch.Tensor:
    """输入最后一步的 500hPa 风，换算成与"每步位移"同量纲的 (dy,dx) km。"""
    # 还原标准化：physical = z * std + mean（不是 (z - mean)/std）
    u = x[:, -1, ctx.u_idx] * ctx.std[ctx.u_idx] + ctx.mean[ctx.u_idx]
    v = x[:, -1, ctx.v_idx] * ctx.std[ctx.v_idx] + ctx.mean[ctx.v_idx]
    per_step = 3.6 * ctx.interval_h
    return torch.stack([v * per_step, u * per_step], dim=-1)


def _pred_velocity_km(pred_delta: torch.Tensor, last_lat: torch.Tensor):
    """预测轨迹的逐时效速度（km/步）。

    pred_delta 是相对最后输入点的**累积**位移，故位置 p_t = pred_delta[t]，
    p_0 = 0，速度 v_t = p_t − p_{t-1}。
    返回 (B, out_len, 2) 的 (dy, dx)，单位 km。
    """
    p = pred_delta
    prev = torch.cat([torch.zeros_like(p[:, :1]), p[:, :-1]], dim=1)
    v = p - prev                                        # (B,T,2) 度
    lat = last_lat.unsqueeze(1)
    dy = v[..., 0] * KM_PER_DEG
    dx = v[..., 1] * KM_PER_DEG * _cos_lat(lat)
    return torch.stack([dy, dx], dim=-1)


def fit_steer_k(pred_true_first: torch.Tensor, x: torch.Tensor,
                last_lat: torch.Tensor, ctx: LawCtx) -> tuple[float, float]:
    """从**训练集**用闭式最小二乘定标引导系数 k，并返回 R²。

        k = (Σ v·u) / (Σ u·u)      （过原点，因为速度与引导同向且同量纲）

    为什么不在训练中让 k 自由学习：`v_pred ≈ k·u` 的联合优化里 k→0 是一个
    退化解（等价于"不要动"），实测 3 epoch 就把 k 从 1.000 压到 0.005，
    惩罚随之退化成速度收缩正则，与物理无关。故改为**由数据一次定标**并固定。

    ⚠️ 必须在**有真实环境场**的样本上拟合。本项目轨迹 1420 个台风、
    环境场只覆盖 128 个（约 8% 的窗口）；若对填充零拟合，R² 会从 0.69 掉到 0.06。
    """
    if not ctx.has_steering:
        raise ValueError("无 u500/v500，无法估计 k")
    v = _pred_velocity_km(pred_true_first.unsqueeze(1), last_lat)[:, 0]  # (B,2)
    ste = _steer_km(x, ctx)                                             # (B,2)
    num = (v * ste).sum()
    den = (ste * ste).sum().clamp_min(1e-12)
    k = float(num / den)
    resid = ((v - k * ste) ** 2).sum()
    total = (v ** 2).sum().clamp_min(1e-12)
    return k, float(1.0 - resid / total)


def steering_penalty(pred_delta: torch.Tensor, x: torch.Tensor,
                     last_lat: torch.Tensor, ctx: LawCtx,
                     k: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """引导流一致性：v_t ≈ k · u_env。

    返回 (归一化惩罚, k)。k 是标量参数（引导系数），需由调用方传入可学参数。
    u_env 取输入窗口**最后一步**的 500hPa 风（引导气流的当前值）。
    500hPa 风的单位是 m/s，换算成"每步位移"要 × 3.6 × interval_h。
    """
    if not ctx.has_steering:
        raise ValueError("特征里没有 u500/v500，无法计算引导流约束")
    k = torch.as_tensor(1.0 if k is None else k,
                        dtype=pred_delta.dtype, device=pred_delta.device)
    steer = _steer_km(x, ctx)                             # (B,2) (dy,dx)

    v_pred = _pred_velocity_km(pred_delta, last_lat)      # (B,T,2)
    diff = v_pred - k * steer.unsqueeze(1)                # (B,T,2)
    mse = (diff ** 2).sum(-1).mean()
    return torch.sqrt(mse + 1e-12) / SCALE_VEL_KM, k


def smooth_penalty(pred_delta: torch.Tensor, last_lat: torch.Tensor) -> torch.Tensor:
    """轨迹平滑：惩罚位置的二阶差分（=速度的变化），单位 km，按 SCALE_ACC_KM 归一化。"""
    if pred_delta.size(1) < 3:
        return pred_delta.new_zeros(())
    p = pred_delta
    lat = last_lat.unsqueeze(1)
    dy = p[..., 0] * KM_PER_DEG
    dx = p[..., 1] * KM_PER_DEG * _cos_lat(lat)
    acc_y = dy[:, 2:] - 2 * dy[:, 1:-1] + dy[:, :-2]
    acc_x = dx[:, 2:] - 2 * dx[:, 1:-1] + dx[:, :-2]
    return torch.sqrt((acc_y ** 2 + acc_x ** 2).mean() + 1e-12) / SCALE_ACC_KM


def curvature_penalty(pred_delta: torch.Tensor, last_lat: torch.Tensor,
                      max_deg: float = TURN_ABS_DEG) -> torch.Tensor:
    """曲率上界：相邻两段位移的夹角超过 max_deg 才惩罚。单位：度（除以 max_deg 归一化）。"""
    if pred_delta.size(1) < 2:
        return pred_delta.new_zeros(())
    v = _pred_velocity_km(pred_delta, last_lat)           # (B,T,2)
    v1, v2 = v[:, :-1], v[:, 1:]
    n1 = torch.sqrt((v1 ** 2).sum(-1) + 1e-6)
    n2 = torch.sqrt((v2 ** 2).sum(-1) + 1e-6)
    cosang = torch.clamp((v1 * v2).sum(-1) / (n1 * n2), -1.0, 1.0)
    ang = torch.rad2deg(torch.arccos(cosang))
    return torch.clamp(ang - max_deg, min=0.0).mean() / max_deg


def law_loss(pred_delta: torch.Tensor, x: torch.Tensor, last_lat: torch.Tensor,
             ctx: LawCtx, k: torch.Tensor | None = None,
             weights: dict | None = None) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
    """三项无标签物理惩罚之和。

    weights 缺省全 1。返回 (总惩罚, 各分项, k)；无引导流特征时 steering 项为 0。
    """
    terms: dict[str, torch.Tensor] = {}
    k_out = None
    if ctx.has_steering:
        s, k_out = steering_penalty(pred_delta, x, last_lat, ctx, k)
        terms["steering"] = s
    terms["smooth"] = smooth_penalty(pred_delta, last_lat)
    terms["curvature"] = curvature_penalty(pred_delta, last_lat)
    w = weights or {}
    total = sum(terms[n] * float(w.get(n, 1.0)) for n in terms)
    return total, terms, k_out
