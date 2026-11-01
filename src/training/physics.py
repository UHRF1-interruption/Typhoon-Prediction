"""路径预测的物理约束：恒速先验 + 有界残差 + 物理损失 + 梯度均衡。

设计依据（三处外部文献 + 本项目实测）
--------------------------------------
1. **StormFormer**（run_stormformer.py, `stormformer_loss`）：
   预测 = 恒速先验 + tanh(网络输出)·max_residual，损失含 MSE/Haversine/
   速度/加速度/端点五项，且各项权重是手工常数（0.012/0.35/0.10/0.20）。

2. **WHC-PINN**（Ghoreishi & Naderan 2026, Sci Rep 16:4201）：
   硬约束的标准形式是 `û = g(x) + ℓ(x)·N(x;θ)`，其中 ℓ=0 在边界上。
   其关键警告——**ℓ 在边界上归零意味着网络项在那里完全不起作用**。
   ℓ 归零处梯度也归零，网络学不到东西；所以硬约束会引入**梯度失衡**，
   该文用梯度加权（λ = 1/(ε(|∇·u|−∇·u)+1)）压制高梯度区来平衡。

3. **Li et al. 2026 (AAS)**：集合成员由**有振幅约束（ε）+ 正交**的
   最优扰动生成，而非自由随机扰动。

本项目实测（`tracks_WP_clean.parquet`，20094 个滑窗，in=72h/out=24h/6h 步长）
--------------------------------------------------------------------------
    4 时效平均        24h
    持续法    338.8 km       544.5 km
    恒速先验  145.6 km       257.6 km     ← 比持续法好 43%，但不如学习模型（119.0）
    已发布 Transformer 119.0 km       208.4 km

  残差（真值 − 恒速先验）幅值 p99：6h 199 / 12h 434 / 18h 703 / 24h 996 km。

**因此"强先验 + 学残差"在本任务是合适的**：先验已把误差压到持续法的 43%，
模型只需学剩下的修正量。

⚠️ 本模块最需要注意的一点：**cap 设得太紧会让 tanh 饱和 → 梯度趋零**，
效果反过来（正是 WHC-PINN 描述的失衡）。所以：
  * cap 按时效由数据定（p99），不用一个手挑的常数；
  * 提供 `saturation_rate` 诊断，饱和率超过约 5% 就说明 cap 过紧。
"""
from __future__ import annotations

import torch

from .losses import haversine_torch_km

# 一度纬度对应的公里数（球面平均）
KM_PER_DEG = 111.32

# ---------------------------------------------------------------------------
# 特征尺度（全部来自**项目自己的窗口构造**：20094 个样本，in=72h/out=24h）
# ---------------------------------------------------------------------------
# 为什么需要这些：Rohrhofer et al. 2023 (IEEE Access) 指出，物理量纲会
# **直接缩放各损失项**，造成多目标失衡——同样的权重换一量纲就失效。
# 因此各项先按特征尺度归一化成无量纲相对量，权重才有可比性。
SCALE_VEL_KM = 115.4        # 观测 6h 位移 p50
SCALE_ACC_KM = 118.6        # 观测相邻位移变化 p50
SCALE_TURN_DEG = 16.8       # 观测转向角 p95
SCALE_ENDPOINT_KM = 145.6   # 恒速先验的平均误差

# 有界残差的 cap：残差（真值−先验）幅值 p99
RESIDUAL_CAP_KM = (199.0, 434.0, 703.0, 996.0)

# 转向角绝对上限：项目窗口上 p99.9=90.0°，仅 0.098% 的样本超过。
# （早先用未清洗的原始 3h 数据算得 162°，与项目窗口口径不一致，已纠正。）
TURN_ABS_DEG = 90.0


def _cos_lat(lat_deg: torch.Tensor) -> torch.Tensor:
    """纬度余弦，并防止极点附近除零。"""
    return torch.clamp(torch.cos(torch.deg2rad(lat_deg)), min=0.15)


def deg_to_km(dlat: torch.Tensor, dlon: torch.Tensor, lat_deg: torch.Tensor):
    """经纬度增量 -> 局地平面位移 (dy_km, dx_km)。lat_deg 为基准纬度。"""
    dy = dlat * KM_PER_DEG
    dx = dlon * KM_PER_DEG * _cos_lat(lat_deg)
    return dy, dx


def km_to_deg(dy: torch.Tensor, dx: torch.Tensor, lat_deg: torch.Tensor):
    """局地平面位移 -> 经纬度增量。"""
    dlat = dy / KM_PER_DEG
    dlon = dx / (KM_PER_DEG * _cos_lat(lat_deg))
    return dlat, dlon


def constant_velocity_prior(x: torch.Tensor, out_len: int,
                            weights: tuple[float, ...] = (0.2, 0.3, 0.5)):
    """用最近 3 步的加权平均速度外推，返回**增量形式的先验**。

    x: (B, in_len, F)，约定 x[:,:,0]=dlat(deg)、x[:,:,1]=dlon(deg)、
       并且第 2 列特征是 wind 之前的轨迹列（本项目 features 约定）。
    返回 (B, out_len, 2) 的增量（deg），与模型的 delta 输出同一语义。

    权重 (0.2,0.3,0.5) 取自 StormFormer：越近的位移权重越大。
    """
    dl = x[:, -len(weights):, 0]
    dn = x[:, -len(weights):, 1]
    w = torch.tensor(weights, device=x.device, dtype=x.dtype).view(1, -1)
    v_lat = (dl * w).sum(dim=1)
    v_lon = (dn * w).sum(dim=1)
    steps = torch.arange(1, out_len + 1, device=x.device, dtype=x.dtype).view(1, -1)
    return torch.stack([v_lat.unsqueeze(1) * steps,
                        v_lon.unsqueeze(1) * steps], dim=-1)


def bounded_residual(raw: torch.Tensor, lat_last: torch.Tensor,
                     cap_km: torch.Tensor | tuple[float, ...]):
    """把网络原始输出变成**模长有界**的残差增量（度）。

    先验是物理底座，网络只能在其上叠加一个受限制的修正：

        r = tanh(|v|) · cap · v/|v|        （|r| ≤ cap，方向自由）

    注意是约束**模长**而不是逐分量：逐分量约束会让矢量最大到 cap·√2，
    界就不成立了（实测过 281 km > 199 km 的越界）。

    ⚠️ cap 过紧会导致 tanh 饱和、梯度消失——用 saturation_rate 监控。

    raw: (B, out_len, 2)，当作 km 向量。
    """
    if not torch.is_tensor(cap_km):
        cap_km = torch.tensor(cap_km, device=raw.device, dtype=raw.dtype)
    cap = (cap_km.view(1, -1) if cap_km.dim() == 1 else cap_km).unsqueeze(-1)
    vy, vx = raw[..., 0], raw[..., 1]
    mag = torch.sqrt(vy ** 2 + vx ** 2 + 1e-12)
    # |r| = tanh(mag)·cap，方向与 raw 一致；mag→0 时 scale→1，无奇点
    scale = torch.tanh(mag) * cap.squeeze(-1) / mag
    dy, dx = vy * scale, vx * scale
    dlat, dlon = km_to_deg(dy, dx, lat_last.unsqueeze(1))
    return torch.stack([dlat, dlon], dim=-1)


def saturation_rate(raw: torch.Tensor, thresh: float = 0.95) -> float:
    """tanh 进入饱和区的比例——**梯度失衡的直接诊断量**。

    |tanh(raw)| > thresh 时导数 < 0.1，该样本几乎不再提供梯度。
    实测经验：超过约 5% 说明 cap 太紧，应放宽或改用软约束。
    """
    return float((torch.tanh(raw).abs() > thresh).float().mean().item())


# ---------------------------------------------------------------------------
# 物理损失项
# ---------------------------------------------------------------------------
def physics_terms(pred_delta: torch.Tensor, true_delta: torch.Tensor,
                  last_lat: torch.Tensor, last_lon: torch.Tensor,
                  turn_abs_deg: float = TURN_ABS_DEG) -> dict:
    """四项物理量，**全部换算成可比较的物理单位**（km 或度）。

    velocity  逐时效位移偏差的 RMS（km）
    accel     相邻步位移变化偏差的 RMS（km）
    turn      **主项只惩罚比观测更尖锐的转向**（度）：relu(角_pred − 角_true)。
              不能无条件惩罚锐角——真实台风确实会急转（实测 6h 转向角
              p95=43°、p99=91°、p99.9=162°），无条件平滑会把最难的样本按平，
              与 CNOP 系列文献强调的"异常路径"正相反。
              仅保留一个 p99.9(90°) 的绝对上限，只拦真正荒谬的锯齿
              （项目窗口上命中 0.098%，不和数据打架）。
    endpoint  末时效球面距离（km）

    用 RMS 而不是均方，是为了四项量级可比（否则 km² 会主导梯度）。
    """
    lat_p = last_lat.unsqueeze(1) + pred_delta[..., 0]
    lon_p = last_lon.unsqueeze(1) + pred_delta[..., 1]
    lat_t = last_t = last_lat.unsqueeze(1) + true_delta[..., 0]
    lon_t = last_lon.unsqueeze(1) + true_delta[..., 1]

    cosl = _cos_lat(last_lat).unsqueeze(1)
    v_lat = (pred_delta[..., 0] - true_delta[..., 0]) * KM_PER_DEG
    v_lon = (pred_delta[..., 1] - true_delta[..., 1]) * KM_PER_DEG * cosl
    velocity = torch.sqrt((v_lat ** 2 + v_lon ** 2).mean() + 1e-12)

    if pred_delta.size(1) >= 2:
        acc_p = pred_delta[:, 1:] - pred_delta[:, :-1]
        acc_t = true_delta[:, 1:] - true_delta[:, :-1]
        a_lat = (acc_p[..., 0] - acc_t[..., 0]) * KM_PER_DEG
        a_lon = (acc_p[..., 1] - acc_t[..., 1]) * KM_PER_DEG * cosl
        accel = torch.sqrt((a_lat ** 2 + a_lon ** 2).mean() + 1e-12)
    else:
        accel = pred_delta.new_zeros(())

    if pred_delta.size(1) >= 2:
        def _ang(d):
            d1, d2 = d[:, :-1], d[:, 1:]
            lat_mid = last_lat.unsqueeze(1) + d1[..., 0]
            c = _cos_lat(lat_mid)
            y1 = d1[..., 0] * KM_PER_DEG
            x1 = d1[..., 1] * KM_PER_DEG * c
            y2 = d2[..., 0] * KM_PER_DEG
            x2 = d2[..., 1] * KM_PER_DEG * c
            n1 = torch.sqrt(y1 ** 2 + x1 ** 2 + 1e-6)
            n2 = torch.sqrt(y2 ** 2 + x2 ** 2 + 1e-6)
            cosang = torch.clamp((y1 * y2 + x1 * x2) / (n1 * n2), -1.0, 1.0)
            return torch.rad2deg(torch.arccos(cosang))

        ang_p, ang_t = _ang(pred_delta), _ang(true_delta)
        # 单向：预测比观测更尖锐才罚。另加一条绝对上限，
        # 防止在观测角也很大（打转）时放大出超物理的锯齿。
        turn = (torch.clamp(ang_p - ang_t, min=0.0)
                + torch.clamp(ang_p - turn_abs_deg, min=0.0)).mean()
    else:
        turn = pred_delta.new_zeros(())

    endpoint = haversine_torch_km(lat_p[:, -1], lon_p[:, -1], lat_t[:, -1], lon_t[:, -1]).mean()
    return {"velocity": velocity, "accel": accel, "turn": turn, "endpoint": endpoint}


# ---------------------------------------------------------------------------
# 梯度均衡（GradNorm 风格）
# ---------------------------------------------------------------------------
class GradNormBalancer:
    """按**梯度范数**自动平衡多个损失项，替代 StormFormer 那组手调常数。

    WHC-PINN 指出硬约束会带来梯度失衡；多损失项权重靠试凑时，训练早期
    某一项可能主导梯度，把共享主干拉向对它有利的方向。这里按 GradNorm
    (Chen et al. 2018) 的思路，让各项对**共享参数**的梯度范数趋于一致：

        w_i ← w_i · (ḡ / g_i)^α        ḡ = 各 g_i 的均值

    用 EMA 平滑并对权重做归一化，避免震荡。

    **参数是实测调出来的（2026-11-01）**：本项目四项的**未加权**梯度范数
    相差 4 个数量级——

        初始化时  velocity 8.3e+00  accel 3.3e+00  turn 4.4e-04  endpoint 7.8e+00

    turn 极小是因为残差零初始化时预测转向恰等于观测转向，该损失与梯度都 ≈0
    （这本身是设计正确的表现）。因此窗口不能开太大：
      * 窗口 [0.05, 50] + α=1.0（第一版）：turn 权重顶到 ~4，
        **反过来主导梯度**，失衡没有改善；
      * 窗口 **[0.2, 5] + α=0.5**（当前）：有效贡献 w·g 从相差约 3000×
        收敛到 **1.6× 以内**（实测 60 步，见下表）。

        step   有效贡献 w·g (velocity / accel / turn / endpoint)
           0     3.44e+00   2.14e+00   1.11e-03   3.18e+00
          60     1.53e+00   9.79e-01   1.27e+00   1.47e+00

    默认 `physics_adaptive=false`：等权 + 特征尺度归一化已可用；
    只有在观察到某项主导时才建议打开。
    """

    def __init__(self, names: list[str], alpha: float = 0.5,
                 ema: float = 0.9, w_min: float = 0.2, w_max: float = 5.0):
        self.names = list(names)
        self.alpha = alpha
        self.ema = ema
        self.w_min = w_min
        self.w_max = w_max
        self.w = {n: 1.0 for n in self.names}
        self.g = {n: None for n in self.names}

    def update(self, losses: dict, params: list[torch.Tensor]) -> dict:
        """losses: {name: 标量 tensor}；params: 参与共享的参数。

        返回更新后的权重（也存进 self.w）。
        """
        for n in self.names:
            if n not in losses or not losses[n].requires_grad:
                continue
            gn = torch.autograd.grad(losses[n], params, retain_graph=True,
                                     allow_unused=True)
            gn = [x for x in gn if x is not None]
            if not gn:
                continue
            norm = torch.sqrt(sum((x.detach() ** 2).sum() for x in gn)) + 1e-12
            prev = self.g[n]
            self.g[n] = float(norm) if prev is None else \
                self.ema * prev + (1 - self.ema) * float(norm)
        live = [n for n in self.names if self.g.get(n)]
        if live:
            mean_g = sum(self.g[n] for n in live) / len(live)
            for n in live:
                ratio = (mean_g / (self.g[n] + 1e-12)) ** self.alpha
                w = self.w[n] * ratio
                self.w[n] = float(min(max(w, self.w_min), self.w_max))
            s = sum(self.w[n] for n in live) / len(live)
            for n in live:
                w = self.w[n] / max(s, 1e-12)
                self.w[n] = float(min(max(w, self.w_min), self.w_max))
        return dict(self.w)


def physics_loss(pred_delta: torch.Tensor, true_delta: torch.Tensor,
                 last_lat: torch.Tensor, last_lon: torch.Tensor,
                 weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    """归一化后的物理损失总和 + 各分项（均无量纲）。

    归一化依据 Rohrhofer et al. 2023：物理量纲会缩放损失项，
    不归一化就必须为每个量纲重调权重。这里统一除以特征尺度，
    使"1.0"代表"误差达到典型量级"。

    weights 缺省全 1；可用 GradNormBalancer 自动给。
    """
    raw = physics_terms(pred_delta, true_delta, last_lat, last_lon)
    norm = {
        "velocity": raw["velocity"] / SCALE_VEL_KM,
        "accel": raw["accel"] / SCALE_ACC_KM,
        "turn": raw["turn"] / SCALE_TURN_DEG,
        "endpoint": raw["endpoint"] / SCALE_ENDPOINT_KM,
    }
    w = weights or {}
    total = sum(norm[k] * float(w.get(k, 1.0)) for k in norm)
    return total, norm
