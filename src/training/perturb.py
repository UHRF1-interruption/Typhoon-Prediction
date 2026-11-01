"""集合扰动的物理约束：振幅有界 + 相互正交 + 增长优化。

依据 Li, Duan, Li et al. 2026 (Adv. Atmos. Sci.)：O-CNOPs 为 AI 模式生成集合
成员时，扰动必须 ① 振幅有界、② 相互正交、③ 使非线性发展最大。无界扰动会产生
物理上不可能、概率上不可信的成员；不正交则成员重复同一个不稳定方向。

与多模型集成（§4.10）是两条路线：那里用不同**模型**提供多样性，这里用同一
**模型** + 受约束的**初始扰动**提供多样性。为什么振幅必须约束：§4.10 的判据
x = σ_弱/σ_强 < −ρ+√(ρ²+3) 要求成员质量相当，无界扰动等于批量制造差成员。

扰动加在**输入窗口最后一步的特征向量**上（dim 维），振幅约束加在前两维
（dlat/dlon）对应的公里位移上。
"""
from __future__ import annotations

import torch

from .physics import KM_PER_DEG, _cos_lat


def amp_km(delta: torch.Tensor, lat) -> torch.Tensor:
    """扰动在位置分量上的公里模长。delta: (..., F)，前两维为 dlat/dlon。

    lat 传标量 → 逐向量用同一基准；传 (B,) → 逐样本换算。
    """
    lat = torch.as_tensor(lat, dtype=delta.dtype, device=delta.device)
    dy = delta[..., 0] * KM_PER_DEG
    dx = delta[..., 1] * KM_PER_DEG * _cos_lat(lat)
    return torch.sqrt(dy ** 2 + dx ** 2)


def scale_to_eps(vec: torch.Tensor, eps_km: float, lat_ref: torch.Tensor) -> torch.Tensor:
    """缩放到位移模长恰为 eps_km（可微，供增长优化内层使用）。"""
    return vec * (eps_km / amp_km(vec, lat_ref).max().clamp_min(1e-9))


def _orthonormal(prev: list[torch.Tensor], dim: int, n: int, seed: int) -> torch.Tensor:
    """与 prev 张成空间正交的 n 个标准正交基（在补空间里随机取）。"""
    g = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    for _ in range(n):
        v = torch.randn(dim, generator=g)
        for u in list(prev) + out:
            for _ in range(2):                      # 两次投影，数值更稳
                v = v - (v @ u) * u
        nv = v.norm()
        if float(nv) > 1e-6:
            out.append(v / nv)
    return torch.stack(out, 0)


def random_bounded(dim: int, eps_km: float, k: int, lat_ref: torch.Tensor,
                   seed: int = 0) -> torch.Tensor:
    """k 个**随机**正交扰动，返回 (k, dim)，每个位移模长 = eps_km。"""
    basis = _orthonormal([], dim, k, seed)
    return torch.stack([scale_to_eps(basis[i], eps_km, lat_ref) for i in range(k)])


def orthogonality_error(p: torch.Tensor) -> float:
    """最大绝对余弦相似度（0 = 完全正交）。"""
    v = p.reshape(p.shape[0], -1)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    g = v @ v.T
    return float((g - torch.diag(torch.diag(g))).abs().max())


def members(x: torch.Tensor, deltas: torch.Tensor,
            pos_std: torch.Tensor | None = None) -> torch.Tensor:
    """把 (k, F) **物理**扰动（度）加到最后一步特征上，返回 (k, B, in_len, F)。

    x 是**已标准化**的特征，所以扰动必须先除以特征标准差再相加。
    漏掉这一步会让实际扰动被 std 放大数倍（实测 24h 误差从 208 涨到 850 km）。
    pos_std 传 spec.std 的前两维即可（只有 dlat/dlon 非零）。
    """
    d = deltas if pos_std is None else deltas / torch.as_tensor(
        pos_std, dtype=deltas.dtype, device=deltas.device)
    xr = x.unsqueeze(0).repeat(d.shape[0], 1, 1, 1).clone()
    xr[:, :, -1, :] += d.view(d.shape[0], 1, -1)
    return xr


def grow(fn, x: torch.Tensor, lat: torch.Tensor, eps_km: float, k: int,
         steps: int = 15, lr: float = 0.5, seed: int = 0,
         pos_std: torch.Tensor | None = None) -> torch.Tensor:
    """逐方向增长优化，正交性与振幅由**构造**保证（O-CNOP 顺序结构）。

    fn(x_members, lat) -> (k, B, out_len, 2)。目标为 CNOP 原义：
    最大化**相对控制预报的发展量** ‖f(x+δ)−f(x)‖。

    两条设计要点都是实测踩出来的：
    * 在**正交补基里优化系数**（d = c @ Q），而非"先 Adam 再投影"——
      后者 Adam 的一二阶矩与投影后参数不一致，实测正交偏差 1.0、
      离散度反降 52%。
    * 目标不能用"跨样本 std"：常数平移不改变跨样本离散度，优化方向无意义。
    """
    F = x.shape[-1]
    lat_ref = lat.reshape(-1).mean()
    with torch.no_grad():
        ctrl = fn(x.unsqueeze(0), lat)[0]
    basis: list[torch.Tensor] = []
    out: list[torch.Tensor] = []
    for j in range(k):
        Q = _orthonormal(basis, F, max(1, F - j), seed + 17 * j)
        c = torch.randn(Q.shape[0],
                        generator=torch.Generator().manual_seed(seed + j))
        c = c.clone().requires_grad_(True)
        opt = torch.optim.Adam([c], lr=lr)
        best_gain, best_u = -1e9, None
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            d = scale_to_eps(c @ Q, eps_km, lat_ref)
            pred = fn(members(x, d.unsqueeze(0), pos_std), lat)[0]
            gain = ((pred[..., -1, :] - ctrl[..., -1, :]) ** 2).sum(-1).mean()
            (-gain).backward()
            opt.step()
            if float(gain.detach()) > best_gain:
                best_gain = float(gain.detach())
                best_u = d.detach() / d.detach().norm().clamp_min(1e-12)
        if best_u is None:
            best_u = Q[0] / Q[0].norm().clamp_min(1e-12)
        basis.append(best_u)
        out.append(scale_to_eps(best_u, eps_km, lat_ref))
    return torch.stack(out, 0)
