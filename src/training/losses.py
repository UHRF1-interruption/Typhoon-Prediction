"""损失函数与评估指标。

关键点：用 Haversine 球面距离作为损失与主指标，而不是 MSE。
直接对经纬度做 MSE 会在低纬-高纬之间尺度不公平，且易受异常点主导。
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.features import haversine_km


def haversine_torch_km(lat1, lon1, lat2, lon2) -> torch.Tensor:
    """torch 版的球面距离，支持广播，单位为 km。"""
    r = 6371.0088
    lat1, lon1, lat2, lon2 = [torch.deg2rad(v) for v in (lat1, lon1, lat2, lon2)]
    dlat = lat2 - lat1
    # 经度差做环形约束，防止跨 180 线时距离爆炸
    dlon = (lon2 - lon1 + torch.pi) % (2 * torch.pi) - torch.pi
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * r * torch.arcsin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))


def delta_to_abs(delta: torch.Tensor, last_lat: torch.Tensor, last_lon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """增量 -> 绝对坐标。delta: (B, T, 2)，last_*: (B,)。"""
    lat = last_lat.unsqueeze(1) + delta[..., 0]
    lon = (last_lon.unsqueeze(1) + delta[..., 1]) % 360.0
    return lat, lon


def compute_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor,
                 last_lat: torch.Tensor, last_lon: torch.Tensor,
                 kind: str = "haversine", huber_delta_km: float = 100.0) -> torch.Tensor:
    """训练损失。

    kind='haversine': 先把增量转绝对坐标，再算球面距离的均值(km)。
        Huber 化处理：对超过 huber_delta_km 的误差做平方收缩，抑制离群轨迹
        主导梯度（台风路径偶尔有异常转向数据）。
    kind='huber': 直接在增量上做 SmoothL1。
    """
    if kind == "huber":
        return torch.nn.functional.smooth_l1_loss(pred_delta, target_delta)

    lat_p, lon_p = delta_to_abs(pred_delta, last_lat, last_lon)
    lat_t, lon_t = delta_to_abs(target_delta, last_lat, last_lon)
    dist = haversine_torch_km(lat_p, lon_p, lat_t, lon_t)

    # 伪 Huber：distance^2 / (distance + delta) 形式，远误差增长变缓
    d = huber_delta_km
    loss = torch.where(dist <= d, dist, 2 * torch.sqrt(dist * d) - d)
    return loss.mean()


# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------
def absolute_from_delta(delta: np.ndarray, last_abs: np.ndarray) -> np.ndarray:
    """numpy 版增量->绝对坐标。delta:(N,T,2), last_abs:(N,2) -> (N,T,2)"""
    lat = last_abs[:, None, 0] + delta[..., 0]
    lon = (last_abs[:, None, 1] + delta[..., 1]) % 360.0
    return np.stack([lat, lon], axis=-1)


def error_km(pred_abs: np.ndarray, true_abs: np.ndarray) -> np.ndarray:
    """逐样本逐时刻的球面误差 (N, T)。"""
    return haversine_km(
        pred_abs[..., 0], pred_abs[..., 1],
        true_abs[..., 0], true_abs[..., 1],
    )


def evaluate_predictions(pred_abs: np.ndarray, true_abs: np.ndarray,
                         interval_hours: float = 6.0) -> dict:
    """计算各时刻的路径误差指标。

    返回:
      mae_by_step  : 每个预测步的平均误差 (km)
      rmse_by_step : 每个预测步的均方根误差 (km)
      horizon      : 对应的预报时效 (小时)
      mae_overall  : 全部步平均误差
      lat_err / lon_err: 分方向误差（度）
    """
    dist = error_km(pred_abs, true_abs)
    mae_by_step = dist.mean(axis=0)
    rmse_by_step = np.sqrt((dist ** 2).mean(axis=0))
    steps = np.arange(1, dist.shape[1] + 1)
    return {
        "mae_by_step": mae_by_step,
        "rmse_by_step": rmse_by_step,
        "horizon": steps * interval_hours,
        "mae_overall": float(dist.mean()),
        "lat_mae": float(np.abs(pred_abs[..., 0] - true_abs[..., 0]).mean()),
        "lon_mae": float(np.abs((pred_abs[..., 1] - true_abs[..., 1] + 180) % 360 - 180).mean()),
        "n_samples": int(dist.shape[0]),
    }


def format_metrics_table(results: dict[str, dict], interval_hours: float = 6.0) -> str:
    """把多个模型的指标整理成 markdown 表格。"""
    if not results:
        return "(无结果)"
    horizon = next(iter(results.values()))["horizon"]
    header = "| 模型 | " + " | ".join(f"{int(h)}h 误差(km)" for h in horizon) + " | 平均(km) |"
    sep = "|" + "---|" * (len(horizon) + 2)
    lines = [header, sep]
    for name, m in results.items():
        cells = " | ".join(f"{v:.1f}" for v in m["mae_by_step"])
        lines.append(f"| {name} | {cells} | {m['mae_overall']:.1f} |")
    return "\n".join(lines)
