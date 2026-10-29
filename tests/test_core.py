"""核心组件的轻量自检（不依赖 pytest，直接 python tests/test_core.py 运行）。

重点验证容易出错的地方：球面几何、跨 180 度经线、滑动窗口形状与标签一致性、
模型输出契约、以及损失函数在"预测=真值"时应为 0。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.features import (  # noqa: E402
    bearing_deg, build_samples, dest_point, haversine_km, wrap_dlon,
)
from src.data.synthetic import generate_synthetic_storms  # noqa: E402
from src.models.typhoon_transformer import (  # noqa: E402
    GRUBaseline, PersistenceBaseline, TyphoonTransformer, count_parameters,
)
from src.training.losses import absolute_from_delta, compute_loss, error_km  # noqa: E402

CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("haversine: 赤道上 1 度经度 ≈ 111.19 km")
def _():
    d = haversine_km(0.0, 0.0, 0.0, 1.0)
    assert abs(d - 111.19) < 0.1, d


@check("haversine: 同点为 0")
def _():
    assert haversine_km(20.0, 130.0, 20.0, 130.0) == 0.0


@check("wrap_dlon: 跨 180 线不产生 358 度跳变")
def _():
    assert abs(wrap_dlon(170.0 - 190.0) - (-20.0)) < 1e-9
    assert abs(wrap_dlon(-179.0 - 179.0) - 2.0) < 1e-9


@check("dest_point: 正北走 111.19km 纬度约 +1 度")
def _():
    lat, lon = dest_point(10.0, 130.0, 0.0, 111.19)
    assert abs(float(lat) - 11.0) < 0.01, lat
    assert abs(float(lon) - 130.0) < 0.01, lon


@check("bearing_deg: 正东方向为 90 度")
def _():
    b = bearing_deg(10.0, 130.0, 10.0, 131.0)
    assert abs(float(b) - 90.0) < 0.5, b


@check("build_samples: 形状与标签自洽（末输入点 + delta == 绝对目标）")
def _():
    df = generate_synthetic_storms(n_storms=30, points_per_storm=30, n_years=3,
                                   start_year=2010, seed=1)
    b = build_samples(df, in_len=6, out_len=2, feature_names=[
        "dlat", "dlon", "dist_prev", "sin_heading", "cos_heading",
        "wind_ms", "dpres_hpa", "lat_norm"])
    assert b["x"].shape[1:] == (6, 8), b["x"].shape
    assert b["y"].shape[1:] == (2, 2), b["y"].shape
    assert b["valid_mask"].shape[1] == 6
    recon = absolute_from_delta(b["y"], b["last_abs"])
    diff = np.abs(recon[..., 0] - b["y_abs"][..., 0])
    assert diff.max() < 1e-4, diff.max()
    dlon = (recon[..., 1] - b["y_abs"][..., 1] + 180) % 360 - 180
    assert np.abs(dlon).max() < 1e-4, np.abs(dlon).max()


@check("模型: 三个模型的输出契约都是 (B, out_len, 2)")
def _():
    B, F, in_len, out_len = 4, 8, 7, 3
    x = torch.randn(B, in_len, F)
    mask = torch.zeros(B, in_len, dtype=torch.bool)
    mask[0, :2] = True
    for model in (
        TyphoonTransformer(F, out_len, d_model=32, nhead=4, num_encoder_layers=2,
                           num_decoder_layers=1, dim_feedforward=64),
        GRUBaseline(F, out_len, hidden=16, num_layers=1, pred_mode="delta"),
        PersistenceBaseline(F, out_len),
    ):
        out = model(x, key_padding_mask=mask)
        assert out.shape == (B, out_len, 2), (type(model).__name__, out.shape)
        assert torch.isfinite(out).all(), type(model).__name__


@check("Transformer: 零初始化输出头 -> 初始预测恒为 0 增量（等于持续法）")
def _():
    m = TyphoonTransformer(8, 4, d_model=32, nhead=4, num_encoder_layers=1,
                           num_decoder_layers=1, dim_feedforward=32)
    out = m(torch.randn(2, 5, 8))
    assert torch.allclose(out, torch.zeros_like(out)), out


@check("loss: 预测等于真值时损失为 0，且梯度有限")
def _():
    last = torch.tensor([[15.0, 130.0], [20.0, 160.0]])
    target = torch.tensor([[[1.0, 2.0], [2.5, 4.0]], [[-1.0, -3.0], [0.5, 1.0]]])
    pred = target.clone().requires_grad_(True)
    loss = compute_loss(pred, target, last[:, 0], last[:, 1], kind="haversine")
    assert loss.detach().item() < 1e-6
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@check("error_km: 经度跨 180 线不会算出巨大误差")
def _():
    pred = np.array([[[20.0, 359.0]]])     # 359 度 == -1 度
    true = np.array([[[20.0, 1.0]]])
    err = error_km(pred, true)
    assert err.max() < 250.0, err   # 2 度经度差 ≈ 200km 量级


@check("参数量统计可用")
def _():
    m = TyphoonTransformer(8, 4, d_model=64, nhead=4, num_encoder_layers=2,
                           num_decoder_layers=1, dim_feedforward=128)
    assert count_parameters(m) > 0


@check("CMA 解析: 时间/坐标/单位按列正确解码")
def _():
    from src.data.cma import _parse_observation
    rec = _parse_observation("2004040514 2  88 1489  3.0      23")
    assert rec is not None
    assert rec["lat"] == 8.8 and abs(rec["lon"] - 148.9) < 1e-6, rec
    assert abs(rec["wind"] - 3.0) < 1e-6, rec
    assert rec["time"].year == 2004 and rec["time"].month == 4 and rec["time"].hour == 14


@check("CMA 解析: 非法行返回 None")
def _():
    from src.data.cma import _parse_observation
    assert _parse_observation("66666 0000   45 0003 0401 1 6 Sudal") is None
    assert _parse_observation("") is None


@check("预报误差表: 可解析并计算分时效均值")
def _():
    from src.data.forecast_errors import summarize_baselines
    import pandas as pd
    tra = pd.DataFrame({
        "method": ["ecmf"] * 4, "label": ["EC"] * 4,
        "lead": [24, 24, 72, 72], "tra_err": [50.0, 60.0, 140.0, 160.0],
    })
    int_ = pd.DataFrame({"method": ["ecmf"] * 4, "lead": [24, 24, 72, 72],
                         "int_err_abs": [5.0, 7.0, 9.0, 11.0]})
    out = summarize_baselines(tra, int_, [24, 72])
    assert abs(float(out.loc[0, "tra_24h_km"]) - 55.0) < 1e-6
    assert abs(float(out.loc[0, "int_72h_ms"]) - 10.0) < 1e-6


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    total = len(CHECKS)
    print(f"\n{total - failed}/{total} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
