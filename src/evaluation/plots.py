"""评估可视化：误差-时效曲线、典型路径对比图。

用 Matplotlib Agg 后端，无需图形界面即可在服务器上出图。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体：按可用性依次尝试，避免图上出现方块字
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def plot_mae_curve(results: dict[str, dict], interval_hours: float, out_path: str | Path) -> Path:
    """画各模型路径误差随时效变化的曲线。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (name, m) in enumerate(results.items()):
        ax.plot(m["horizon"], m["mae_by_step"], marker=markers[i % len(markers)],
                linewidth=1.8, label=f"{name} (avg {m['mae_overall']:.0f} km)")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Mean track error (km)")
    ax.set_title("Typhoon track error vs. lead time (test set)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] 误差曲线 -> {out_path}")
    return out_path


def _to_180(lon: np.ndarray) -> np.ndarray:
    return (lon + 180.0) % 360.0 - 180.0


def plot_track_examples(pred: dict, model_name: str, out_path: str | Path,
                        n_examples: int = 6, seed: int = 0) -> Path:
    """随机挑几个测试样本，画出观测轨迹 + 预测轨迹。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_abs = pred["pred_abs"]
    true_abs = pred["true_abs"]
    last_abs = pred["last_abs"]
    n = len(pred_abs)
    if n == 0:
        return out_path

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_examples, n), replace=False)
    cols = min(3, len(idx))
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), dpi=150,
                             squeeze=False)

    for ax, k in zip(axes.ravel(), idx):
        t_lat, t_lon = true_abs[k, :, 0], _to_180(true_abs[k, :, 1])
        p_lat, p_lon = pred_abs[k, :, 0], _to_180(pred_abs[k, :, 1])
        o_lat, o_lon = last_abs[k, 0], _to_180(np.array([last_abs[k, 1]]))[0]

        ax.plot(np.concatenate([[o_lon], t_lon]), np.concatenate([[o_lat], t_lat]),
                "o-", color="#1f77b4", markersize=4, label="observed")
        ax.plot(np.concatenate([[o_lon], p_lon]), np.concatenate([[o_lat], p_lat]),
                "s--", color="#d62728", markersize=4, label="predicted")
        ax.scatter([o_lon], [o_lat], marker="*", s=140, color="black", zorder=5,
                   label="last obs")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"sample #{k}", fontsize=9)
        ax.grid(alpha=0.3)
        # 等比例，避免经纬度拉伸视觉误导
        ax.set_aspect(1.0 / max(0.2, np.cos(np.radians(float(np.mean(t_lat))))))

    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Track examples on test set — {model_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] 路径示例 -> {out_path}")
    return out_path


def plot_training_curve(history: list[dict], out_path: str | Path) -> Path:
    """训练/验证损失曲线。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss_km"] for h in history], label="train loss (km)")
    val = [h["val_mae_km"] if h["val_mae_km"] is not None else np.nan for h in history]
    if np.isfinite(val).any():
        ax.plot(epochs, val, label="val MAE (km)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("km")
    ax.set_title("Training curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_error_curves(curves: dict[str, "pd.DataFrame"], value_col: str,
                      lead_range: tuple[float, float], xlabel: str, ylabel: str,
                      title: str, out_path: str | Path, ylog: bool = False) -> Path:
    """多方法误差-时效曲线。

    curves: {显示名: 含 'lead' 与 value_col 两列的 DataFrame}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    for i, (name, df) in enumerate(curves.items()):
        agg = df.groupby("lead")[value_col].mean()
        agg = agg[(agg.index >= lead_range[0]) & (agg.index <= lead_range[1])]
        if agg.empty:
            continue
        ax.plot(agg.index, agg.values, marker=markers[i % len(markers)],
                markersize=4, linewidth=1.6, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if ylog:
        ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] 曲线 -> {out_path}")
    return out_path
