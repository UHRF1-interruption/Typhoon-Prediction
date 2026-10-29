"""命令行入口: 业务预报误差基线分析（EC / 中央气象台 / AI 模式 / 持续法）。

数据来自工程附带的 CMA 年度预报误差表（2024/2025 年西北太平洋），
无需下载即可运行：

  python scripts/baseline_report.py

产出:
  outputs/baselines/
    ├── baseline_table.md          各方法 24/48/72/120h 路径与强度误差
    ├── tra_error_vs_lead.png      路径误差-时效曲线（重点方法）
    ├── int_error_vs_lead.png      强度误差-时效曲线
    ├── method_ranking.png         24h / 72h 路径误差排名
    └── metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _data_path(cfg, value):
    """把配置路径解析成绝对路径，支持 TYPHOON_DATA_ROOT 挂载外部数据。"""
    from src.config import resolve_path
    return resolve_path(cfg, value, ROOT)

sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.forecast_errors import (  # noqa: E402
    METHOD_LABELS, clean_outliers, error_by_horizon, load_forecast_errors, summarize_baselines,
)
from src.evaluation.plots import plot_error_curves  # noqa: E402

# 报告里重点展示的方法：最强数值模式 + 最强 AI 模式 + 中国业务 + 对照
FOCUS = ["ecmf", "aifs", "babj", "japn", "avno", "shtm", "pangu", "fengwu", "fuxi", "fourcastnet"]


def to_markdown(df, floatfmt: str = "{:.1f}") -> str:
    """把 DataFrame 渲染成 markdown 表格（不依赖 tabulate）。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="业务预报误差基线分析")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    data_dir = _data_path(cfg, cfg.data.raw_dir) / "cma_attachments"
    out_dir = ROOT / cfg.output.dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    tra, int_ = load_forecast_errors(data_dir)
    tra_c = clean_outliers(tra)

    horizons = [24, 48, 72, 120]
    summ = summarize_baselines(tra_c, int_, horizons)

    # 只保留报告列
    table = summ[["method", "label", "n"] +
                 [c for c in summ.columns if c.startswith(("tra_", "int_"))]].copy()
    table.columns = ["方法", "说明", "样本数"] + [
        c.replace("tra_", "路径 ").replace("int_", "强度 ") for c in table.columns[3:]
    ]
    md = to_markdown(table)

    # 分组平均值
    tra_c["group"] = tra_c["method"].map(lambda m: METHOD_LABELS.get(m, m))
    lines = [
        "# 台风预报误差基线（2024–2025 西北太平洋）",
        "",
        f"- 台风样本: {tra_c['tc_id'].nunique()} 个"
        f"（{', '.join(str(int(y)) for y in sorted(tra_c['year'].dropna().unique()))} 年）",
        f"- 预报方法: {tra_c['method'].nunique()} 个，含数值模式与 AI 模式",
        f"- 路径误差记录: {len(tra_c)} 条（已剔除 >3000km 的异常记录 {len(tra) - len(tra_c)} 条）",
        "",
        "## 各方法分时效误差",
        "",
        md,
        "",
        "## 与自研模型的关系",
        "",
        "自研 Transformer 模型必须同时满足两个门槛才算有效：",
        "1. 显著优于 `persistence`（持续法）与 `linear` 线性基线；",
        "2. 在相近时效上接近或优于上表的业务模式（24h 约 55 km 是当前上限水平）。",
    ]
    (out_dir / "baseline_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md)

    # 曲线
    focus = [m for m in FOCUS if m in set(tra_c["method"])]
    curves = {METHOD_LABELS.get(m, m): tra_c[tra_c["method"] == m]
              for m in focus}
    plot_error_curves(curves, "tra_err", (0, 120), "Forecast lead time (h)",
                      "Mean track error (km)", "Track error vs lead time (2024-2025 WP)",
                      out_dir / "tra_error_vs_lead.png")
    int_curves = {METHOD_LABELS.get(m, m): int_[int_["method"] == m] for m in focus if m in set(int_["method"])}
    if int_curves:
        plot_error_curves(int_curves, "int_err_abs", (0, 120), "Forecast lead time (h)",
                          "Mean intensity error (m/s)", "Intensity error vs lead time (2024-2025 WP)",
                          out_dir / "int_error_vs_lead.png")
    plot_method_ranking(tra_c, [24, 72], out_dir / "method_ranking.png")

    metrics = {
        m["method"]: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in m.items() if k.startswith(("tra_", "int_"))}
        for m in summ.to_dict(orient="records")
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出目录: {out_dir}")
    return 0


def plot_method_ranking(tra: pd.DataFrame, leads: list[int], out_path: Path) -> Path:
    """各方法在指定时效的误差排名条形图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(leads)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.0), dpi=150, squeeze=False)
    for ax, lead in zip(axes[0], leads):
        sub = tra[tra["lead"] == lead].groupby("method")["tra_err"].mean().sort_values()
        labels = [METHOD_LABELS.get(m, m) for m in sub.index]
        colors = ["#d62728" if m in {"pangu", "aifs", "fengwu", "fengqing", "fuxi",
                                     "graphcast", "fourcastnet", "jawt"} else "#1f77b4"
                  for m in sub.index]
        ax.barh(range(len(sub)), sub.values, color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Mean track error (km)")
        ax.set_title(f"{lead}h track error")
        ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(sub.values):
            ax.text(v, i, f" {v:.0f}", va="center", fontsize=6)
    fig.suptitle("Operational & AI model ranking (blue: NWP, red: AI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] 排名图 -> {out_path}")
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
