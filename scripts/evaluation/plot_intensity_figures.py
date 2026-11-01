"""命令行入口: 生成技术报告 §4 的配图。

只用**已保存的结果文件**重画，不重新训练，因此可以随时复现：

    .venv/bin/python scripts/evaluation/plot_intensity_figures.py

产出（outputs/figures/）：
    intensity_mae_vs_horizon.png   强度误差随时效（对应报告 §4.1.1 表格）
    cloud_ablation_rounds.png      云图四轮消融（对应 §4.1.2 表格，核心结论）
    ri_auc_compare.png             RI 事件 AUC / PR-AUC（对应 §4.2 表格）
    feature_ablation_delta.png     地形/暖心特征消融（对应 §4.3/§4.4）

所有数字都从结果文件读取，脚本内不写任何硬编码的指标值——图表与表格
必然一致。若某个结果文件缺失，对应子图跳过并提示，不会画空图。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.plots import plt as _styled  # noqa: F401,E402  复用中文字体配置

HORIZON = [3, 6, 9, 12]           # 强度任务时效（小时）

# 结果文件 -> 图例名。顺序即绘图顺序。
MODEL_LABELS = [
    ("gru", "GRU only", "C0"),
    ("fusion_transformer+云图", "CNN + satellite", "C3"),
    ("feature_transformer+云图", "Feature + satellite", "C2"),
    ("persistence", "Persistence", "C7"),
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"[skip] 缺少 {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _model_tags(pf: dict) -> list[str]:
    """从 per_fold 的一个 job 里挑出模型名。

    per_fold 的条目混有模型结果与元信息（fold_seed/fold/n_storms），
    后两者中 n_storms 也是 dict，不能只按类型判断——必须看是否含
    mae_by_step 字段，否则会把它当成模型。
    """
    first = next(iter(pf.values()))
    return [t for t, v in first.items()
            if isinstance(v, dict) and "mae_by_step" in v and "overall" in v]


def _per_fold(metric_path: Path) -> dict[str, np.ndarray]:
    """{模型: (n_job,) 各 job 的平均 MAE}。"""
    d = _load(metric_path)
    if d is None:
        return {}
    pf = d["per_fold"]
    return {t: np.array([pf[j][t]["overall"] for j in pf], dtype=float)
            for t in _model_tags(pf)}


def _per_fold_step(metric_path: Path) -> dict[str, np.ndarray]:
    """{模型: (n_job, n_step) 逐时效 MAE}。"""
    d = _load(metric_path)
    if d is None:
        return {}
    pf = d["per_fold"]
    return {t: np.array([pf[j][t]["mae_by_step"] for j in pf], dtype=float)
            for t in _model_tags(pf)}


def fig_intensity_horizon(history: Path, out: Path) -> None:
    """强度误差随时效，带 15 个 job 的标准差误差棒。"""
    data = _per_fold_step(history)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    markers = ["o", "s", "^", "D"]
    for i, (tag, label, colour) in enumerate(MODEL_LABELS):
        if tag not in data:
            continue
        m = data[tag].mean(axis=0)
        s = data[tag].std(axis=0)
        ax.errorbar(HORIZON, m, yerr=s, marker=markers[i % 4], color=colour,
                    linewidth=1.8, capsize=3, label=f"{label} (avg {m.mean():.2f} m/s)")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Mean intensity error (m/s)")
    ax.set_title("Intensity error vs. lead time (test set, 15 jobs = 5 folds × 3 splits)")
    ax.set_xticks(HORIZON)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] 强度误差曲线 -> {out}")


def fig_cloud_rounds(rounds: list[tuple[str, Path]], backup: str, out: Path) -> None:
    """云图四轮消融：ΔMAE（对照 GRU）+ 胜率。

    每轮从自己的结果文件算 ΔMAE = fusion 的 MAE − gru 的 MAE（逐 job 平均），
    胜率 = ΔMAE < 0 的 job 数 / 总 job 数。不读取任何手填数值。
    """
    labels, deltas, errs, wins, totals = [], [], [], [], []
    for label, path in rounds:
        pf = _per_fold(path)
        if not pf or backup not in pf or "fusion_transformer+云图" not in pf:
            print(f"[skip] {label}: 文件缺少 fusion 或 {backup}")
            continue
        d = pf["fusion_transformer+云图"] - pf[backup]
        labels.append(label)
        deltas.append(float(d.mean()))
        errs.append(float(d.std(ddof=1)) if len(d) > 1 else 0.0)
        wins.append(int((d < 0).sum()))
        totals.append(int(len(d)))
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=150)
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, yerr=errs, capsize=4, color="C3", alpha=0.85,
                  edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=1.0)
    # 标注放在"柱顶 + 误差棒"之上，否则会与 errorbar 叠在一起
    top = max(h + e for h, e in zip(deltas, errs))
    for b, w, n, h, e in zip(bars, wins, totals, deltas, errs):
        ax.text(b.get_x() + b.get_width() / 2, h + e + 0.006 * top,
                f"win {w}/{n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ΔMAE vs GRU (m/s)   ← better | worse →")
    ax.set_title("Satellite imagery ablation: all four rounds are worse than GRU\n"
                 "(bar = mean over jobs, whisker = std; positive = imagery hurts)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] 云图四轮消融 -> {out}")


def fig_ri_auc(ri_dir: Path, out: Path) -> None:
    """RI 事件：两个阈值下各模型的 AUC，带 10 折标准差。"""
    series = {}
    for thr, fname in ((10, "cv10_thr10_summary.json"), (15, "cv10_thr15_summary.json")):
        d = _load(ri_dir / fname)
        if d is None:
            continue
        series[thr] = d["summary"]
    if not series:
        return

    models = list(next(iter(series.values())).keys())
    # 图内标签用英文，与报告其他图保持一致（中文名见正文表格）
    label_map = {"CNN+云图": "CNN+sat", "特征+云图": "Feat+sat",
                 "纯时序GRU": "GRU", "持续法": "Persistence"}
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=150)
    w = 0.8 / max(1, len(series))
    x = np.arange(len(models))
    for i, (thr, summ) in enumerate(series.items()):
        vals = [summ[m]["AUC_mean"] for m in models]
        errs = [summ[m]["AUC_std"] for m in models]
        pos = x + i * w - 0.4 + w / 2
        ax.bar(pos, vals, width=w, yerr=errs, capsize=3,
               label=f"RI threshold {int(thr)} m/s")
        # 数值标注放在"柱顶 + 误差棒"之上，避免与 errorbar 的横帽重叠
        for xp, v, e in zip(pos, vals, errs):
            ax.text(xp, v + e + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(m, m) for m in models], fontsize=9)
    # AUC 的无技术基线是 0.5；从 0.5 起画，避免截断 y 轴夸大组间差异
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUC (10-fold CV)")
    if len(series) == 1:
        thr_only = next(iter(series))
        ax.set_title(f"Rapid-intensification detection: AUC at threshold {int(thr_only)} m/s")
    else:
        ax.set_title("Rapid-intensification event detection: AUC by model and threshold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] RI AUC 对比 -> {out}")


def fig_feature_ablation(variants: list[tuple[str, Path]], out: Path,
                         base_path: Path) -> None:
    """地形 / 暖心特征消融：相对 base 的 ΔMAE（逐 job 平均）。

    base_path 必须**显式传入**：早期版本从变体路径反推
    （`变体目录/../intensity_rerun_base`），当变体改名为 224 台风批量时
    会反推到 90 台风的 base，导致跨数据集比较而不会报错。显式传入后
    由调用方负责保证 base 与变体同源。
    """
    base = _per_fold(base_path)
    if not base:
        print(f"[skip] base 无数据：{base_path}")
        return

    # 排除 persistence：特征只进标量输入，持续法不看特征，ΔMAE 恒为 0，
    # 画出来是一根空柱，只会让人误以为缺数据
    models = [t for t, _, _ in MODEL_LABELS if t in base and t != "persistence"]
    rows = []
    for label, path in variants:
        cur = _per_fold(path)
        if not cur:
            continue
        rows.append((label, cur))
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=150)
    w = 0.8 / max(1, len(models))
    x = np.arange(len(rows))
    for i, m in enumerate(models):
        vals, errs = [], []
        for _, cur in rows:
            if m not in cur:
                vals.append(0.0)
                errs.append(0.0)
                continue
            d = cur[m] - base[m]
            vals.append(float(d.mean()))
            errs.append(float(d.std(ddof=1)) if len(d) > 1 else 0.0)
        label = dict((t, lbl) for t, lbl, _ in MODEL_LABELS).get(m, m)
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, yerr=errs, capsize=3, label=label)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("ΔMAE vs base (m/s)   ← better | worse →")
    ax.set_title("Feature ablation on 224 storms: warm-core hurts, terrain is neutral\n"
                 "(bar = mean ΔMAE over 15 jobs, whisker = std; positive = worse)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] 特征消融 -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成技术报告 §4 的配图")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "figures"))
    ap.add_argument("--history", default=str(ROOT / "outputs" / "intensity" / "sat_intensity_history"))
    ap.add_argument("--ri-dir", default=str(ROOT / "outputs" / "ri" / "events_64px"))
    ap.add_argument("--ablation", default=str(ROOT / "outputs" / "intensity" / "rerun_224"),
                    help="特征消融结果所在目录（读 intensity_rerun_224_*）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = Path(args.history)

    # 图 A：当前默认配置（64px + z-score + 23 维特征）的逐时效误差
    fig_intensity_horizon(hist / "cv5_summary.json", out_dir / "intensity_mae_vs_horizon.png")

    # 图 B：§4.1.2 四轮消融
    fig_cloud_rounds(
        [
            ("#1 90 storms, 32px", hist / ".." / "rerun" / "intensity_rerun_base" / "cv5_summary.json"),
            ("#2 224 storms, 32px", hist / "cv5_32px-255norm-15feat_summary.json"),
            ("#3 + dropout 0.25", hist / "cv5_satellite-model-dropout-0-25_65c72974_summary.json"),
            ("#4 64px + zscore + 23 feat", hist / "cv5_summary.json"),
        ],
        backup="gru",
        out=out_dir / "cloud_ablation_rounds.png",
    )

    # 图 C：RI
    fig_ri_auc(Path(args.ri_dir), out_dir / "ri_auc_compare.png")

    # 图 D：地形 / 暖心消融
    # 224 台风版本（2026-10-30 重跑）。base 必须与变体同源，否则会变成
    # 跨数据集的错误比较。
    abl = Path(args.ablation) if getattr(args, "ablation", None) else \
        ROOT / "outputs" / "intensity" / "rerun_224"
    fig_feature_ablation(
        [
            ("+ warm-core", abl / "intensity_rerun_224_warmcore" / "cv5_summary.json"),
            ("+ terrain", abl / "intensity_rerun_224_terrain" / "cv5_summary.json"),
        ],
        out=out_dir / "feature_ablation_delta.png",
        base_path=abl / "intensity_rerun_224_base" / "cv5_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
