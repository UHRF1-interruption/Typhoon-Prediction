"""为技术报告 §4.13 生成三张配图。

数据来源：`outputs/physics/laws/law_summary.json`（由 `law_summary.py` 生成，
其中每个数字都由 checkpoint 重新推理得到，无手工填入值）。

    fig_lawA_delta_by_seed.png   A 组：4 个种子的逐时效配对差
    fig_lawA_step_profile.png    A 组：逐时效位移剖面（模型 vs 实况）
    fig_lawB_diagnosis.png       B 组：引导流先验自身的误差 + 24h 位移欠账

用法:
    python scripts/evaluation/law_summary.py
    python scripts/evaluation/plot_physics_law_figures.py
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

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "outputs" / "figures"
LEADS = [6, 12, 18, 24]
C_BASE, C_LAW, C_TRUE = "#7a7a7a", "#2b6cb0", "#1a1a1a"


def fig_delta_by_seed(d, path):
    seeds = sorted(d["A"]["per_seed"], key=int)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for i, s in enumerate(seeds):
        y = [d["A"]["per_seed"][s][f"{h}h"]["delta"] for h in LEADS]
        ax.plot(LEADS, y, "o-", lw=1.2, ms=4, alpha=0.55,
                label=f"seed {s}", color=plt.cm.tab10(i % 10))
    mean = [d["A"]["cross_seed"][f"{h}h"]["mean_delta"] for h in LEADS]
    sd = [d["A"]["cross_seed"][f"{h}h"]["sd_delta"] for h in LEADS]
    ax.errorbar(LEADS, mean, yerr=sd, fmt="s-", color="k", lw=2.4, ms=7,
                capsize=4, label="均值 ± 种子 SD", zorder=5)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(LEADS)
    ax.set_xlabel("预报时效 (h)")
    ax.set_ylabel("ΔMAE（w=0.1 − w=0，km）")
    ax.set_title("A 组：负值=更优；4 个种子中 12/18/24h 均为 4/4 更优")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    for h, x in zip(LEADS, LEADS):
        p = d["A"]["cross_seed"][f"{h}h"]["p"]
        ax.annotate(f"p={p:.2f}", (x, ax.get_ylim()[1]), xytext=(0, -10),
                    textcoords="offset points", ha="center", fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_step_profile(d, path):
    runs = d["A"]["runs"]
    seeds = sorted({k.split("_")[0] for k in runs})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for w, color, label in [(0.0, C_BASE, "w=0"), (0.1, C_LAW, "w=0.1")]:
        arr = np.array([runs[f"{s}_w{w}"]["step_km"] for s in seeds])
        ax.plot(LEADS, arr.mean(0), "o-", color=color, lw=2, label=label)
        ax.fill_between(LEADS, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                        color=color, alpha=0.15)
    ax.plot(LEADS, d["A"]["truth"]["step_km"], "k--", lw=1.6, label="实况")
    ax.set_xticks(LEADS)
    ax.set_xlabel("预报时效 (h)")
    ax.set_ylabel("该时效内的位移 (km)")
    ax.set_title("A 组：w=0.1 抑制了模型末段的减速（阴影=种子 SD）")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_b_diagnosis(d, path):
    b = d["B"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), dpi=140)

    ax = axes[0]
    ax.plot(LEADS, b["runs"]["w0.0"]["mae_by_step"], "o-", color=C_BASE, lw=2, label="模型 w=0")
    ax.plot(LEADS, b["truth"]["mae_by_step"], "s--", color="#dd6b20", lw=2,
            label="引导流先验 $v=k\\,u_{500}$")
    ax.set_xticks(LEADS)
    ax.set_xlabel("预报时效 (h)")
    ax.set_ylabel("MAE (km)")
    ax.set_title(f"引导流先验自身误差（k={b['k']:.3f}, $R^2$={b['k_r2']:.3f}）")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    vals = [b["runs"]["w0.0"]["decomp24"]["pred_disp_km"],
            b["runs"]["w0.3"]["decomp24"]["pred_disp_km"],
            b["truth"]["cum_step_km"][-1]]
    labels = ["w=0", "w=0.3", "实况"]
    colors = [C_BASE, C_LAW, C_TRUE]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v:.0f}",
                ha="center", fontsize=9)
    ax.set_ylabel("24h 累计位移 (km)")
    ax.set_title("B 组：误差的主体是“走得太慢”")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.03, 0.80,
            f"w=0 的 24h 误差：沿轨 {b['runs']['w0.0']['decomp24']['along']:+.0f} km、"
            f"横轨 {b['runs']['w0.0']['decomp24']['cross_abs']:.0f} km",
            transform=ax.transAxes, fontsize=8, color="#444")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 §4.13 配图")
    ap.add_argument("--json", default="outputs/physics/laws/law_summary.json")
    args = ap.parse_args()
    d = json.loads((ROOT / args.json).read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    fig_delta_by_seed(d, OUT / "fig_lawA_delta_by_seed.png")
    fig_step_profile(d, OUT / "fig_lawA_step_profile.png")
    if d.get("B", {}).get("runs"):
        fig_b_diagnosis(d, OUT / "fig_lawB_diagnosis.png")
    for p in ["fig_lawA_delta_by_seed.png", "fig_lawA_step_profile.png", "fig_lawB_diagnosis.png"]:
        f = OUT / p
        print(f"[saved] {f} {'✓' if f.exists() else '(skipped)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
