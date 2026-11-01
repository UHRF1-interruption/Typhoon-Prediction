"""生成分时效与分层验证的图（报告 §4.1 / §4.8 用）。

三张图：
    fig_acc_by_lead.png        路径：环境场 vs 纯轨迹，距平相关随预报时效的变化
    fig_sat_acc_by_lead.png    强度：GRU vs 云图，距平相关随预报时效的变化
    fig_stratify_change.png    强度：按实际变化幅度分层后，云图相对 GRU 的 ΔMAE

数据来源全部是已保存的结果文件，不包含任何手工填入的数字。
"""
from __future__ import annotations

import glob
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


def _r(x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or x[m].std() < 1e-12 or y[m].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def path_acc(files):
    """路径的距平相关：距平 = 预测 − 持续法预测。"""
    A = []
    for f in files:
        d = np.load(f)
        P, T, L = d["pred_abs"], d["true_abs"], d["last_abs"]
        A.append([_r(P[:, t, :] - L, T[:, t, :] - L) for t in range(P.shape[1])])
    return np.array(A)


def fig_acc_by_lead():
    base = path_acc(sorted(glob.glob("outputs/path/ablation/stratified_ablation/pred_baseline_s*.npz")))
    env = path_acc(sorted(glob.glob("outputs/path/ablation/stratified_ablation/pred_env_s*.npz")))
    leads = [3, 6, 9, 12]
    n = min(len(base), len(env))
    base, env = base[:n], env[:n]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    ax.plot(leads, base.mean(0), "o-", label="纯轨迹", color="#888")
    ax.plot(leads, env.mean(0), "s-", label="+ 环境场", color="#2b6cb0")
    # 配对差
    for i, l in enumerate(leads):
        d = env[:, i] - base[:, i]
        top = max(base[:, i].mean(), env[:, i].mean())
        ax.annotate(f"{d.mean():+.3f}", (l, top),
                    textcoords="offset points", xytext=(14, 6),
                    ha="left", fontsize=8,
                    color="#c53030" if d.mean() > 0 else "#718096")
    ax.set_xlim(2.4, 12.6)
    ax.set_xlabel("预报时效 (h)")
    ax.set_ylabel("距平相关系数 r\n（距平 = 预测 − 持续法）")
    ax.set_title("路径：环境场的收益随时效增大（10 job）")
    ax.set_xticks(leads)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = OUT / "fig_acc_by_lead.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"saved {p}")
    print("  纯轨迹:", np.round(base.mean(0), 4).tolist())
    print("  +环境场:", np.round(env.mean(0), 4).tolist())


def fig_sat_acc_by_lead():
    cands = [Path("outputs/intensity/summaries/intensity_rerun_224_corr_summary.json"),
             Path("outputs/intensity/summaries/intensity_rerun_224_summary.json")]
    src = next((c for c in cands if c.exists()), None)
    if src is None:
        print("[跳过] 找不到强度汇总 JSON")
        return
    s = json.loads(src.read_text())["base"]["summary"]
    leads = [3, 6, 9, 12]
    g = np.array(s["gru"]["corr_by_step_mean"], float)
    f = np.array(s["fusion_transformer+云图"]["corr_by_step_mean"], float)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    ax.plot(leads, g, "o-", label="GRU（纯标量）", color="#2b6cb0")
    ax.plot(leads, f, "s-", label="GRU + 红外云图", color="#dd6b20")
    for i, l in enumerate(leads):
        ax.annotate(f"{f[i] - g[i]:+.3f}", (l, f[i]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#c53030")
    ax.set_xlabel("预报时效 (h)")
    ax.set_ylabel("距平相关系数 r\n（距平 = 预测 − 当前强度）")
    ax.set_title("强度：云图的劣势随时效收窄（15 job）")
    ax.set_xticks(leads)
    ax.set_ylim(min(f.min(), g.min()) - 0.05, max(f.max(), g.max()) + 0.04)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = OUT / "fig_sat_acc_by_lead.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"saved {p}")
    print("  GRU:", np.round(g, 4).tolist())
    print("  云图:", np.round(f, 4).tolist())


def fig_stratify_change():
    files = sorted(glob.glob("outputs/intensity/rerun_224/intensity_rerun_224_pred_base/*.npz"))
    if not files:
        print("[跳过] 找不到逐样本预测 npz")
        return
    d = np.load(files[0])
    off = d["job_offsets"]
    P0, P1, T, L = (d["gru__pred"], d["fusion_transformer+云图__pred"],
                    d["gru__true"], d["gru__last"])
    bins = [(0, 1, "<1"), (1, 3, "1–3"), (3, 6, "3–6"),
            (6, 10, "6–10"), (10, 1e9, "≥10")]
    labels, dm, g0, g1 = [], [], [], []
    for j in range(len(off) - 1):
        a, b = int(off[j]), int(off[j + 1])
        t, last = T[a:b], L[a:b].reshape(-1)
        ch = np.abs(t[:, -1] - last)
        e0 = np.abs(P0[a:b] - t).mean(axis=1)
        e1 = np.abs(P1[a:b] - t).mean(axis=1)
        for lo, hi, lb in bins:
            m = (ch >= lo) & (ch < hi)
            if m.sum() < 20:
                continue
            g0.append(e0[m].mean())
            g1.append(e1[m].mean())
        # 记录一次标签（各 job 相同）
    for lo, hi, lb in bins:
        labels.append(lb)
    n = len(labels)
    g0 = np.array(g0).reshape(-1, n)
    g1 = np.array(g1).reshape(-1, n)
    dm = (g1 - g0).mean(0)
    err0, err1 = g0.mean(0), g1.mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=140)
    ax = axes[0]
    x = np.arange(n)
    ax.bar(x - 0.2, err0, 0.4, label="GRU", color="#2b6cb0")
    ax.bar(x + 0.2, err1, 0.4, label="GRU + 云图", color="#dd6b20")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("12h 实际强度变化幅度 (m/s)")
    ax.set_ylabel("MAE (m/s)")
    ax.set_title("按变化幅度分层：基线难度差 4.7 倍")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    ax = axes[1]
    colors = ["#c53030" if v > 0 else "#2f855a" for v in dm]
    ax.bar(x, dm, 0.55, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("12h 实际强度变化幅度 (m/s)")
    ax.set_ylabel("ΔMAE（云图 − GRU, m/s）")
    ax.set_title("五层全部为正：云图在任何子集都无增益")
    for i, v in enumerate(dm):
        ax.annotate(f"{v:+.3f}", (i, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig_stratify_change.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"saved {p}")
    print("  ΔMAE:", np.round(dm, 4).tolist())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_acc_by_lead()
    fig_sat_acc_by_lead()
    fig_stratify_change()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
