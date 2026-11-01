"""RI 的概率验证：Brier 分数、可靠性图、ROC、校准误差。

为什么需要这个
--------------
现有 RI 只报 AUC / PR-AUC / CSI。AUC 只衡量**排序能力**（discrimination），
完全不衡量**概率是否可信**（calibration）。一个把所有人预测成 0.5 的模型
AUC 只有 0.5，但一个把 5% 正样本预测成 0.9 的模型 AUC 可以是 1.0——
后者在业务上完全不可用，AUC 却看不出来。Brier 分数同时惩罚这两类错误。

**必须先处理的坑：类别权重破坏了概率校准**
--------------------------------------------
train_ri.py 用加权 BCE 训练：

    w_pos = N/(2·n_pos)，w_neg = N/(2·n_neg)

加权 BCE 的最优解不是真实后验 p(x)，而是一个被先验偏移后的 q(x)：

    q = w_pos·p / (w_pos·p + w_neg·(1−p))

反过来可精确还原：

    p = w_neg·q / (w_pos·(1−q) + w_neg·q)

RI 正样本率约 4.8%（阈值 15 m/s）时 w_pos ≈ 10.4，模型输出的 q 被系统性**抬高**
（实测平均约 0.5 而不是 0.048）。直接拿 q 算 Brier 会严重高估误差，且可靠性图
会显示"过度自信"。本脚本同时报告 q 与还原后的 p。

**注意**：还原式只在模型完美最小化加权 BCE 时严格成立；实际是近似。
但它是一个**固定的解析变换**，不含任何在测试集上拟合的参数，所以不引入泄漏。

用法:
    python scripts/evaluation/prob_verify.py \\
        --npz outputs/ri/events_64px/cv10_thr15_summary.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

N_BINS = 10


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_skill(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Brier Skill Score = 1 − BS/BS_ref，参照为气候态（训练集正样本率）。

    气候态概率用 **验证集** 的正样本率估计，避免用测试集信息。
    """
    base = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    bs_ref = brier(y, np.full_like(p, base))
    bs = brier(y, p)
    return 1.0 - bs / bs_ref if bs_ref > 0 else float("nan"), base


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS):
    """可靠性表：等宽分箱后的 预测均值 / 观测频率 / 样本数。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows, edges


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error：按样本数加权的 |观测频率 − 预测均值|。"""
    rows, _ = reliability(y, p, n_bins)
    n = len(y)
    return float(sum(abs(o - f) * c / n for f, o, c in rows))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC（Mann-Whitney），与主脚本口径一致。"""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def unweight_prob(q: np.ndarray, pos_rate: float) -> np.ndarray:
    """把加权 BCE 训练出的 q 还原为真实后验 p。

    w_pos = 1/(2·pos_rate)，w_neg = 1/(2·(1−pos_rate))（相差常数因子不影响比值）。
    """
    w_pos = 1.0 / (2.0 * max(pos_rate, 1e-9))
    w_neg = 1.0 / (2.0 * max(1.0 - pos_rate, 1e-9))
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return w_neg * q / (w_pos * (1 - q) + w_neg * q)


def platt_fit(sv: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
    """在**验证集**上拟合 Platt 校准 p = sigmoid(a·logit(q) + b)。

    只用验证集，不用测试集 → 无泄漏。相比解析还原式，它不假设模型精确最小化
    加权 BCE，因此能吸收早停/正则带来的偏差，得到更可信的校准概率。
    """
    from scipy.optimize import minimize

    q = np.clip(sv, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))

    def nll(theta):
        a, b = theta
        t = a * z + b
        # 数值稳定的 log(1+exp(-t)) 形式 BCE
        return float(np.mean(np.logaddexp(0.0, -np.where(yv > 0.5, t, -t))))

    r = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    return float(r.x[0]), float(r.x[1])


def platt_apply(q: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def main() -> int:
    ap = argparse.ArgumentParser(description="RI 概率验证")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pos-rate", type=float, default=None,
                    help="训练集正样本率；默认从文件名/数据估计")
    ap.add_argument("--fig", default=None,
                    help="输出可靠性图 + ROC 的路径（png）")
    args = ap.parse_args()

    d = np.load(args.npz)
    # 键格式: f{fold}_{model}_{y_val|s_val|y_test|s_test}
    # 模型名本身可能含下划线（如 feature_transformer_云图），
    # 所以从**尾部**剥离两个字段，不能按位置切。
    SUFFIX = {"y_val", "s_val", "y_test", "s_test"}
    folds, models = set(), set()
    for k in d.files:
        parts = k.split("_")
        if len(parts) < 3 or "_".join(parts[-2:]) not in SUFFIX:
            continue
        folds.add(parts[0])
        models.add("_".join(parts[1:-2]))
    folds, models = sorted(folds), sorted(models)
    print(f"[输入] {Path(args.npz).name}")
    print(f"       {len(folds)} 折，模型 {models}")

    # 正样本率：用全部验证集的正样本率（验证集不参与测试评估）
    yv_all = np.concatenate([d[f"{f}_{models[0]}_y_val"] for f in folds])
    pos_rate = args.pos_rate if args.pos_rate is not None else float(yv_all.mean())
    print(f"       训练侧正样本率（用验证集估计）= {pos_rate:.4f}")

    print("\n" + "=" * 88)
    print("概率验证（测试集；q=原始 sigmoid，p=去权重还原）")
    print("=" * 88)
    print(f"{'模型':22s}{'AUC':>7s}{'Brier(q)':>10s}{'Brier(p)':>10s}{'Brier(pl)':>11s}"
          f"{'BSS(p)':>8s}{'BSS(pl)':>9s}{'ECE(q)':>8s}{'ECE(p)':>8s}{'ECE(pl)':>9s}")
    summary = {}
    skipped = []
    for m in models:
        # persistence 等基线返回的是**强度变化量**（m/s）而不是概率，
        # 对它算 Brier/ECE 会得到 2.09、0.84 这种无意义数值。
        # 判据：取值超出 [0,1] 即认为不是概率，只报 AUC（排序指标仍有效）。
        s_all = np.concatenate([d[f"{f}_{m}_s_test"] for f in folds])
        if s_all.min() < -1e-9 or s_all.max() > 1 + 1e-9:
            skipped.append((m, float(s_all.min()), float(s_all.max())))
            aucs = [auc(d[f"{f}_{m}_y_test"], d[f"{f}_{m}_s_test"]) for f in folds]
            print(f"{m:26s}{np.nanmean(aucs):>7.3f}"
                  f"{'—':>10s}{'—':>10s}{'—':>8s}{'—':>8s}{'—':>8s}"
                  f"{'—':>8s}{'—':>8s}   ← 非概率输入")
            continue
        aucs, bq, bp, bpl, bss, bss_pl = [], [], [], [], [], []
        eq, ep, epl, mq, mp = [], [], [], [], []
        for f in folds:
            y = d[f"{f}_{m}_y_test"]
            q = d[f"{f}_{m}_s_test"]
            # 正样本率用**该折的验证集**估计，避免测试集泄漏
            pr = float(d[f"{f}_{m}_y_val"].mean())
            p = unweight_prob(q, pr)
            # Platt 校准：在**验证集**上拟合，再作用于测试集
            a_, b_ = platt_fit(d[f"{f}_{m}_s_val"], d[f"{f}_{m}_y_val"])
            pl = platt_apply(q, a_, b_)
            aucs.append(auc(y, q))
            bq.append(brier(y, q))
            bp.append(brier(y, p))
            bpl.append(brier(y, pl))
            bss.append(brier_skill(y, p)[0])
            bss_pl.append(brier_skill(y, pl)[0])
            eq.append(ece(y, q))
            ep.append(ece(y, p))
            epl.append(ece(y, pl))
            mq.append(q.mean())
            mp.append(p.mean())
        summary[m] = dict(auc=float(np.nanmean(aucs)),
                          brier_q=float(np.nanmean(bq)),
                          brier_p=float(np.nanmean(bp)),
                          brier_platt=float(np.nanmean(bpl)),
                          bss=float(np.nanmean(bss)),
                          bss_platt=float(np.nanmean(bss_pl)),
                          ece_q=float(np.nanmean(eq)),
                          ece_p=float(np.nanmean(ep)),
                          ece_platt=float(np.nanmean(epl)))
        print(f"{m:22s}{np.nanmean(aucs):>7.3f}{np.nanmean(bq):>10.4f}"
              f"{np.nanmean(bp):>10.4f}{np.nanmean(bpl):>11.4f}"
              f"{np.nanmean(bss):>8.3f}{np.nanmean(bss_pl):>9.3f}"
              f"{np.nanmean(eq):>8.3f}{np.nanmean(ep):>8.3f}{np.nanmean(epl):>9.3f}")

    if skipped:
        print(f"\n[跳过概率指标] 以下模型输出不是概率（取值超出 [0,1]），"
              f"只报 AUC：")
        for m, lo, hi in skipped:
            print(f"  {m}: 范围 [{lo:.2f}, {hi:.2f}]")

    print("\n判读要点")
    print("  · AUC 高只说明**排序**好；Brier/BSS 才反映概率能不能直接当概率用。")
    print("  · ECE(q) 通常很大——这是加权训练抬高了输出，不是模型缺陷。")
    print("  · ECE(p)/BSS(p) 用**解析还原式**，它假设模型精确最小化加权 BCE；"
          "早停与正则会破坏该假设。")
    print("  · **以 pl 列为准**：Platt 只用验证集拟合，不依赖上述假设。"
          "实测 feature_transformer 的 BSS 在解析还原下为负、在 Platt 下转正。")
    print("  · BSS>0 表示优于气候态基准；BSS≤0 表示还不如直接报气候频率。")

    # 可靠性图数据（所有折合并，便于画图）
    print("\n" + "=" * 88)
    print("可靠性表（所有折合并；f=预测均值, o=观测频率, n=样本数）")
    print("=" * 88)
    skip_names = {x[0] for x in skipped}
    for m in models:
        if m in skip_names:
            continue
        y = np.concatenate([d[f"{f}_{m}_y_test"] for f in folds])
        q = np.concatenate([d[f"{f}_{m}_s_test"] for f in folds])
        pr = float(np.mean([d[f"{f}_{m}_y_val"].mean() for f in folds]))
        p = unweight_prob(q, pr)
        print(f"\n  [{m}]  实际正样本率 {y.mean():.4f}")
        for tag, v in (("q 原始", q), ("p 还原", p)):
            rows, _ = reliability(y, v)
            print(f"    {tag}: " + "  ".join(
                f"{f:.2f}→{o:.2f}(n={n})" for f, o, n in rows))
    if args.fig:
        _plot(d, folds, [m for m in models if m not in skip_names],
              unweight_prob, Path(args.fig))
    return 0


def _plot(d, folds, models, unweight_prob, out: Path):
    """三联图：可靠性图（q 与 p）、ROC、Brier 对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#805ad5"]
    if not models:
        print("[跳过绘图] 没有可作概率验证的模型")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=140)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
    for i, m in enumerate(models):
        y = np.concatenate([d[f"{f}_{m}_y_test"] for f in folds])
        q = np.concatenate([d[f"{f}_{m}_s_test"] for f in folds])
        # Platt 必须**逐折**拟合再拼接（校准参数依赖该折的验证集）
        p = np.concatenate([
            platt_apply(d[f"{f}_{m}_s_test"],
                        *platt_fit(d[f"{f}_{m}_s_val"], d[f"{f}_{m}_y_val"]))
            for f in folds])
        for v, ls, tag in ((q, ":", "原始 q"), (p, "-", "Platt 校准")):
            rows, _ = reliability(y, v)
            if not rows:
                continue
            ax.plot([r[0] for r in rows], [r[1] for r in rows], ls,
                    marker="o", ms=3, color=colors[i % len(colors)],
                    label=f"{m[:12]} {tag}")
    ax.set_xlabel("预测概率")
    ax.set_ylabel("观测频率")
    ax.set_title("可靠性图（越接近对角线越好）")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, m in enumerate(models):
        y = np.concatenate([d[f"{f}_{m}_y_test"] for f in folds])
        s_ = np.concatenate([d[f"{f}_{m}_s_test"] for f in folds])
        thr = np.unique(np.quantile(s_, np.linspace(0, 1, 200)))
        tpr, fpr = [], []
        for t in thr:
            prd = s_ >= t
            tpr.append(float((prd & (y == 1)).sum() / max(1, (y == 1).sum())))
            fpr.append(float((prd & (y == 0)).sum() / max(1, (y == 0).sum())))
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                label=f"{m[:14]} AUC={auc(y, s_):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("虚警率 FPR")
    ax.set_ylabel("命中率 POD")
    ax.set_title("ROC 曲线")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    x = np.arange(len(models))
    bq, bp = [], []
    for m in models:
        q = np.concatenate([d[f"{f}_{m}_s_test"] for f in folds])
        y = np.concatenate([d[f"{f}_{m}_y_test"] for f in folds])
        pl = np.concatenate([
            platt_apply(d[f"{f}_{m}_s_test"],
                        *platt_fit(d[f"{f}_{m}_s_val"], d[f"{f}_{m}_y_val"]))
            for f in folds])
        bq.append(brier(y, q))
        bp.append(brier(y, pl))
    ax.bar(x - 0.2, bq, 0.4, label="原始 q（加权训练输出）", color="#c53030")
    ax.bar(x + 0.2, bp, 0.4, label="Platt 校准", color="#2f855a")
    ax.set_xticks(x)
    ax.set_xticklabels([m[:12] for m in models], rotation=15)
    ax.set_ylabel("Brier 分数（越低越好）")
    ax.set_title("Brier：原始输出虚高，校准后接近基准")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
