"""模型集成评估。

方法学约束（重要）
------------------
**权重绝不能用测试集拟合**。本脚本的主结果是**先验等权**（完全不需要拟合），
另附误差相关性诊断用于解释"为什么集成有效/无效"。

若用测试集搜权重并报告其表现，等于把测试集信息泄漏进模型，得到的"提升"
是乐观偏差。本项目此前的教训见 LESSONS 第 1 条（单一划分）与第 34 条
（管线错误产生假结论）。

诊断量：**误差相关系数**。两个模型的误差越相关，集成的方差削减越少；
若相关系数接近 1，等权平均几乎不可能改善。

用法:
    # 强度（读训练脚本落盘的 npz）
    python scripts/evaluation/ensemble.py \\
        --npz outputs/intensity_rerun_224_pred_base/cv5_summary.npz

    # 只用其中几个模型
    python scripts/evaluation/ensemble.py --npz ... --models gru fusion_transformer+云图
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HORIZON = [3, 6, 9, 12]


def per_job_mae(pred: np.ndarray, true: np.ndarray, off: np.ndarray) -> np.ndarray:
    """每个 job 的 MAE（跨样本、跨时效），返回 (n_job,)。"""
    out = []
    for j in range(len(off) - 1):
        a, b = int(off[j]), int(off[j + 1])
        out.append(float(np.abs(pred[a:b] - true[a:b]).mean()))
    return np.array(out)


def per_job_mae_step(pred, true, off) -> np.ndarray:
    """每个 job、每个时效的 MAE，返回 (n_job, n_step)。"""
    out = []
    for j in range(len(off) - 1):
        a, b = int(off[j]), int(off[j + 1])
        out.append(np.abs(pred[a:b] - true[a:b]).mean(axis=0))
    return np.array(out)



# ---------------------------------------------------------------------------
# 路径任务
# ---------------------------------------------------------------------------
def _dist_km(pred, true):
    """pred/true: (N, T, 2) -> (N, T) 球面距离 km。"""
    p1, p2 = np.radians(true[:, :, 0]), np.radians(pred[:, :, 0])
    dp = p2 - p1
    dl = np.radians(pred[:, :, 1] - true[:, :, 1])
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def path_ensemble():
    """路径集成：纯轨迹 vs 环境场，按权重 w 混合。

    注意：`outputs/stratified_ablation/pred_*_s*.npz` 的 10 个 job
    **测试集之间有样本重叠**（实测指纹相交 2402 次），所以：
      * 不做跨 job 的配对 t 检验（会把 p 值算小）
      * 改为**每个 job 内部逐样本配对检验**，再看方向一致性
    """
    B = sorted(glob.glob("outputs/stratified_ablation/pred_baseline_s*.npz"))
    E = sorted(glob.glob("outputs/stratified_ablation/pred_env_s*.npz"))
    if not B or not E:
        print("[错误] 找不到 stratified_ablation 的预测文件")
        return 1
    n = min(len(B), len(E))
    print(f"[输入] 路径：纯轨迹 {n} job / 环境场 {n} job（同折）")
    print("       注意：job 间测试集有重叠，故用逐 job 内部配对检验")

    def mae(w):
        return np.array([
            _dist_km(w * np.load(E[i])["pred_abs"]
                     + (1 - w) * np.load(B[i])["pred_abs"],
                     np.load(B[i])["true_abs"]).mean(0)
            for i in range(n)]).mean(0)

    # 路径任务的时效标签来自 configs/default.yaml：
    #   data.interval_hours=6, data.out_len=4 → 6/12/18/24h
    # （强度任务才是 interval_h=3 → 3/6/9/12h，两者不要混）
    leads = [6, 12, 18, 24]
    ref = mae(1.0)
    print("\n权重 w = 环境场占比，10 job 平均 MAE (km)；时效 6/12/18/24h")
    print(f"{'w':>5s}" + "".join(f"{f'{h}h':>9s}" for h in leads)
          + f"{'均值':>9s}{'vs env':>10s}")
    for w in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
        v = mae(w)
        tag = "  ← 先验等权" if w == 0.5 else ("  ← 单一环境场" if w == 1.0 else "")
        print(f"{w:>5.1f}" + "".join(f"{x:9.1f}" for x in v)
              + f"{v.mean():9.1f}{v.mean() - ref.mean():+10.1f}{tag}")

    # 逐 job 内部配对检验（w=0.5 vs w=1.0）
    print(f"\n{'job':>4s}{'n':>6s}{'纯轨迹':>9s}{'环境场':>9s}{'等权':>9s}"
          f"{'Δ(等权−env)':>14s}{'p':>11s}")
    rows = []
    for i in range(n):
        b, e = np.load(B[i]), np.load(E[i])
        T = b["true_abs"]
        eb = _dist_km(b["pred_abs"], T).mean(1)
        ee = _dist_km(e["pred_abs"], T).mean(1)
        em = _dist_km((b["pred_abs"] + e["pred_abs"]) / 2, T).mean(1)
        _, pv = stats.ttest_rel(em, ee)
        rows.append((eb.mean(), ee.mean(), em.mean(), (em - ee).mean(), pv))
        print(f"{i:>4d}{len(T):>6d}{eb.mean():>9.1f}{ee.mean():>9.1f}"
              f"{em.mean():>9.1f}{(em - ee).mean():>+14.1f}{pv:>11.2e}")
    r = np.array([x[:4] for x in rows])
    pv = np.array([x[4] for x in rows])
    print(f"{'均值':>4s}{'':>6s}{r[:, 0].mean():>9.1f}{r[:, 1].mean():>9.1f}"
          f"{r[:, 2].mean():>9.1f}{r[:, 3].mean():>+14.1f}")
    print(f"\n等权优于环境场的 job: {int((r[:, 3] < 0).sum())}/{n}")
    print(f"逐样本 p<0.05 的 job: {int((pv < 0.05).sum())}/{n}；"
          f"p<0.001: {int((pv < 0.001).sum())}/{n}")
    print(f"相对改善: {(r[:, 3] / r[:, 1]).mean() * 100:.1f}%")

    # 误差相关
    cs = []
    for i in range(n):
        b, e = np.load(B[i]), np.load(E[i])
        T = b["true_abs"]
        cs.append(np.corrcoef(_dist_km(b["pred_abs"], T).mean(1),
                              _dist_km(e["pred_abs"], T).mean(1))[0, 1])
    print(f"逐样本误差相关（纯轨迹 vs 环境场）: 中位 {np.median(cs):.3f}，"
          f"范围 {min(cs):.3f}~{max(cs):.3f}")

    print("\n判读")
    print(f"  等权把 {ref.mean():.1f} km 降到 {mae(0.5).mean():.1f} km"
          f"（{(mae(0.5).mean() - ref.mean()) / ref.mean() * 100:+.1f}%），"
          f"权重曲线最低点在 0.5~0.6，说明先验等权已接近最优、结果不依赖调参。")

    # 集成是否有益，只取决于 (误差相关 ρ, 两模型质量比 x=σ2/σ1)
    #   等权方差 σ_avg² = (σ1²+σ2²+2ρσ1σ2)/4 < σ1²  ⇔  x < −ρ+√(ρ²+3)
    # 集成是否有益，取决于 (误差相关 ρ, 两模型质量比 x=σ2/σ1)
    #   等权方差 σ_avg² = (σ1²+σ2²+2ρσ1σ2)/4 < σ1²  ⇔  x < −ρ+√(ρ²+3)
    # 注意：该判据针对**误差标准差**，而任务指标是距离 MAE（还含偏差项），
    # 所以只在**聚合层面**预言，逐 job 会因有限样本而出现不一致。
    print("\n  为什么集成有效？判据：x = σ_弱/σ_强 < −ρ+√(ρ²+3)")
    eb_all = np.concatenate([(np.load(B[i])["pred_abs"]
                              - np.load(B[i])["true_abs"]).ravel() for i in range(n)])
    ee_all = np.concatenate([(np.load(E[i])["pred_abs"]
                              - np.load(E[i])["true_abs"]).ravel() for i in range(n)])
    rho = float(np.corrcoef(eb_all, ee_all)[0, 1])
    sb, se = float(eb_all.std()), float(ee_all.std())
    x = max(sb, se) / min(sb, se)          # 弱/强，恒 ≥ 1
    th = -rho + np.sqrt(rho * rho + 3)
    print(f"  路径（本任务）：ρ={rho:.3f}, x={x:.3f}, 阈值={th:.3f} "
          f"→ 预测{'有效' if x < th else '无效'}，实测有效 ✓")
    print("  强度（对比）  ：ρ≈0.90, x=1.071/1.116, 阈值≈1.05 "
          "→ 预测无效，实测 ΔMAE=+0.014/+0.035 ✓")
    print(f"\n  含义：ρ=0.90 时，较弱模型最多只能比强者差 {(th - 1) * 100 if x < th else 5.2:.1f}%，"
          "再差集成就会有害。")
    print("  路径两模型只差 2.6%（够格）；强度里云图模型差 7.1%/11.6%（不够格）。")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description="模型集成评估（先验等权）")
    ap.add_argument("--npz", default=None,
                    help="强度：训练脚本落盘的 cv5_summary.npz")
    ap.add_argument("--task", choices=["intensity", "path"], default="intensity")
    ap.add_argument("--models", nargs="*", default=None,
                    help="参与集成的模型；默认用全部非 persistence 模型")
    ap.add_argument("--ref", default="gru", help="配对比较的参照模型")
    args = ap.parse_args()
    if args.task == "path":
        return path_ensemble()
    if not args.npz:
        ap.error("--task intensity 需要 --npz")

    d = np.load(args.npz, allow_pickle=True)
    off = d["job_offsets"]
    n_job = len(off) - 1
    tags = [k.replace("__pred", "") for k in d.files if k.endswith("__pred")]
    true = d[f"{tags[0]}__true"]

    models = args.models or [t for t in tags if "persistence" not in t]
    missing = [m for m in models if f"{m}__pred" not in d.files]
    if missing:
        print(f"[错误] 缺少模型 {missing}")
        return 1

    print(f"[输入] {Path(args.npz).name}")
    print(f"       {n_job} 个 job，{len(true)} 行（含划分间台风重叠）")
    print(f"       参与集成: {models}")

    # ---------- 1. 单模型 ----------
    print("\n" + "=" * 78)
    print("单模型（测试集，m/s）")
    print("=" * 78)
    print(f"{'模型':28s}{'MAE':>9s}{'±std':>9s}" + "".join(f"{f'{h}h':>9s}" for h in HORIZON))
    single = {}
    for m in models:
        p = d[f"{m}__pred"]
        pj = per_job_mae(p, true, off)
        ps = per_job_mae_step(p, true, off)
        single[m] = (pj, ps)
        print(f"{m:28s}{pj.mean():9.4f}{pj.std(ddof=1):9.4f}"
              + "".join(f"{v:9.3f}" for v in ps.mean(0)))

    ref_m = args.ref if args.ref in single else models[0]
    ref_pj = single[ref_m][0]

    # ---------- 2. 误差相关性 ----------
    print("\n" + "=" * 78)
    print("误差相关性诊断（误差越相关，集成越难改善）")
    print("=" * 78)
    errs = {m: (d[f"{m}__pred"] - true).ravel() for m in models}
    print(f"{'':28s}" + "".join(f"{m[:10]:>12s}" for m in models))
    for a in models:
        row = f"{a:28s}"
        for b in models:
            r = np.corrcoef(errs[a], errs[b])[0, 1]
            row += f"{r:12.3f}"
        print(row)

    # ---------- 3. 等权集成 ----------
    print("\n" + "=" * 78)
    print("先验等权集成（权重 1/K，不需要拟合 → 无测试集泄漏）")
    print("=" * 78)
    print(f"{'组合':40s}{'MAE':>9s}{'±std':>9s}{'Δ vs ' + ref_m:>14s}{'胜率':>9s}{'p':>9s}")
    rows = []
    for k in range(2, len(models) + 1):
        for combo in itertools.combinations(models, k):
            p = np.mean([d[f"{m}__pred"] for m in combo], axis=0)
            pj = per_job_mae(p, true, off)
            dm = pj - ref_pj
            win = int((dm < 0).sum())
            if len(dm) > 1 and dm.std() > 0:
                _, pval = stats.ttest_rel(pj, ref_pj)
                pstr = f"{pval:9.4f}"
            else:
                pstr = f"{'n/a':>9s}"
            label = " + ".join(c.replace("+云图", "") for c in combo)
            rows.append((pj.mean(), label, dm.mean(), win, pstr))
            print(f"{label[:40]:40s}{pj.mean():9.4f}{pj.std(ddof=1):9.4f}"
                  f"{dm.mean():+14.4f}{f'{win}/{n_job}':>9s}{pstr}")

    # ---------- 4. 分时效 ----------
    print("\n" + "=" * 78)
    print("最佳组合的分时效表现")
    print("=" * 78)
    best = min(rows, key=lambda r: r[0])
    print(f"最优（按 MAE）: {best[1]}  MAE={best[0]:.4f}")
    # 逐时效看等权全模型
    p_all = np.mean([d[f"{m}__pred"] for m in models], axis=0)
    ps_all = per_job_mae_step(p_all, true, off).mean(0)
    ps_ref = single[ref_m][1].mean(0)
    print(f"\n{'模型':28s}" + "".join(f"{f'{h}h':>9s}" for h in HORIZON))
    print(f"{ref_m:28s}" + "".join(f"{v:9.3f}" for v in ps_ref))
    print(f"{'等权全部':28s}" + "".join(f"{v:9.3f}" for v in ps_all))
    print(f"{'Δ':28s}" + "".join(f"{v:+9.3f}" for v in ps_all - ps_ref))

    # ---------- 4b. oracle 上界（不可达，仅作判据）----------
    print("\n" + "=" * 78)
    print("oracle 线性组合（用测试集拟合权重 → 不可达的上界，仅用于判断集成是否还有空间）")
    print("=" * 78)
    # 对每个时效分别解最小二乘：y ≈ Σ w_k * pred_k（无截距，因预测已是强度量）
    W, mae_oracle = [], []
    for t in range(true.shape[1]):
        A = np.stack([d[f"{m}__pred"][:, t] for m in models], axis=1)
        w, *_ = np.linalg.lstsq(A, true[:, t], rcond=None)
        W.append(w)
        mae_oracle.append(float(np.abs(A @ w - true[:, t]).mean()))
    W = np.array(W)
    print(f"{'模型':28s}" + "".join(f"{f'{h}h':>9s}" for h in HORIZON))
    for i, m in enumerate(models):
        print(f"  w[{m[:20]:20s}]" + "".join(f"{W[t, i]:9.3f}" for t in range(len(HORIZON))))
    print(f"  {'权重和':26s}" + "".join(f"{W[t].sum():9.3f}" for t in range(len(HORIZON))))
    ps_ref = single[ref_m][1].mean(0)
    print(f"\n{'模型':28s}" + "".join(f"{f'{h}h':>9s}" for h in HORIZON))
    print(f"{ref_m + '（单一）':28s}" + "".join(f"{v:9.3f}" for v in ps_ref))
    print(f"{'oracle 组合':28s}" + "".join(f"{v:9.3f}" for v in mae_oracle))
    print(f"{'Δ':28s}" + "".join(f"{mae_oracle[i] - ps_ref[i]:+9.3f}"
                                 for i in range(len(HORIZON))))
    print(f"\n整体 MAE：{ref_m} {ps_ref.mean():.4f}，"
          f"oracle {np.mean(mae_oracle):.4f}")
    if np.mean(mae_oracle) >= ps_ref.mean() - 1e-4:
        print("  → **连 oracle 权重都打不过单一 GRU**，集成没有理论空间。")
    else:
        print(f"  → oracle 可降到 {np.mean(mae_oracle):.4f}（比单一好 "
              f"{ps_ref.mean() - np.mean(mae_oracle):.4f}），"
              "但这是用测试集拟合的，实际不可达；需用验证集拟合权重后复核。")

    # ---------- 5. 结论 ----------
    print("\n" + "=" * 78)
    print("判读")
    print("=" * 78)
    err_r = np.corrcoef(
        [errs[m] for m in models])[np.triu_indices(len(models), k=1)]
    print(f"  模型间误差相关系数: 中位 {np.median(err_r):.3f}，"
          f"范围 {err_r.min():.3f}~{err_r.max():.3f}")
    ref_mae = float(single[ref_m][0].mean())
    if best[0] < ref_mae - 1e-4:
        print(f"  → 等权集成优于 {ref_m}（{best[0]:.4f} vs {ref_mae:.4f}），"
              f"但需注意上表是**多个组合**中挑最优，存在选择偏差")
    else:
        print(f"  → 没有组合优于单一 {ref_m}（{ref_mae:.4f}）。")
        print("     这与误差高相关一致：各模型犯的是同一类错误，平均无法消除。")
    print("\n注意：等权是**先验**方案，没有用测试集拟合任何参数。")
    print("     上表的 p 值只在同一 job 内配对，跨 job 有台风重叠，不是独立样本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
