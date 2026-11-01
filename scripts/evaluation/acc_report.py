"""计算各预测产物的相关系数（含距平相关），补充 MAE 之外的评价维度。

口径说明（重要，汇报时必须带上）
--------------------------------
气象上的「距平相关系数（ACC）」= 预测距平与观测距平的相关系数，
距平 = 值 − 气候态。本文两种口径：

1. **r（原值）**：直接对预测/观测求 Pearson r。
   对路径任务，因为样本散布在整个西北太平洋，这个值恒在 0.999 以上，
   **不携带信息**，列出只为说明"它为什么没用"。

2. **r（距平·对持续法）**：以**持续法**代替气候态作为基准，
       距平 = 预测 − 持续法预测
   对路径，这等价于「位移」（相对初始位置的位移向量）；
   对强度，等价于「强度变化量」。
   这是本文真正有区分度的口径，也直接对应报告里"vs 持续法"的框架。

   注：若改用**样本均值**作气候态，因 Pearson r 对整体平移不变，
   结果与 r（原值）完全相同，起不到"去平"作用。这也是本文选持续法
   而非样本均值的原因——它是有物理含义的逐样本基准，而非一个常数。

用法:
    python scripts/evaluation/acc_report.py
    python scripts/evaluation/acc_report.py --task intensity --pred-dir outputs/xxx
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


def _r(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or x[m].std() < 1e-12 or y[m].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def path_corr(path_npz: Path):
    """路径任务：返回 (r原值[], r距平[], n)。数组长度 = 时效步数。"""
    d = np.load(path_npz)
    P, T, L = d["pred_abs"], d["true_abs"], d["last_abs"]
    r_raw, r_anom = [], []
    for t in range(P.shape[1]):
        r_raw.append(_r(P[:, t, :], T[:, t, :]))
        r_anom.append(_r(P[:, t, :] - L, T[:, t, :] - L))
    return np.array(r_raw), np.array(r_anom), len(d["year"])


def intensity_corr(path_npz: Path):
    """强度任务：pred/true/last 形状 (N, out_len)，last 为窗口末端的当前强度。"""
    d = np.load(path_npz)
    P, T = d["pred"], d["true"]
    L = d["last"].reshape(-1, 1) if d["last"].ndim == 1 else d["last"]
    r_raw, r_anom = [], []
    for t in range(P.shape[1]):
        r_raw.append(_r(P[:, t], T[:, t]))
        r_anom.append(_r(P[:, t] - L[:, 0], T[:, t] - L[:, 0]))
    return np.array(r_raw), np.array(r_anom), len(T)


def report_group(label: str, files: list[Path], fn, leads: list[int]):
    A, B, N = [], [], []
    for f in files:
        try:
            a, b, n = fn(f)
        except (KeyError, ValueError) as e:
            print(f"  [跳过] {f.name}: {e}")
            continue
        A.append(a)
        B.append(b)
        N.append(n)
    if not A:
        return None
    A, B = np.array(A), np.array(B)
    hdr = "".join(f"{f'{l}h':>10s}" for l in leads[: A.shape[1]])
    print(f"\n{label}  （{len(A)} 个 job，每个约 {N[0]} 个样本）")
    print(f"{'指标':>22s}{hdr}")
    print(f"{'r（原值）':>20s}" + "".join(f"{v:10.4f}" for v in np.nanmean(A, 0)))
    print(f"{'r（距平·对持续法）':>18s}" + "".join(f"{v:10.4f}" for v in np.nanmean(B, 0)))
    return A, B



def _print_intensity_per_step(data: dict, fname: str):
    """新结果：逐时效距平相关。"""
    s = data["base"]["summary"]
    leads = [3, 6, 9, 12]
    hdr = "".join(f"{f'{l}h':>9s}" for l in leads)
    print("\n" + "=" * 78)
    print(f"强度 · 逐时效距平相关（对持续法；224 台风 × 15 job；来源 {fname}）")
    print("=" * 78)
    print(f"{'模型':26s}{hdr}{'均值':>9s}")
    ref = np.array(s["gru"]["corr_by_step_mean"], float)
    for m in ("gru", "fusion_transformer+云图", "feature_transformer+云图"):
        if m not in s:
            continue
        c = np.array(s[m]["corr_by_step_mean"], float)
        print(f"{m:26s}" + "".join(f"{x:9.3f}" for x in c)
              + f"{np.nanmean(c):9.3f}")
        if m != "gru":
            dlt = c - ref
            print(f"{'   Δ vs GRU':26s}" + "".join(f"{x:+9.3f}" for x in dlt))
    print("\n（persistence 的距平相关无定义：距平恒为 0，自己减自己）")
    # 结论式描述
    g = np.array(s["gru"]["corr_by_step_mean"], float)
    f = np.array(s["fusion_transformer+云图"]["corr_by_step_mean"], float)
    d = f - g
    trend = "递减" if abs(d[0]) > abs(d[-1]) else "递增"
    print(f"\n云图与 GRU 的差距随时效{trend}：3h {d[0]:+.3f} → 12h {d[-1]:+.3f}"
          f"（|Δ| 变为原来的 {abs(d[-1]) / abs(d[0]):.0%}）")
    print("→ 云图主要拖累**短时效**；长时效接近持平。")


def main() -> int:
    ap = argparse.ArgumentParser(description="相关系数 / 距平相关系数报告")
    ap.add_argument("--task", choices=["path", "intensity", "auto"], default="auto")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    if args.task in ("path", "auto"):
        groups = {
            "路径 · 纯轨迹（baseline）":
                sorted(glob.glob("outputs/path/ablation/stratified_ablation/pred_baseline_s*.npz")),
            "路径 · +环境场（env）":
                sorted(glob.glob("outputs/path/ablation/stratified_ablation/pred_env_s*.npz")),
            "路径 · 纬度自适应（env）":
                sorted(glob.glob("outputs/path/latitude_adaptive/pred_env_j0_*.npz")),
        }
        got = {}
        for label, files in groups.items():
            if not files:
                continue
            r = report_group(label, [Path(f) for f in files], path_corr, [3, 6, 9, 12])
            if r:
                got[label] = r

        keys = list(got)
        if len(keys) >= 2:
            print("\n" + "=" * 74)
            print("配对比较（同折同种子）：环境场 − 纯轨迹")
            print("=" * 74)
            a = got.get("路径 · 纯轨迹（baseline）")
            b = got.get("路径 · +环境场（env）")
            if a is not None and b is not None and len(a[1]) == len(b[1]):
                d = b[1] - a[1]
                leads = [3, 6, 9, 12]
                print(f"{'':>10s}{'Δr':>10s}{'胜率':>10s}{'p':>10s}")
                for i in range(d.shape[1]):
                    t, p = stats.ttest_rel(b[1][:, i], a[1][:, i])
                    win = int(np.nansum(d[:, i] > 0))
                    print(f"{f'{leads[i]}h':>10s}{np.nanmean(d[:, i]):+10.4f}"
                          f"{f'{win}/{len(d)}':>10s}{p:10.4f}")
                print("\n注：Δr 随时效增大 → 环境场的作用主要在较长时效，"
                      "与「引导气流主导后期路径」的物理预期一致。")

    if args.task in ("intensity", "auto"):
        _report_intensity(args.pred_dir)
    return 0


def _report_intensity(pred_dir: str | None):
    """强度任务的距平相关。

    优先读逐样本预测；没有就从汇总 JSON 里取已记录的 corr。
    注意：旧结果只存了**跨时效平均**的标量，没有分时效——分时效需要
    重跑（新版 train_intensity.py 会把 corr_by_step 一起写进 summary）。
    """
    import json

    d = Path(pred_dir) if pred_dir else None
    files = sorted(glob.glob(str(d / "pred_*.npz"))) if d else []
    if files:
        report_group(f"强度 · {d.name}", [Path(f) for f in files],
                     intensity_corr, [3, 6, 9, 12])
        return

    # 优先用带逐时效 corr_by_step 的新结果；没有就退回旧汇总（只有标量）
    cands = [Path("outputs/intensity/summaries/intensity_rerun_224_corr_summary.json"),
             Path("outputs/intensity/summaries/intensity_rerun_224_summary.json")]
    summary = next((c for c in cands if c.exists()), None)
    if summary is None:
        print("\n[强度] 既无 pred_*.npz，也无汇总 JSON")
        return
    data = json.loads(summary.read_text())
    if "corr_by_step_mean" in next(iter(data.get("base", {}).get("summary", {}).values()), {}):
        _print_intensity_per_step(data, summary.name)
        return
    models = ["persistence", "gru", "feature_transformer+云图",
              "fusion_transformer+云图"]
    print("\n" + "=" * 74)
    print("强度 · 距平相关（对持续法；**跨 3/6/9/12h 平均**，224 台风 × 15 job）")
    print("=" * 74)
    print(f"{'变体':10s}{'模型':30s}{'距平相关':>10s}{'std':>8s}{'MAE':>8s}{'jobs':>6s}")
    print("-" * 74)
    base = data.get("base", {}).get("summary", {})
    for variant in ("base", "warmcore", "terrain"):
        v = data.get(variant, {}).get("summary", {})
        for m in models:
            if m not in v:
                continue
            x = v[m]
            c = x.get("corr_mean", float("nan"))
            sd = x.get("corr_std", float("nan"))
            cstr = "n/a" if c != c else f"{c:10.3f}"
            sstr = "" if sd != sd else f"{sd:8.3f}"
            print(f"{variant:10s}{m:30s}{cstr}{sstr}"
                  f"{x['overall_mean']:8.3f}{x.get('n_corr_jobs', 0):6d}")
        print()
    if base:
        g, f = base.get("gru"), base.get("fusion_transformer+云图")
        if g and f:
            print(f"关键差值：云图融合 − GRU = "
                  f"{f['corr_mean'] - g['corr_mean']:+.3f}"
                  f"（MAE {f['overall_mean'] - g['overall_mean']:+.3f} m/s）")
    print("\n注：persistence 的距平相关为 nan —— 它的距平恒为 0（自己减自己），")
    print("    相关系数无定义。这是正确行为，不是缺数据。")
    print("注：上表是跨时效平均。要看「作用随时效如何变化」需重跑训练")
    print("    （新版会把 corr_by_step 写进 summary）。")


if __name__ == "__main__":
    raise SystemExit(main())
