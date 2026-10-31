"""按「强度实际变化幅度」分层，检验卫星云图是否只在某些样本上有益。

动机
----
我们观察到两个任务的分时效结论方向相反：

| 任务 | 特征 | 短时效 | 长时效 |
|---|---|---|---|
| 路径 | 环境场 | Δr −0.003 | Δr +0.019 |
| 强度 | 云图 | Δr −0.067 | Δr −0.027 |

对强度，一个假说是：**3h 时几乎没有变化（MAE 0.62 m/s 已很小），
加任何结构特征都是净负担**；而强度快速变化的样本才可能有增量。
若成立，就应按 |Δ强度| 分层看到"平稳段最差、剧变段持平或略好"。

本脚本就是检验这个假说。若不成立，说明云图在所有样本上都只是噪声。

口径
----
* 分层依据：**实际** 12h 强度变化量的绝对值 |true[:, -1] − last|。
  用实际值而非预测值，避免"预测越准的样本被分到平稳组"的选择偏差。
* 每层内的 ΔMAE = 云图融合 MAE − GRU MAE（配对，同一样本）。
* 按 job 分别计算再汇总，因为不同划分之间有台风重叠，
  直接合并会把同一样本算 3 次（种子 42/0/7 各一次）。

用法:
    python scripts/evaluation/stratify_by_change.py \\
        --npz outputs/intensity_rerun_224_corr_base/cv5_..._summary.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

# 按 12h 实际强度变化量分层（m/s）。
# 断点依据：3 m/s 约等于一个热带风暴等级跨度，10 m/s 接近 RI 阈值。
BINS = [(0.0, 1.0, "稳定  <1"), (1.0, 3.0, "弱变  1-3"),
        (3.0, 6.0, "中变  3-6"), (6.0, 10.0, "强变  6-10"),
        (10.0, 1e9, "剧变  ≥10")]


def main() -> int:
    ap = argparse.ArgumentParser(description="按强度变化幅度分层")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--base", default="gru")
    ap.add_argument("--test", default="fusion_transformer+云图")
    ap.add_argument("--step", type=int, default=-1,
                    help="用第几个时效的变化量分层（-1=最后一个，默认 12h）")
    args = ap.parse_args()

    d = np.load(args.npz)
    off = d["job_offsets"]
    P0, T0 = d[f"{args.base}__pred"], d[f"{args.base}__true"]
    P1 = d[f"{args.test}__pred"]
    LAST = d[f"{args.base}__last"]
    n_job = len(off) - 1
    print(f"[输入] {n_job} 个 job，共 {len(T0)} 行（含划分间重复）")

    rows = []
    for j in range(n_job):
        a, b = int(off[j]), int(off[j + 1])
        t = T0[a:b]
        last = LAST[a:b].reshape(-1, 1)
        change = np.abs(t[:, args.step] - last[:, 0])
        e0 = np.abs(P0[a:b] - t).mean(axis=1)      # 样本级平均误差（跨时效）
        e1 = np.abs(P1[a:b] - t).mean(axis=1)
        for lo, hi, _ in BINS:
            m = (change >= lo) & (change < hi)
            if m.sum() < 20:
                continue
            rows.append((j, lo, m.sum(), e0[m].mean(), e1[m].mean()))

    if not rows:
        print("[错误] 没有足够的样本分层")
        return 1

    print("\n" + "=" * 78)
    print(f"按 12h 实际强度变化幅度分层（依据 |true[:,{args.step}] − 窗口末端强度|）")
    print("=" * 78)
    print(f"{'分层':14s}{'job数':>6s}{'样本/job':>10s}"
          f"{args.base[:8]:>10s}{args.test[:10]:>12s}{'ΔMAE':>10s}{'胜率':>9s}{'p':>9s}")
    print("-" * 78)
    for lo, hi, label in BINS:
        sel = [r for r in rows if r[1] == lo]
        if not sel:
            continue
        g = np.array([r[3] for r in sel])
        q = np.array([r[4] for r in sel])
        dm = q - g
        n = int(np.median([r[2] for r in sel]))
        win = int((dm < 0).sum())
        if len(dm) > 1 and dm.std() > 0:
            _, p = stats.ttest_rel(q, g)
            pstr = f"{p:9.4f}"
        else:
            pstr = f"{'n/a':>9s}"
        print(f"{label:14s}{len(sel):6d}{n:10d}"
              f"{g.mean():10.3f}{q.mean():12.3f}{dm.mean():+10.3f}"
              f"{f'{win}/{len(dm)}':>9s}{pstr}")

    print("\n判读：" )
    print("  若假说成立，应看到 ΔMAE 从「稳定」到「剧变」**单调下降**"
          "（甚至转为负）。")
    print("  若各层 ΔMAE 都为正且相近，说明云图在所有样本上都是净负担。")
    print("\n注意：行数为各 job 的**中位数**，跨 job 有台风重叠，"
          "p 值仅在同 job 内配对，不可当独立样本解读。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
