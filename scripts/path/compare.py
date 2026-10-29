"""命令行入口: 基线对比 + 可视化。

在同一次运行里训练/评估多个模型，并在测试集上做公平对比:
  persistence(持续法) / linear(线性) / gru / transformer

用法:
  python scripts/compare.py                                   # 全部四个模型
  python scripts/compare.py --models persistence gru          # 只跑指定基线
  python scripts/compare.py --set train.epochs=3              # 减少轮数快速验证
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.dataset import load_bundle  # noqa: E402
from src.evaluation.plots import plot_mae_curve, plot_track_examples  # noqa: E402
from src.training.losses import format_metrics_table  # noqa: E402
from src.training.trainer import train  # noqa: E402

DEFAULT_MODELS = ["persistence", "linear", "gru", "transformer"]


def main() -> int:
    ap = argparse.ArgumentParser(description="多模型基线对比")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()

    base_cfg = load_config(args.config, args.set)
    bundle, spec = load_bundle(ROOT / base_cfg.data.processed_dir)
    interval = float(base_cfg.data.interval_hours)

    results, preds = {}, {}
    for name in args.models:
        cfg = load_config(args.config, args.set + [f"model.name={name}",
                                                   f"output.run_name=compare_{name}"])
        print("\n" + "#" * 70)
        print(f"# 模型: {name}")
        print("#" * 70)
        result = train(cfg, bundle, spec)
        if result["test_metrics"] is None:
            print(f"[skip] {name} 无测试集")
            continue
        m = result["test_metrics"]
        results[name] = m
        preds[name] = result["pred"]

    if not results:
        print("没有任何模型完成了测试集评估")
        return 1

    table = format_metrics_table(results, interval)
    print("\n" + "=" * 70)
    print("测试集路径误差（越大越差）")
    print("=" * 70)
    print(table)

    out_dir = ROOT / base_cfg.output.dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items() if kk not in {"pred_abs", "true_abs"}}
                    for k, v in results.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "metrics.md").write_text(table + "\n", encoding="utf-8")

    plot_mae_curve(results, interval, out_dir / "mae_vs_horizon.png")
    best = min(results, key=lambda k: results[k]["mae_overall"])
    if preds.get(best) is not None:
        plot_track_examples(preds[best], best, out_dir / "track_examples.png")
    print(f"\n图表与指标已保存到: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
