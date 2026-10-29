"""命令行入口: 训练模型。

用法:
  python scripts/train.py                                      # 用默认配置
  python scripts/train.py --set model.name=gru train.epochs=10 # 换基线
  python scripts/train.py --set device=cpu                     # 强制 CPU
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

import torch  # noqa: E402

from src.config import dump_config, load_config  # noqa: E402
from src.data.dataset import load_bundle, split_by_year  # noqa: E402
from src.training.losses import format_metrics_table  # noqa: E402
from src.training.trainer import train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="训练台风路径预测模型")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置，如 model.name=gru")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    processed = _data_path(cfg, cfg.data.processed_dir)
    bundle, spec = load_bundle(processed)
    print(f"[data] 载入 {len(bundle['x'])} 个样本，特征维度 {bundle['x'].shape[-1]}")

    result = train(cfg, bundle, spec)

    out_dir = result["out_dir"]
    dump_config(cfg, out_dir / "used_config.yaml")
    print("\n" + "=" * 70)
    print(f"训练完成: 最佳 epoch={result['best_epoch']}, 验证误差={result['best_val_mae_km']:.1f} km")
    if result["test_metrics"]:
        m = result["test_metrics"]
        row = {str(cfg.model.name): {"mae_by_step": m["mae_by_step"], "horizon": m["horizon"],
                                     "mae_overall": m["mae_overall"]}}
        print(format_metrics_table(row, float(cfg.data.interval_hours)))
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
