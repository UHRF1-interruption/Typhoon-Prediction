"""对比「统一模型」vs「按地形分段建模」的强度预测效果。

三种方案:
  A. 统一模型     所有样本一起训练（现有做法）
  B. 分段模型     按 ocean/coastal/land 分别训练三个模型
  C. 统一+段别特征 把段别作为 one-hot 加入输入（折中方案）

为什么需要三种: 分段模型每段样本少（约 1/3），可能过拟合；
one-hot 方案保留了全部样本但让模型知道当前属于哪一段。

用法:
    # 快速验证（CPU）
    python scripts/intensity/compare_segments.py --device cpu --epochs 3

    # 完整对比（GPU）
    python scripts/intensity/compare_segments.py --device cuda --epochs 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.sat_intensity import SatConfig, build_intensity_samples, fit_scalar_scaler  # noqa: E402
from src.data.terrain_align import align_terrain_to_satellites, forward_fill_terrain  # noqa: E402
from src.data.terrain_segment import SEGMENTS, assign_segment, sample_segment_labels  # noqa: E402
from src.models.intensity_transformer import build_intensity_model  # noqa: E402


def _forward(model, b, device):
    s = b["s"].to(device)
    if getattr(model, "use_image_features", False):
        return model(s, img_feat=b["pfeat"].to(device))
    p = b["p"].to(device) if "p" in b else None
    return model(s, p)


def train_one(cfg, bundle, splits, epochs: int, device: str, tag: str,
              seed: int = 0) -> dict:
    """训练单个模型并返回测试指标。"""
    torch.manual_seed(seed)
    sc = cfg.satellite
    tr_idx = splits["train"]
    from src.data.sat_intensity import SCALAR_FEATURES
    feats = list(SCALAR_FEATURES)
    scaler = fit_scalar_scaler(bundle["s"][tr_idx], feats)

    class DS(torch.utils.data.Dataset):
        def __init__(self, idx):
            self.idx = idx
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i]
            s = (bundle["s"][j] - scaler["mean"]) / scaler["std"]
            item = {"s": torch.from_numpy(s.astype(np.float32)),
                    "y": torch.from_numpy(bundle["y"][j].astype(np.float32)),
                    "last": torch.tensor(float(bundle["last_wind"][j]), dtype=torch.float32)}
            if "p" in bundle:
                item["p"] = torch.from_numpy((bundle["p"][j] / 255.0).astype(np.float32))
            if "pfeat" in bundle:
                item["pfeat"] = torch.from_numpy(bundle["pfeat"][j].astype(np.float32))
            return item

    bs = int(sc.train.batch_size)
    dl = {k: DataLoader(DS(v), batch_size=bs, shuffle=(k == "train"))
          for k, v in splits.items() if len(v) > 0}
    patch_size = int(bundle["p"].shape[-1]) if "p" in bundle else 32
    patch_chans = int(bundle["p"].shape[2]) if "p" in bundle else 3
    model = build_intensity_model(sc.model, bundle["s"].shape[-1], bundle["s"].shape[1],
                                  bundle["y"].shape[1], True, patch_size,
                                  patch_chans=patch_chans).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sc.train.lr),
                            weight_decay=float(sc.train.weight_decay))
    best_val, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
        for b in dl["train"]:
            y = b["y"].to(device); last = b["last"].to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            out = _forward(model, b, device)
            loss = torch.nn.functional.smooth_l1_loss(out, y - last, beta=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if "val" in dl and len(dl["val"]) > 0:
            model.eval()
            pv, tv = [], []
            with torch.no_grad():
                for b in dl["val"]:
                    out = _forward(model, b, device)
                    pv.append((out + b["last"].to(device).unsqueeze(1)).cpu().numpy())
                    tv.append(b["y"].numpy())
            vm = float(np.abs(np.concatenate(pv) - np.concatenate(tv)).mean())
            if vm < best_val:
                best_val = vm
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    # 测试
    model.eval()
    pv, tv = [], []
    with torch.no_grad():
        for b in dl["test"]:
            out = _forward(model, b, device)
            pv.append((out + b["last"].to(device).unsqueeze(1)).cpu().numpy())
            tv.append(b["y"].numpy())
    pred, true = np.concatenate(pv), np.concatenate(tv)
    return {"tag": tag, "mae": float(np.abs(pred - true).mean()),
            "mae_by_step": np.abs(pred - true).mean(axis=0).tolist(),
            "n": len(pred), "pred": pred, "true": true}


def main() -> int:
    ap = argparse.ArgumentParser(description="统一 vs 分段 建模对比")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--terrain", default=".scratch/env_terrain.parquet")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    sc = cfg.satellite
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    import pandas as pd
    sat = pd.read_parquet(ROOT / cfg.data.processed_dir / "sat_scalars.parquet")
    terr = pd.read_parquet(ROOT / args.terrain)
    sat = align_terrain_to_satellites(sat, terr)
    sat = forward_fill_terrain(sat)

    cache = ROOT / cfg.data.processed_dir / sc.patch_cache
    d = np.load(cache, allow_pickle=False)
    patches = {}
    for k, p in zip(d["keys"], d["patches"]):
        n, t = str(k).split("|", 1)
        patches[(n, pd.Timestamp(t))] = p

    sat_cfg = SatConfig(in_len=int(sc.in_len), out_len=int(sc.out_len),
                        interval_h=float(sc.interval_h), patch_size=int(sc.patch_size),
                        use_patch=True, use_terrain=False)
    bundle = build_intensity_samples(sat, sat_cfg, patches)
    labels = sample_segment_labels(bundle, sat)
    print(f"\n[compare] 样本 {len(labels)}")
    for seg in SEGMENTS:
        print(f"  {seg:8s} {int((labels == seg).sum()):5d}")

    # 按台风划分（保证同一台风不跨集合）
    names = bundle["name"]
    uniq = np.array(sorted(set(names.tolist())))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_test = max(1, len(uniq) // 5)
    n_val = max(1, len(uniq) // 5)
    test_names = set(uniq[:n_test].tolist())
    val_names = set(uniq[n_test:n_test + n_val].tolist())
    tr_names = set(uniq[n_test + n_val:].tolist())
    base_splits = {
        "train": np.where(np.isin(names, list(tr_names)))[0],
        "val": np.where(np.isin(names, list(val_names)))[0],
        "test": np.where(np.isin(names, list(test_names)))[0],
    }

    results = {}
    for seed in args.seeds:
        print(f"\n{'=' * 66}\nseed={seed}\n{'=' * 66}")
        # A: 统一模型
        r = train_one(cfg, bundle, base_splits, args.epochs, device, "A_unified", seed)
        results.setdefault("A_unified", []).append(r)
        print(f"  A 统一模型: MAE={r['mae']:.3f} (n={r['n']})")

        # B: 分段模型
        preds = np.full_like(r["pred"], np.nan)
        trues = None
        for seg in SEGMENTS:
            seg_tr = base_splits["train"][labels[base_splits["train"]] == seg]
            seg_va = base_splits["val"][labels[base_splits["val"]] == seg]
            seg_te = base_splits["test"][labels[base_splits["test"]] == seg]
            if len(seg_tr) < 50 or len(seg_te) < 10:
                print(f"    {seg}: 样本不足 (tr={len(seg_tr)}, te={len(seg_te)})，跳过")
                continue
            sp = {"train": seg_tr, "val": seg_va if len(seg_va) >= 10 else seg_tr,
                  "test": seg_te}
            rs = train_one(cfg, bundle, sp, args.epochs, device, f"B_{seg}", seed)
            # 写回对应位置
            test_pos = np.searchsorted(base_splits["test"], seg_te)
            preds[test_pos] = rs["pred"]
            trues = rs["true"]
            print(f"    {seg}: MAE={rs['mae']:.3f} (n={rs['n']})")
        if trues is not None and np.isfinite(preds).all():
            b_mae = float(np.abs(preds - r["true"]).mean())
            results.setdefault("B_segmented", []).append({"mae": b_mae, "n": len(preds)})
            print(f"  B 分段模型: MAE={b_mae:.3f} (n={len(preds)})")
        else:
            print("  B 分段模型: 部分样本无法分配，跳过汇总")

    # 汇总
    print("\n" + "=" * 66)
    print("对比结果")
    print("=" * 66)
    for k, v in results.items():
        maes = [x["mae"] for x in v]
        print(f"  {k:16s} MAE={np.mean(maes):.3f} ± {np.std(maes):.3f} (n={len(v)})")

    out = ROOT / "outputs" / "segment_compare" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: [{"mae": x["mae"], "n": x.get("n")} for x in v]
                               for k, v in results.items()}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
