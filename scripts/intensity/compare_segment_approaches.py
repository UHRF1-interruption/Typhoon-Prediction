"""对比三种地形信息利用方式（不拆分样本）。

为什么是这三种
--------------
之前验证过「按地形分段训练独立模型」无效（MAE 1.703 vs 统一 1.564），
原因是每段样本降到 1/3，训练不足。但仍存在一个已验证的事实：
    纯海洋 12h 平均 +1.19 m/s，陆地 −1.79 m/s（方向相反）

本脚本用两种**不减少样本量**的方式利用这个信息：

  方式1  one-hot 拼接
      输入特征末尾追加 3 维段别 one-hot（ocean/coastal/land）。
      样本量不变，模型自己决定是否利用段别信息。
      代价: 特征多 3 维。

  方式2  残差修正
      先用统一模型预测，再加一个**每段一个标量**的偏置修正：
          final = pred + bias[segment]
      bias 只有 3 个参数，用训练集拟合（每段的平均残差），几乎不可能过拟合。
      这是最符合「针对性修正」思路的实现。

基线:
  baseline  统一模型，不加任何地形信息（现有做法）

用法:
    python scripts/intensity/compare_segment_approaches.py --device cuda --epochs 60
    python scripts/intensity/compare_segment_approaches.py --device cpu --epochs 5
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
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.sat_intensity import (  # noqa: E402
    SCALAR_FEATURES, SatConfig, build_intensity_samples,
)
from src.data.terrain_align import align_terrain_to_satellites, forward_fill_terrain  # noqa: E402
from src.data.terrain_segment import SEGMENTS, assign_segment, sample_segment_labels  # noqa: E402
from src.models.intensity_transformer import build_intensity_model  # noqa: E402

SEG_INDEX = {s: i for i, s in enumerate(SEGMENTS)}


class SegDataset(Dataset):
    """带段别标签的强度 Dataset。

    extra_onehot: 是否在标量末尾拼接 3 维 one-hot
    """

    def __init__(self, bundle, idx, scaler, extra_onehot: bool):
        self.idx = idx
        self.extra_onehot = extra_onehot
        self.s = bundle["s"][idx]
        self.y = bundle["y"][idx]
        self.last = bundle["last_wind"][idx]
        self.p = bundle.get("p")
        self.p = self.p[idx] if self.p is not None else None
        self.pfeat = bundle.get("pfeat")
        self.pfeat = self.pfeat[idx] if self.pfeat is not None else None
        self.seg = bundle["_seg_idx"][idx]          # int 数组 0/1/2

        mean = np.asarray(scaler["mean"], np.float32)
        std = np.asarray(scaler["std"], np.float32)
        self.s = (self.s - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        s = self.s[i]
        if self.extra_onehot:
            oh = np.zeros((s.shape[0], len(SEGMENTS)), np.float32)
            oh[:, self.seg[i]] = 1.0
            s = np.concatenate([s, oh], axis=-1)
        item = {
            "s": torch.from_numpy(s.astype(np.float32)),
            "y": torch.from_numpy(self.y[i].astype(np.float32)),
            "last": torch.tensor(float(self.last[i]), dtype=torch.float32),
            "seg": torch.tensor(int(self.seg[i]), dtype=torch.long),
        }
        if self.p is not None:
            item["p"] = torch.from_numpy((self.p[i] / 255.0).astype(np.float32))
        if self.pfeat is not None:
            item["pfeat"] = torch.from_numpy(self.pfeat[i].astype(np.float32))
        return item


def _forward(model, b, device):
    s = b["s"].to(device)
    if getattr(model, "use_image_features", False):
        return model(s, img_feat=b["pfeat"].to(device))
    p = b["p"].to(device) if "p" in b else None
    return model(s, p)


def train_model(cfg, bundle, splits, scaler, onehot: bool, epochs: int,
                device: str, seed: int) -> dict:
    """训练一个模型，返回 model 与各集合的预测。"""
    torch.manual_seed(seed)
    sc = cfg.satellite
    ds = {k: SegDataset(bundle, v, scaler, onehot) for k, v in splits.items() if len(v)}
    bs = int(sc.train.batch_size)
    dl = {k: DataLoader(v, batch_size=bs, shuffle=(k == "train")) for k, v in ds.items()}

    # 注意: ds.s 是标准化后的原始特征（不含 one-hot），
    # one-hot 在 __getitem__ 里拼接，因此要用 batch 实际维度
    feat_dim = ds["train"].s.shape[-1] + (len(SEGMENTS) if onehot else 0)
    patch_size = int(bundle["p"].shape[-1]) if "p" in bundle else 32
    patch_chans = int(bundle["p"].shape[2]) if "p" in bundle else 3
    model = build_intensity_model(sc.model, feat_dim, bundle["s"].shape[1],
                                  bundle["y"].shape[1], True, patch_size,
                                  patch_chans=patch_chans).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sc.train.lr),
                            weight_decay=float(sc.train.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for b in dl["train"]:
            y = b["y"].to(device)
            last = b["last"].to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            out = _forward(model, b, device)
            loss = torch.nn.functional.smooth_l1_loss(out, y - last, beta=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if "val" in dl:
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

    model.eval()
    out = {}
    with torch.no_grad():
        for k, loader in dl.items():
            pv, tv, sg, lw = [], [], [], []
            for b in loader:
                o = _forward(model, b, device)
                pv.append((o + b["last"].to(device).unsqueeze(1)).cpu().numpy())
                tv.append(b["y"].numpy())
                sg.append(b["seg"].numpy())
                lw.append(b["last"].numpy())
            out[k] = {"pred": np.concatenate(pv), "true": np.concatenate(tv),
                      "seg": np.concatenate(sg), "last": np.concatenate(lw)}
    return {"model": model, "preds": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="对比 one-hot vs 残差修正")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--terrain", default=".scratch/env_terrain.parquet")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    sc = cfg.satellite
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
    bundle["_seg_idx"] = np.array([SEG_INDEX.get(s, 0) for s in labels], dtype=np.int64)
    n_total = len(labels)
    print(f"\n[compare] 样本 {n_total}")
    for s in SEGMENTS:
        print(f"  {s:8s} {int((labels == s).sum()):5d}")

    # 标准化参数（只用训练集会填，先按全体算，后面覆盖）
    from src.data.sat_intensity import fit_scalar_scaler
    names = bundle["name"]
    uniq = np.array(sorted(set(names.tolist())))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_test = max(1, len(uniq) // 5)
    n_val = max(1, len(uniq) // 5)
    test_names = set(uniq[:n_test].tolist())
    val_names = set(uniq[n_test:n_test + n_val].tolist())
    tr_names = set(uniq[n_test + n_val:].tolist())
    splits = {
        "train": np.where(np.isin(names, list(tr_names)))[0],
        "val": np.where(np.isin(names, list(val_names)))[0],
        "test": np.where(np.isin(names, list(test_names)))[0],
    }
    scaler = fit_scalar_scaler(bundle["s"][splits["train"]], list(SCALAR_FEATURES))

    results = {}
    for seed in args.seeds:
        print(f"\n{'=' * 66}\nseed={seed}\n{'=' * 66}")

        # ---------- baseline: 统一模型，无段别信息 ----------
        r0 = train_model(cfg, bundle, splits, scaler, False, args.epochs, device, seed)
        te0 = r0["preds"]["test"]
        mae0 = float(np.abs(te0["pred"] - te0["true"]).mean())
        results.setdefault("baseline(统一)", []).append({"mae": mae0})
        print(f"  baseline(统一)       MAE={mae0:.4f}")

        # ---------- 方式1: one-hot 拼接 ----------
        r1 = train_model(cfg, bundle, splits, scaler, True, args.epochs, device, seed)
        te1 = r1["preds"]["test"]
        mae1 = float(np.abs(te1["pred"] - te1["true"]).mean())
        results.setdefault("方式1(one-hot)", []).append({"mae": mae1})
        print(f"  方式1(one-hot)       MAE={mae1:.4f}")

        # ---------- 方式2: 残差修正 ----------
        # 用训练集的残差拟合每段偏置（3 个标量）
        tr = r0["preds"]["train"]
        resid = (tr["true"] - tr["pred"]).reshape(len(tr["true"]), -1)
        bias = {}
        for si, seg in enumerate(SEGMENTS):
            m = tr["seg"] == si
            bias[si] = float(resid[m].mean()) if m.sum() >= 5 else 0.0
        print(f"    段偏置: " + ", ".join(f"{SEGMENTS[i]}={bias[i]:+.3f}" for i in range(len(SEGMENTS))))
        corrected = te0["pred"] + np.array([bias.get(int(s), 0.0) for s in te0["seg"]])[:, None]
        mae2 = float(np.abs(corrected - te0["true"]).mean())
        results.setdefault("方式2(残差修正)", []).append({"mae": mae2})
        print(f"  方式2(残差修正)      MAE={mae2:.4f}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 66)
    print("对比结果（测试集 MAE, m/s）")
    print("=" * 66)
    base = np.mean([x["mae"] for x in results["baseline(统一)"]])
    for k, v in results.items():
        maes = np.array([x["mae"] for x in v])
        delta = maes.mean() - base
        mark = "  ← 更好" if delta < -1e-4 else ("  ← 更差" if delta > 1e-4 else "")
        print(f"  {k:18s} {maes.mean():.4f} ± {maes.std():.4f}   "
              f"({delta:+.4f} vs baseline){mark}")

    out = ROOT / "outputs" / "segment_approaches" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: [{"mae": x["mae"]} for x in v]
                               for k, v in results.items()}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
