"""A 阶段：微波接入的快速链路验证（单折、少 epoch）。

目的不是得出结论，而是**确认链路正确**：
    标量表 -> 轨迹窗口 -> 挂微波锚观测 -> Dataset -> 模型前向 -> 指标

对照三组（同一折、同一 seed）：
    gru            纯标量时序（现有基线）
    mw_transformer 标量 + 微波锚观测
    mw_only        仅看微波的朴素回归（证明微波本身有信息）

用法:
    python scripts/intensity/smoke_microwave.py --epochs 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _data_path(cfg, value):
    from src.config import resolve_path
    return resolve_path(cfg, value, ROOT)


sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.microwave import attach_microwave, load_microwave  # noqa: E402
from src.data.sat_intensity import (  # noqa: E402
    SCALAR_FEATURES, SatConfig, build_intensity_samples, fit_scalar_scaler,
)
from src.models.intensity_transformer import build_intensity_model  # noqa: E402
from src.data.fysatty import patch_cache_name  # noqa: E402


class MWDataset(Dataset):
    def __init__(self, b, idx, scl, pix):
        self.s = b["s"][idx]
        self.y = b["y"][idx]
        # 统计量模式下 bundle 里没有 "mw"（只有拼进 s 的 mwfeat）
        self.mw = b["mw"][idx] if "mw" in b else None
        self.dt = b["mw_dt"][idx] if "mw_dt" in b else np.zeros(len(idx), np.float32)
        self.mask = b["mw_mask"][idx] if "mw_mask" in b else np.zeros((len(idx), 4), np.uint8)
        # has: 该样本是否真的有微波观测。dual_head 靠它路由，
        # 不能用全零 patch 或 dt=0 推断（那与"有观测但 dt 恰为 0"无法区分）
        self.has = b["has_mw"][idx].astype(bool)
        if scl is not None:
            self.s = (self.s - np.asarray(scl["mean"], np.float32)) / \
                     np.asarray(scl["std"], np.float32)
        if pix is not None and self.mw is not None:
            m, s = pix
            self.mw = (self.mw - m[None, :, None, None]) / s[None, :, None, None]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "s": torch.from_numpy(self.s[i].astype(np.float32)),
            "y": torch.from_numpy(self.y[i].astype(np.float32)),
            "mw": (torch.zeros(4, 64, 64) if self.mw is None
                   else torch.from_numpy(self.mw[i].astype(np.float32))),
            "dt": torch.tensor(float(self.dt[i]) if np.isfinite(self.dt[i]) else 0.0,
                               dtype=torch.float32),
            "mask": torch.from_numpy(self.mask[i].astype(np.float32)),
            "has": bool(self.has[i]),
        }


def kfold_by_storm(names, k, seed):
    uniq = np.array(sorted(set(names.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    out = []
    for i in range(k):
        test = set(folds[i].tolist())
        rest = np.concatenate([folds[j] for j in range(k) if j != i])
        n_val = max(1, int(len(rest) * 0.15))
        val = set(rest[:n_val].tolist())
        tr = set(rest[n_val:].tolist())
        m = lambda st: np.where(np.isin(names, list(st)))[0]  # noqa: E731
        out.append({"train": m(tr), "val": m(val), "test": m(test)})
    return out


def _forward(model, b, dev, use_mw, is_dual):
    """统一前向：三种模型共用。dual_head 额外传 has_mw 以路由两个头。"""
    s = b["s"].to(dev)
    if not use_mw:
        return model(s)
    kw = dict(p=None, mw=b["mw"].to(dev), mw_dt=b["dt"].to(dev),
              mw_mask=b["mask"].to(dev))
    if is_dual:
        kw["has_mw"] = b["has"].to(dev)
    return model(s, **kw)


def evaluate(model, dl, dev, use_mw, is_dual=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in dl:
            out = _forward(model, b, dev, use_mw, is_dual)
            preds.append(out.cpu().numpy())
            trues.append(b["y"].numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return np.abs(p - t).mean(axis=0), float(np.abs(p - t).mean())


def evaluate_split(model, dl, dev, use_mw, is_dual):
    """按"是否带微波"拆开评估，用来定位差异来源。

    双头模型的结构性质：无微波样本走的分支与 GRU 基线逐位同构。
    因此整体差异**必然**来自带微波的那部分样本——除非共享的 GRU
    编码器被融合头拉偏。分开看就能区分这两种情况。
    """
    model.eval()
    P, T, H = [], [], []
    with torch.no_grad():
        for b in dl:
            out = _forward(model, b, dev, use_mw, is_dual)
            P.append(out.cpu().numpy())
            T.append(b["y"].numpy())
            H.append(b["has"].numpy())
    p, t, h = np.concatenate(P), np.concatenate(T), np.concatenate(H)
    err = np.abs(p - t)
    res = {}
    for tag, m in (("有微波", h), ("无微波", ~h)):
        res[tag] = (float(err[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)
    return res


def train_one(cfg, bundle, splits, scl, pix, mname, epochs, dev, bs=64,
              patience=15, mw_loss_weight=0.0):
    torch.manual_seed(0)
    n_step = bundle["y"].shape[1]
    use_mw = mname in ("mw_transformer", "dual_head")
    is_dual = mname == "dual_head"
    mw_size = bundle["mw"].shape[-1] if "mw" in bundle else 64
    if mname == "gru":
        mcfg = type("C", (), {"name": "gru", "hidden": 96, "num_layers": 2,
                              "dropout": 0.2})()
        model = build_intensity_model(mcfg, bundle["s"].shape[-1],
                                      bundle["s"].shape[1], n_step, False,
                                      mw_size).to(dev)
    else:
        mcfg = type("C", (), {"name": mname, "hidden": 96,
                              "num_layers": 2, "dropout": 0.2, "img_feat": 64})()
        model = build_intensity_model(mcfg, bundle["s"].shape[-1],
                                      bundle["s"].shape[1], n_step, True,
                                      bundle["mw"].shape[-1],
                                      patch_chans=bundle["mw"].shape[1]).to(dev)
    ds = {k: MWDataset(bundle, splits[k], scl, pix if use_mw else None)
          for k in ("train", "val", "test")}
    pin = dev.type == "cuda"
    dl = {k: DataLoader(v, batch_size=bs, shuffle=(k == "train"),
                        num_workers=0, pin_memory=pin)
          for k, v in ds.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = dev.type == "cuda"
    amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    for ep in range(epochs):
        model.train()
        for b in dl["train"]:
            y = b["y"].to(dev)
            last = b["s"][:, -1, 0].to(dev).unsqueeze(1)   # 未标准化的强度在 s[:,:,0]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = _forward(model, b, dev, use_mw, is_dual)
                # 逐元素 smooth_l1，便于做样本加权（默认权重全 1）
                per = torch.nn.functional.smooth_l1_loss(
                    out, y, beta=2.0, reduction="none")
                if is_dual and mw_loss_weight:
                    # 融合头只从带微波样本拿梯度（约占 1/3）。
                    # 加权把两个头的有效梯度量级拉平，否则融合头欠拟合
                    # 会被误读成"架构更差"（§7.4 的陷阱）。
                    w = 1.0 + (mw_loss_weight - 1.0) * b["has"].to(dev).float()
                    w = w / w.mean()          # 归一化，保持总损失量级不变
                    loss = (per * w.unsqueeze(-1)).mean()
                else:
                    loss = per.mean()
            if use_amp:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        sched.step()
        _, vm = evaluate(model, dl["val"], dev, use_mw, is_dual)
        if vm < best - 1e-4:
            best, best_ep, bad = vm, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience and bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    per_step, overall = evaluate(model, dl["test"], dev, use_mw, is_dual)
    split = evaluate_split(model, dl["test"], dev, use_mw, is_dual)
    n_param = sum(p.numel() for p in model.parameters())
    return per_step, overall, n_param, best_ep, split


def main() -> int:
    ap = argparse.ArgumentParser(description="微波接入链路冒烟验证")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--fold-seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--max-dt-h", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0=自动（cuda 用 256，cpu 用 64）")
    ap.add_argument("--only-mw", action="store_true",
                    help="只用带微波的样本（A1 方案）。默认 False=使用全部样本")
    ap.add_argument("--dual", action="store_true",
                    help="双头模型：有微波走融合头、无微波走标量头（标量头与 GRU "
                         "基线同构，故无微波时逐位退化为基线）。"
                         "与 --stats 互斥")
    ap.add_argument("--mw-loss-weight", type=float, default=0.0,
                    help="双头模型下带微波样本的损失权重。0=自动"
                         "（取 (1-p)/p 拉平两个头的梯度量级），1=不加权")
    ap.add_argument("--stats", action="store_true",
                    help="A1-1：把微波手工统计量拼进标量特征，用同一个 GRU 训练。"
                         "用于判定\"微波有信息但 CNN 学不好\"还是\"信息冗余\"")
    args = ap.parse_args()
    if (not args.dual) and (not args.only_mw) and (not args.stats):
        print("[警告] 未加 --dual 也未加 --only-mw：全量样本里有 65.6% 没有微波，"
              "零填充 patch 会污染 CNN 的 BatchNorm。混合场景请用 --dual。")
    if args.dual and args.stats:
        print("[错误] --dual 与 --stats 互斥（前者用 patch，后者用统计量）")
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[提示] cuda 不可用，改用 cpu")
        args.device = "cpu"
    dev = torch.device(args.device)

    cfg = load_config(args.config, args.set)
    sc = cfg.satellite
    proc = _data_path(cfg, cfg.data.processed_dir)

    print("[1] 载入标量表与微波 ...")
    scalars = pd.read_parquet(proc / "sat_scalars.parquet")
    mw_index = load_microwave(proc)

    print("[2] 构造标量序列样本（复用现有逻辑，不带云图）...")
    cfg_sat = SatConfig(in_len=int(sc.in_len), out_len=int(sc.out_len),
                        interval_h=float(sc.interval_h),
                        patch_size=int(sc.patch_size), use_patch=False)
    bundle = build_intensity_samples(scalars, cfg_sat, None)

    print("[3] 挂载微波 ...")
    if args.stats:
        from src.data.microwave import MW_FEATURE_NAMES, attach_mw_features
        bundle = attach_mw_features(bundle, mw_index)
        # 把统计量拼到标量特征末尾（只对带微波的样本有意义；
        # 用 --only-mw 时恒有值，否则缺失处置 0 并由 has_mw 区分）
        feats = np.nan_to_num(bundle["mwfeat"], nan=0.0)
        T = bundle["s"].shape[1]
        # 统计量是"窗口末端的观测"，广播到每一步（与标量序列同长）
        bundle["s"] = np.concatenate(
            [bundle["s"], np.repeat(feats[:, None, :], T, axis=1)], axis=-1)
        SCALAR_FEATURES.extend(MW_FEATURE_NAMES)
        print(f"    标量特征扩到 {bundle['s'].shape[-1]} 维")
    else:
        bundle = attach_microwave(bundle, mw_index, args.max_dt_h)

    if args.dual and args.only_mw:
        print("[提示] --dual 配合 --only-mw 时所有样本都有微波，"
              "标量头收不到梯度，等于退化成一个融合模型；"
              "建议去掉 --only-mw 以检验路由本身的价值")
    if args.only_mw:
        from src.data.microwave import subset_with_mw
        n0 = len(bundle["y"])
        bundle = subset_with_mw(bundle)
        print(f"[A1] 只保留带微波样本: {n0} -> {len(bundle['y'])} "
              f"({len(bundle['y']) / n0:.1%})")

    # 标准化参数：标量用训练集拟合；微波像素用**带微波样本**的均值/方差
    has = bundle["has_mw"]
    if has.sum() == 0:
        print("[错误] 没有任何样本带微波，检查 names 字段是否已补")
        return 1
    scl = fit_scalar_scaler(bundle["s"], SCALAR_FEATURES)
    if args.stats:
        pix = None          # 统计量模式不需要 patch 像素标准化
        print("    统计量模式：无像素标准化")
    else:
        px = bundle["mw"][has]
        pix = (np.nanmean(px, axis=(0, 2, 3)).astype(np.float32),
               np.maximum(np.nanstd(px, axis=(0, 2, 3)), 1e-3).astype(np.float32))
        print(f"    微波像素 均值={np.round(pix[0], 1).tolist()} "
              f"标准差={np.round(pix[1], 1).tolist()}")

    results = {}
    for fs in args.fold_seeds:
        folds = kfold_by_storm(bundle["name"], args.folds, fs)
        for fi, splits in enumerate(folds):
            n_te = len(splits["test"])
            n_mw = int(bundle["has_mw"][splits["test"]].sum())
            print(f"\n=== seed={fs} fold={fi + 1}/{args.folds}  "
                  f"test={n_te} 样本 (带微波 {n_mw}) ===")
            if args.stats:
                models = ("gru",)
            elif args.dual:
                models = ("gru", "dual_head")
            else:
                models = ("gru", "mw_transformer")
            for mname in models:
                bs = args.batch_size or (256 if dev.type == "cuda" else 64)
                w = args.mw_loss_weight
                if args.dual and w <= 0:
                    # 自动：按训练集里带微波的比例算，把两个头梯度量级拉平
                    p_mw = float(bundle["has_mw"][splits["train"]].mean())
                    w = (1.0 - p_mw) / max(p_mw, 1e-6)
                    w = float(np.clip(w, 1.0, 6.0))
                ps, ov, npar, bep, spl = train_one(cfg, bundle, splits, scl, pix,
                                                   mname, args.epochs, dev, bs=bs,
                                                   mw_loss_weight=w)
                results.setdefault(mname, []).append((ps, ov, bep, spl))
                extra = f" w={w:.2f}" if (args.dual and mname == "dual_head") else ""
                print(f"  {mname:16s} MAE={ov:.3f} "
                      f"[{' '.join(f'{x:.2f}' for x in ps)}] "
                      f"params={npar:,} best_ep={bep}{extra}  "
                      f"有无微波={spl['有微波'][0]:.3f}/{spl['无微波'][0]:.3f}")

    print("\n" + "=" * 60)
    print(f"冒烟结果（{args.epochs} epoch, {len(args.fold_seeds) * args.folds} job）")
    print("=" * 60)
    for mname, lst in results.items():
        ov = np.array([x[1] for x in lst])
        be = [x[2] for x in lst]
        print(f"  {mname:16s} MAE={ov.mean():.3f} ± {ov.std():.3f}  "
              f"best_ep={be}  (n={len(ov)})")
    print("\n注意：这是链路验证（少 epoch、少折），**不能作为结论**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
