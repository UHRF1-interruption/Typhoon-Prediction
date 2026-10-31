"""命令行入口: 卫星云图强度预测训练与对比。

用法:
    # 训练融合模型（云图 + 时序）
    python scripts/train_intensity.py

    # 消融：不看云图，只用时序 GRU（验证云图是否真的带来增益）
    python scripts/train_intensity.py --set satellite.model.name=gru

    # 三个模型一起对比 + 出图
    python scripts/train_intensity.py --compare

    # 强制 CPU（沙箱内验证用）
    python scripts/train_intensity.py --set device=cpu train.epochs=5
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _data_path(cfg, value):
    """把配置路径解析成绝对路径，支持 TYPHOON_DATA_ROOT 挂载外部数据。"""
    from src.config import resolve_path
    return resolve_path(cfg, value, ROOT)

sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.fysatty import patch_cache_name  # noqa: E402
from src.data.sat_intensity import (  # noqa: E402
    SCALAR_FEATURES, SatConfig, build_intensity_samples, fit_scalar_scaler,
)
from src.models.intensity_transformer import build_intensity_model  # noqa: E402

HORIZON_H = 3.0     # 每步 3 小时


def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if pref == "cuda":
            raise RuntimeError("要求 cuda 但当前不可用")
        return torch.device("cpu")
    return torch.device("cuda")


def load_patches(cache_path: Path) -> dict:
    """读取缓存的 patch，返回 {(name, timestamp): (C,S,S)} 视图。

    视图共享同一个底层数组，**不复制**帧数据：64px 下全量约 800MB，
    若逐帧 list 化会再翻一倍。调用方不要修改返回值。
    （不用 mmap：npz 是压缩格式，mmap 读不了；且只读 mmap 会让后续
      np.stack 无法直接接管内存，反而更慢更卡。）
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"缺少云图缓存 {cache_path}，请先运行: "
            f".venv/bin/python scripts/build_sat_patches.py")
    import pandas as pd

    data = np.load(cache_path, allow_pickle=False)
    keys, patches = data["keys"], data["patches"]
    out = {}
    for i, k in enumerate(keys):
        name, ts = str(k).split("|", 1)
        out[(name, pd.Timestamp(ts))] = patches[i]      # 视图，共享同一底层数组
    print(f"[intensity] 载入云图 cache {len(out)} 帧, 形状 {patches.shape[1:]}")
    return out


class IntensityDataset(Dataset):
    def __init__(self, bundle: dict, idx: np.ndarray, scaler: dict | None):
        self.s = bundle["s"][idx]
        self.y = bundle["y"][idx]
        self.last = bundle["last_wind"][idx]
        # bundle["p"][j] 是第 j 个样本的帧下标（in_len 个），指向共享表 pframes。
        # 若在这里展开并整体标准化，会为每个 Dataset 复制约 N*in_len*3*S*S
        # 的 float32：64px 下每个 train/val/test 各约 5.4GB，三份同时存在会吃光内存。
        # 因此只保留引用，在 __getitem__ 里按需取帧并标准化。
        self.p = bundle.get("p")
        self.pframes = bundle.get("pframes")
        self.p_idx = idx if self.p is not None else None
        self.pfeat = bundle.get("pfeat")[idx] if "pfeat" in bundle else None
        if scaler is not None:
            mean = np.asarray(scaler["mean"], np.float32)
            std = np.asarray(scaler["std"], np.float32)
            self.s = (self.s - mean) / std
            # 云图特征量纲差异极大（亮温 ~230、面积占比 0-1），必须同样标准化，
            # 否则会与标量特征的尺度严重失配，模型学不动。
            if self.pfeat is not None:
                fm = np.asarray(scaler["img_mean"], np.float32)
                fs = np.asarray(scaler["img_std"], np.float32)
                self.pfeat = (self.pfeat - fm) / fs
        # 云图像素标准化参数：用**训练集实测亮温的均值/标准差**逐通道标准化。
        # 旧实现是 p / 255.0，但数据里是开尔文（实际范围约 101-340 K），
        # 除完得到 0.40-1.33、均值 0.89、标准差仅约 0.1：相当于把云图缩小 10 倍，
        # ReLU 网络对它几乎不敏感。scaler 里若没有像素统计量则退回除以 255。
        pm = scaler.get("pix_mean") if scaler else None
        ps = scaler.get("pix_std") if scaler else None
        if pm is not None and ps is not None:
            self._pix_mean = np.asarray(pm, np.float32)[:, None, None]
            self._pix_std = np.asarray(ps, np.float32)[:, None, None]
        else:
            self._pix_mean, self._pix_std = 0.0, 255.0

    def __len__(self) -> int:
        return len(self.y)

    def _patch(self, j: int) -> np.ndarray:
        """取第 j 个样本的 patch 栈并标准化 -> (in_len, C, S, S) float32。"""
        arr = self.pframes[self.p[self.p_idx[j]]].astype(np.float32)  # (in_len, C, S, S)
        return (arr - self._pix_mean) / self._pix_std

    def __getitem__(self, i: int) -> dict:
        item = {
            "s": torch.from_numpy(self.s[i].astype(np.float32)),
            "y": torch.from_numpy(self.y[i].astype(np.float32)),
            "last": torch.tensor(float(self.last[i]), dtype=torch.float32),
        }
        if self.p is not None:
            item["p"] = torch.from_numpy(self._patch(i))
        if self.pfeat is not None:
            item["pfeat"] = torch.from_numpy(self.pfeat[i].astype(np.float32))
        return item


def intensity_metrics(pred: np.ndarray, true: np.ndarray,
                      base_wind: np.ndarray | None = None) -> dict:
    """强度误差指标。pred/true 形状 (N, out_len)，单位 m/s。"""
    err = pred - true
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    # 强度倾向（相对"当前强度维持"的变化量）的预测能力。
    # 注意: 基准必须是**输入最后一刻的强度**，而不是预测序列的第一点，
    # 否则 3h 那一步两边都减自己，相关系数恒为 nan。
    base = base_wind.reshape(-1, 1) if base_wind is not None else true[:, [0]]
    d_true = true - base
    d_pred = pred - base
    corr = []
    for i in range(true.shape[1]):
        a, b = d_true[:, i], d_pred[:, i]
        corr.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan"))
    return {"mae_by_step": mae, "rmse_by_step": rmse,
            "trend_corr_by_step": np.array(corr),
            "mae_overall": float(np.abs(err).mean()),
            "horizon": (np.arange(1, true.shape[1] + 1) * HORIZON_H)}


def _forward(model, b, device):
    """按模型类型选择输入：CNN 版吃 p，特征版吃 pfeat，纯时序不吃。"""
    s = b["s"].to(device)
    if getattr(model, "use_image_features", False):
        return model(s, img_feat=b["pfeat"].to(device))
    p = b["p"].to(device) if "p" in b else None
    return model(s, p)


def run_training(cfg, bundle: dict, splits: dict, use_patch: bool) -> dict:
    sc = cfg.satellite
    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))

    scaler = fit_scalar_scaler(bundle["s"][splits["train"]], SCALAR_FEATURES)
    if "pfeat" in bundle:
        tr_feat = bundle["pfeat"][splits["train"]].reshape(-1, bundle["pfeat"].shape[-1])
        img_mean = np.nanmean(tr_feat, axis=0)
        img_std = np.nanstd(tr_feat, axis=0)
        img_std = np.where(img_std < 1e-6, 1.0, img_std)
        scaler["img_mean"] = img_mean.tolist()
        scaler["img_std"] = img_std.tolist()
    if "p" in bundle:
        # 云图像素标准化参数，同样**只用训练集**拟合（避免测试集信息泄漏）。
        # 形状 (C,)：逐通道在全部训练帧、全部像素上统计。
        # 训练帧可能被多个窗口共享，只取去重后的唯一帧，既省内存也避免重复计数。
        uniq = sorted({int(f) for j in splits["train"] for f in bundle["p"][j]})
        tr_pix = bundle["pframes"][uniq]
        pix_mean = np.nanmean(tr_pix, axis=(0, 2, 3))
        pix_std = np.nanstd(tr_pix, axis=(0, 2, 3))
        pix_std = np.where(pix_std < 1e-6, 1.0, pix_std)
        scaler["pix_mean"] = pix_mean.tolist()
        scaler["pix_std"] = pix_std.tolist()
        print(f"[pix] 训练集亮温 均值={np.round(pix_mean, 1).tolist()} "
              f"标准差={np.round(pix_std, 2).tolist()} (K，逐通道)")
    ds = {
        "train": IntensityDataset(bundle, splits["train"], scaler),
        "val": IntensityDataset(bundle, splits["val"], scaler),
        "test": IntensityDataset(bundle, splits["test"], scaler),
    }
    bs = int(sc.train.batch_size)
    dl = {k: DataLoader(v, batch_size=bs, shuffle=(k == "train"),
                        num_workers=int(sc.train.num_workers))
          for k, v in ds.items()}

    g0 = bundle["pframes"][bundle["p"][splits["train"][0]][0]] if "p" in bundle else None
    patch_size = int(g0.shape[-1]) if g0 is not None else int(sc.patch_size)
    patch_chans = int(g0.shape[0]) if g0 is not None else 3
    img_dim = int(bundle["pfeat"].shape[-1]) if "pfeat" in bundle else None
    model = build_intensity_model(sc.model, bundle["s"].shape[-1],
                                  bundle["s"].shape[1], bundle["y"].shape[1],
                                  use_patch and "p" in bundle, patch_size,
                                  patch_chans=patch_chans, img_dim=img_dim).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tag = f"{sc.model.name}{'+云图' if getattr(model, 'use_patch', False) else ''}"
    print(f"\n{'#' * 60}\n# {tag}  参数量={n_param:,}  设备={device}\n{'#' * 60}")
    print(f"样本数 train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])}")

    if n_param == 0:
        # 持续法：无需训练
        return _evaluate(model, dl, device, tag, n_param, best_epoch=0, history=[])

    opt = torch.optim.AdamW(model.parameters(), lr=float(sc.train.lr),
                            weight_decay=float(sc.train.weight_decay))
    epochs = int(sc.train.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = bool(sc.train.amp) and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    def predict(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for b in loader:
                out = _forward(model, b, device)
                last = b["last"].to(device).unsqueeze(1)
                preds.append((out + last).cpu().numpy())     # 残差 -> 绝对强度
                trues.append(b["y"].numpy())
        return np.concatenate(preds), np.concatenate(trues)

    best_val, best_epoch, bad = float("inf"), -1, 0
    history = []
    best_state = None
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        run, seen = 0.0, 0
        for b in dl["train"]:
            y = b["y"].to(device)
            last = b["last"].to(device).unsqueeze(1)
            n_batch = y.size(0)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = _forward(model, b, device)
                target = y - last                          # 残差目标
                loss = torch.nn.functional.smooth_l1_loss(out, target, beta=2.0)
            if scaler_amp is not None:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(sc.train.grad_clip))
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(sc.train.grad_clip))
                opt.step()
            run += loss.detach().item() * n_batch
            seen += n_batch
        sched.step()

        pv, tv = predict(dl["val"])
        val_mae = float(np.abs(pv - tv).mean())
        history.append({"epoch": epoch, "train_loss": run / max(1, seen),
                        "val_mae_ms": val_mae, "seconds": round(time.time() - t0, 1)})
        print(f"[epoch {epoch:2d}] train={run / max(1, seen):.3f} val_mae={val_mae:.2f} m/s "
              f"{history[-1]['seconds']}s")
        if val_mae < best_val - 1e-4:
            best_val, best_epoch, bad = val_mae, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(sc.train.early_stop_patience):
                print(f"[早停] epoch {epoch}，最佳 epoch {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _evaluate(model, dl, device, tag, n_param, best_epoch, history,
                     best_state=best_state)


def _evaluate(model, dl, device, tag, n_param, best_epoch, history,
              best_state=None) -> dict:
    model.eval()
    preds, trues, lasts = [], [], []
    with torch.no_grad():
        for b in dl["test"]:
            out = _forward(model, b, device)
            last = b["last"].to(device).unsqueeze(1)
            preds.append((out + last).cpu().numpy())
            trues.append(b["y"].numpy())
            lasts.append(b["last"].numpy())
    pred, true = np.concatenate(preds), np.concatenate(trues)
    last_wind = np.concatenate(lasts)
    m = intensity_metrics(pred, true, base_wind=last_wind)
    print(f"\n[{tag}] 测试集强度误差:")
    for h, mae, rmse, corr in zip(m["horizon"], m["mae_by_step"], m["rmse_by_step"],
                                 m["trend_corr_by_step"]):
        print(f"  {h:4.0f}h  MAE={mae:.2f} m/s  RMSE={rmse:.2f}  趋势相关={corr:+.3f}")
    print(f"  平均 MAE={m['mae_overall']:.2f} m/s  (最佳 epoch {best_epoch})")
    return {"tag": tag, "metrics": m, "n_params": n_param,
            "best_epoch": best_epoch, "history": history, "state": best_state,
            "pred": pred, "true": true,
            # 窗口末端的当前强度。分层验证要用它把样本按"实际强度变化幅度"
            # 分组（变化量 = true - last）。注意不能用 true[:,0] 代替——
            # 那是 3h 之后的值，不是窗口末端。
            "last": last_wind,
            # 逐时效距平相关（对持续法）也存下来，便于事后按时效分析。
            # 旧版只汇总成一个标量，时效结构就丢了。
            "corr_by_step": m["trend_corr_by_step"]}


def kfold_by_storm(bundle: dict, n_folds: int, seed: int) -> list[dict]:
    """按台风分层的 K 折交叉验证切分。

    每次留出 1/n_folds 的台风作为测试集，再从剩余台风里划出同样比例作为验证集。
    同一条台风（及其所有滑动窗口）只出现在一个集合中，避免数据泄漏。

    为什么用 K 折而不是单次切分: 台风总数只有几十个，单次划分的测试集只有
    6-8 个台风，误差波动很大；K 折可以用全部台风做测试，结论更可靠。
    """
    names = bundle["name"]
    uniq = np.array(sorted(set(names.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)
    out = []
    for k in range(n_folds):
        test_names = set(folds[k].tolist())
        rest = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        n_val = max(1, int(len(rest) * 0.15))
        val_names = set(rest[:n_val].tolist())
        train_names = set(rest[n_val:].tolist())
        mask = lambda st: np.where(np.isin(names, list(st)))[0]  # noqa: E731
        out.append({
            "fold": k,
            "train": mask(train_names),
            "val": mask(val_names),
            "test": mask(test_names),
            "n_train_storms": len(train_names),
            "n_val_storms": len(val_names),
            "n_test_storms": len(test_names),
        })
        print(f"[fold {k}] 台风 train={len(train_names)} val={len(val_names)} "
              f"test={len(test_names)} | 样本 train={len(out[-1]['train'])} "
              f"val={len(out[-1]['val'])} test={len(out[-1]['test'])}")
    return out


def split_by_storm(bundle: dict, test_ratio: float, seed: int) -> dict:
    """按台风名切分（同台风的所有样本只进一个集合，避免泄漏）。"""
    names = bundle["name"]
    uniq = np.array(sorted(set(names.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_ratio))
    n_val = max(1, int(len(uniq) * test_ratio))
    test_names = set(uniq[:n_test].tolist())
    val_names = set(uniq[n_test:n_test + n_val].tolist())
    mask = lambda st: np.where(np.isin(names, list(st)))[0]  # noqa: E731
    tr = np.where(~np.isin(names, list(test_names | val_names)))[0]
    print(f"[split] 台风切分: train={len(uniq) - n_test - n_val} 个台风, "
          f"val={n_val}, test={n_test}")
    return {"train": tr, "val": mask(val_names), "test": mask(test_names)}


def _flatten(results_by_fold: dict) -> dict:
    """把 {job: {tag: result}} 里每个 tag 的 pred/true 按 job 纵向拼接。

    各 job 的测试台风不同，直接拼接不会重复计数；但**不同划分之间有台风重叠**，
    做统计时要记住这一点（与 --vs 的口径说明一致）。
    """
    tags = list(next(iter(results_by_fold.values())).keys())
    out = {}
    for tag in tags:
        out[tag] = {
            "pred": np.concatenate([f[tag]["pred"] for f in results_by_fold.values()]),
            "true": np.concatenate([f[tag]["true"] for f in results_by_fold.values()]),
            "last": np.concatenate([f[tag]["last"] for f in results_by_fold.values()]),
        }
    return out


def _nan_agg(x: np.ndarray, fn) -> float:
    """忽略 nan 求均值/标准差；**全部为 nan 时返回 nan**（不填 0 冒充数值）。

    持续法的趋势相关恒为 nan（基准恒定、变化量无方差），若在这里
    用 np.nanmean 会抛 "Mean of empty slice" 警告并被静默填成 nan 之外的值。
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(fn(x[ok]))


def summarize_cv(results_by_fold: dict) -> dict:
    """把各 job 结果汇总成 均值±标准差（nan 安全）。"""
    out = {}
    tags = list(next(iter(results_by_fold.values())).keys())
    for tag in tags:
        maes = np.array([f[tag]["metrics"]["mae_by_step"] for f in results_by_fold.values()])
        overall = np.array([f[tag]["metrics"]["mae_overall"] for f in results_by_fold.values()])
        corr = np.array([
            [c if c == c else np.nan for c in f[tag]["metrics"]["trend_corr_by_step"]]
            for f in results_by_fold.values()
        ])
        # 每个 job 内跨时效求平均相关；整个 job 无有效值时为 nan
        corr_per_job = np.array([_nan_agg(corr[i], np.mean) for i in range(corr.shape[0])])
        out[tag] = {
            "mae_mean": maes.mean(axis=0),
            "mae_std": maes.std(axis=0),
            "overall_mean": float(overall.mean()),
            "overall_std": float(overall.std()),
            # 逐时效距平相关（nan 安全），供"环境场/云图的作用随时效如何变化"用
            "corr_by_step_mean": np.array([_nan_agg(corr[:, i], np.mean)
                                           for i in range(corr.shape[1])]),
            "corr_by_step_std": np.array([_nan_agg(corr[:, i], np.std)
                                          for i in range(corr.shape[1])]),
            "corr_mean": _nan_agg(corr, np.mean),
            "corr_std": _nan_agg(corr_per_job, np.std),
            "n_corr_jobs": int(np.isfinite(corr_per_job).sum()),
            "n_params": next(iter(results_by_fold.values()))[tag]["n_params"],
            "mae_per_fold": maes.tolist(),
        }
    return out


def cv_out_path(out_dir: Path, cv: int, overrides: list[str]) -> Path:
    """CV 汇总文件名带上**影响模型**的配置指纹，避免不同实验互相覆盖。

    只认 satellite.* / seed= 这类改变结果的覆盖项；device、run_name、batch_size
    等不影响数值的参数不参与，免得换个设备就生成新文件名。
    没有任何相关覆盖时仍用旧名 cv{N}_summary.json，保持向后兼容。
    """
    key = sorted(o for o in overrides
                 if (o.startswith("satellite.") and not o.startswith("satellite.run_name"))
                 or o.startswith("seed="))
    if not key:
        return out_dir / f"cv{cv}_summary.json"
    import hashlib
    joined = "|".join(key)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^0-9A-Za-z]+", "-", joined).strip("-")[:60]
    return out_dir / f"cv{cv}_{slug}_{digest}_summary.json"


def paired_vs(results_by_fold: dict, vs: str = "gru") -> dict:
    """对照模型做配对比较：每个 job 是同一套切分下的两种模型，逐 job 取差值。

    每个 job 的 mae_overall 已是该 job 整个测试集的样本均值，配对已吸收
    "台风难度差异"这一主要噪声源。但同一划分 seed 内的 K 折测试集互斥，
    不同划分 seed 之间台风会重叠，故 p 值偏乐观，必须结合胜率看。
    """
    tags = list(next(iter(results_by_fold.values())).keys())
    if vs not in tags:
        return {"error": f"对照模型 {vs} 不在 {tags}"}
    out = {"baseline": vs, "n_jobs": len(results_by_fold)}
    for tag in tags:
        if tag == vs:
            continue
        d = np.array([f[tag]["metrics"]["mae_overall"] - f[vs]["metrics"]["mae_overall"]
                      for f in results_by_fold.values()])
        n = len(d)
        mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
        t = mean / (sd / np.sqrt(n)) if sd > 0 else float("nan")
        entry = {"delta": mean, "sd": sd, "wins": int((d < 0).sum()),
                 "win_rate": float((d < 0).mean()), "n": n, "t": float(t)}
        try:
            from scipy import stats  # 可选依赖
            entry["p"] = float(stats.ttest_1samp(d, 0.0).pvalue)
        except Exception:
            entry["p"] = float("nan")
        out[tag] = entry
    return out


def print_paired(res: dict) -> None:
    if "error" in res:
        print(f"[paired] {res['error']}  ← 数值可疑！")
        return
    print(f"\n配对比较（对照 = {res['baseline']}，{res['n_jobs']} job，负值=更好）")
    print("| 模型 | ΔMAE | 胜率 | t | p |")
    print("|---|---|---|---|---|")
    for tag, e in res.items():
        if tag in ("baseline", "n_jobs") or not isinstance(e, dict):
            continue
        p = "nan" if e["p"] != e["p"] else (f"{e['p']:.2g}" if e["p"] >= 1e-4 else "<1e-4")
        print(f"| {tag} | {e['delta']:+.3f} | {e['wins']}/{e['n']} "
              f"({e['win_rate']:.0%}) | {e['t']:+.2f} | {p} |")


def _fmt_pm(mean: float, std: float, sign: str = "") -> str:
    """均值±标准差；无有效值时显示 n/a，绝不显示 +0.000 之类的假数。"""
    if not (np.isfinite(mean) and np.isfinite(std)):
        return "n/a"
    return f"{mean:{sign}.3f}±{std:.3f}" if sign else f"{mean:.2f}±{std:.2f}"


def print_cv_table(cv: dict, horizon: np.ndarray) -> None:
    head = "| 模型 | " + " | ".join(f"{int(h)}h" for h in horizon) + " | 平均 | 趋势相关 |"
    print(head)
    print("|" + "---|" * (len(horizon) + 3))
    for tag, v in cv.items():
        cells = " | ".join(f"{m:.2f}±{s:.2f}"
                           for m, s in zip(v["mae_mean"], v["mae_std"]))
        corr = _fmt_pm(v["corr_mean"], v["corr_std"], "+")
        print(f"| {tag} | {cells} | {v['overall_mean']:.2f}±{v['overall_std']:.2f} "
              f"| {corr} |")


def main() -> int:
    ap = argparse.ArgumentParser(description="卫星云图强度预测")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--compare", action="store_true",
                    help="同时跑 fusion_transformer / gru / persistence 三个模型")
    ap.add_argument("--cv", type=int, default=0,
                    help="K 折交叉验证的折数（如 5）。0 表示单次按台风切分")
    ap.add_argument("--fold-seeds", nargs="*", type=int, default=[],
                    help="用多个折划分 seed 重复 K 折，检验结论对划分的稳健性。"
                         "路径实验证明单次划分的结论可能不成立")
    ap.add_argument("--vs", default="gru",
                    help="CV 模式下与哪个模型做配对比较（胜率 + 配对 t 检验）。"
                         "设为空字符串跳过")
    ap.add_argument("--terrain", default=None,
                    help="地形特征 parquet（如 env_terrain.parquet）。"
                         "提供后会把 7 维地形特征加入标量输入")
    ap.add_argument("--terrain-fill", action="store_true",
                    help="对地形特征做短间隔前向填充（提升覆盖率）")
    ap.add_argument("--warmcore", action="store_true",
                    help="从红外云图反演暖心结构特征并加入标量输入（10 维）")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    sc = cfg.satellite
    import pandas as pd

    scalars_path = _data_path(cfg, cfg.data.processed_dir) / "sat_scalars.parquet"
    if not scalars_path.exists():
        print(f"[intensity] 缺少 {scalars_path}，请先生成标量表")
        return 1
    scalars = pd.read_parquet(scalars_path)

    cache = _data_path(cfg, cfg.data.processed_dir) / patch_cache_name(cfg)
    patches = load_patches(cache)

    use_terrain = bool(args.terrain)
    if use_terrain:
        from src.data.terrain_align import align_terrain_to_satellites, forward_fill_terrain
        terr_path = Path(args.terrain)
        if not terr_path.exists():
            print(f"[错误] 找不到地形文件 {terr_path}")
            return 1
        terrain = pd.read_parquet(terr_path)
        print(f"[terrain] 载入 {len(terrain)} 行地形特征")
        scalars = align_terrain_to_satellites(scalars, terrain)
        if args.terrain_fill:
            scalars = forward_fill_terrain(scalars)
            from src.data.sat_intensity import TERRAIN_FEATURES
            ok = scalars[TERRAIN_FEATURES].notna().all(axis=1).mean()
            print(f"[terrain] 前向填充后完整率 {ok:.1%}")

    use_warmcore = bool(args.warmcore)
    if use_warmcore:
        from src.data.warm_core import WARMCORE_FEATURES, compute_warmcore_features
        print(f"[warmcore] 从 {len(patches)} 帧红外云图反演 {len(WARMCORE_FEATURES)} 维暖心特征")
        wc_rows = []
        for (nm, tt), pch in patches.items():
            wc_rows.append({"name": nm, "time": tt,
                            **dict(zip(WARMCORE_FEATURES, compute_warmcore_features(pch)))})
        wc = pd.DataFrame(wc_rows)
        scalars = scalars.merge(wc, on=["name", "time"], how="left")
        n_ok = scalars[WARMCORE_FEATURES].notna().all(axis=1).mean()
        print(f"[warmcore] 合并后完整率 {n_ok:.1%}")

    sat_cfg = SatConfig(in_len=int(sc.in_len), out_len=int(sc.out_len),
                        interval_h=float(sc.interval_h),
                        patch_size=int(sc.patch_size), use_patch=True,
                        use_terrain=use_terrain, use_warmcore=use_warmcore)
    bundle = build_intensity_samples(scalars, sat_cfg, patches)
    print(f"[intensity] 样本 {len(bundle['y'])}, 台风 {len(set(bundle['name']))}")

    if args.compare:
        models = ["fusion_transformer", "feature_transformer", "gru", "persistence"]
    else:
        models = [str(sc.model.name)]

    configs = {}
    for mname in models:
        sub = load_config(args.config, args.set + [f"satellite.model.name={mname}"])
        use_patch = mname in ("fusion_transformer", "feature_transformer")
        configs[mname] = (sub, use_patch)

    out_dir = ROOT / cfg.output.dir / sc.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = np.arange(1, int(sc.out_len) + 1) * float(sc.interval_h)

    if args.cv and args.cv > 1:
        fseeds = args.fold_seeds or [int(cfg.seed)]
        results_by_fold = {}
        # 每个 job 的来源（划分 seed + 折号），per_fold 里必须带上，
        # 否则多种划分混在一起看不出方差来自"折"还是"划分"。
        job_meta = {}
        fi = 0
        for fs in fseeds:
            folds = kfold_by_storm(bundle, args.cv, int(fs))
            for k, splits in enumerate(folds):
                print(f"\n{'=' * 70}\n划分seed={fs} fold {k + 1}/{len(folds)}\n{'=' * 70}")
                res = {}
                for mname, (sub_cfg, use_patch) in configs.items():
                    sub_bundle = (bundle if use_patch else
                                  {k: v for k, v in bundle.items() if k not in ("p", "pfeat")})
                    r = run_training(sub_cfg, sub_bundle, splits, use_patch)
                    res[r["tag"]] = r
                results_by_fold[fi] = res
                job_meta[str(fi)] = {
                    "fold_seed": int(fs),
                    "fold": k,
                    "n_top_tr": int(len(set(bundle["name"][splits["train"]].tolist()))),
                    "n_top_val": int(len(set(bundle["name"][splits["val"]].tolist()))),
                    "n_top_test": int(len(set(bundle["name"][splits["test"]].tolist()))),
                }
                fi += 1
        print(f"\n[folds] 共 {len(results_by_fold)} 个 job "
              f"({args.cv} 折 × {len(fseeds)} 种划分 {fseeds})")

        cv = summarize_cv(results_by_fold)
        print("\n" + "=" * 70)
        print(f"{args.cv} 折交叉验证汇总（测试集，MAE 单位 m/s）"
              f"  [n={len(results_by_fold)} job = {args.cv} 折 × {len(fseeds)} 种划分 {fseeds}]")
        print("=" * 70)
        print_cv_table(cv, horizon)
        print("\n列顺序: ± 前 = 各 job 均值；± 后 = 各 job 标准差（非均值标准误）")
        print("       两种划分之间台风重叠，job 之间不独立，勿据此下结论；"
              "胜率/配对检验见 --vs")

        payload = {
            # n_folds 只是折数；真正决定统计量的是 n_jobs。
            "n_folds": args.cv,
            "n_fold_seeds": len(fseeds),
            "fold_seeds": [int(s) for s in fseeds],
            "n_jobs": len(results_by_fold),
            "metric": "MAE m/s, test set, 3/6/9/12h",
            # 口径声明：汇总用各 job 均值，std 为跨 job 标准差（ddof=0），
            # 且在同一个配置内所有 job 共用模型 seed（见 config.seed），
            # 因此 std 只反映"折/划分间的数据差异"，不含训练随机性。
            "aggregation": {
                "mean": "mean over jobs",
                "std": "std over jobs (ddof=0)",
                "model_seed": int(cfg.seed),
                "note": "job 间不独立（台风在不同划分间重叠），std 不是均值标准误",
            },
            "summary": {tag: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in d.items()}
                        for tag, d in cv.items()},
            "per_fold": {
                str(fi): {
                    "fold_seed": job_meta[str(fi)]["fold_seed"],
                    "fold": job_meta[str(fi)]["fold"],
                    "n_storms": {k: job_meta[str(fi)][f"n_top_{k}"]
                                 for k in ("tr", "val", "test")},
                    **{tag: {"mae_by_step": r["metrics"]["mae_by_step"].tolist(),
                             "overall": r["metrics"]["mae_overall"],
                             "best_epoch": r["best_epoch"]}
                       for tag, r in res.items()},
                }
                for fi, res in results_by_fold.items()
            },
        }
        if args.vs:
            payload["paired"] = paired_vs(results_by_fold, vs=args.vs)
            print_paired(payload["paired"])
        path = cv_out_path(out_dir, args.cv, args.set)
        payload["output"] = str(path.relative_to(ROOT))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nCV 结果已保存: {path}")

        # 同时落一份逐样本预测（npz）。JSON 存这些会让文件膨胀到几十 MB，
        # 而逐样本预测是做"分层验证"（如按强度变化幅度分组）的前提。
        pred_path = path.with_suffix(".npz")
        flat = _flatten(results_by_fold)
        arrays = {}
        for tag, r in flat.items():
            arrays[f"{tag}__pred"] = r["pred"].astype(np.float32)
            arrays[f"{tag}__true"] = r["true"].astype(np.float32)
            arrays[f"{tag}__last"] = r["last"].astype(np.float32)
        # 记录每个 job 的行区间：**不同划分之间有台风重叠**，分层统计必须
        # 能按 job 分别算，否则同一样本被算了 3 次（种子 42/0/7 各一次）。
        offsets = np.cumsum([0] + [len(f[list(f.keys())[0]]["pred"])
                                   for f in results_by_fold.values()])
        arrays["job_offsets"] = offsets.astype(np.int64)
        np.savez_compressed(pred_path, **arrays)
        print(f"逐样本预测已保存: {pred_path}")
        return 0

    # 单次切分模式
    splits = split_by_storm(bundle, float(sc.test_ratio), int(cfg.seed))
    results = {}
    for mname, (sub_cfg, use_patch) in configs.items():
        sub_bundle = (bundle if use_patch else
                      {k: v for k, v in bundle.items() if k not in ("p", "pfeat")})
        r = run_training(sub_cfg, sub_bundle, splits, use_patch)
        results[r["tag"]] = r

    print("\n" + "=" * 70)
    print("强度预测对比（测试集，MAE 单位 m/s，越小越好）")
    print("=" * 70)
    print("| 模型 | " + " | ".join(f"{int(h)}h" for h in horizon) + " | 平均 | 趋势相关 |")
    print("|" + "---|" * (len(horizon) + 3))
    for tag, r in results.items():
        m = r["metrics"]
        cells = " | ".join(f"{v:.2f}" for v in m["mae_by_step"])
        corr = _nan_agg(m["trend_corr_by_step"], np.mean)
        print(f"| {tag} | {cells} | {m['mae_overall']:.2f} | "
              f"{'n/a' if not np.isfinite(corr) else f'{corr:+.3f}'} |")

    summary = {tag: {"mae_by_step": r["metrics"]["mae_by_step"].tolist(),
                     "rmse_by_step": r["metrics"]["rmse_by_step"].tolist(),
                     "trend_corr": r["metrics"]["trend_corr_by_step"].tolist(),
                     "mae_overall": r["metrics"]["mae_overall"],
                     "n_params": r["n_params"], "best_epoch": r["best_epoch"],
                     "history": r["history"]}
               for tag, r in results.items()}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
