"""命令行入口: RI（快速增强）事件预测。

结合了两个改进方向:
  1. 预测目标改为 RI 二分类（24h 内强度增幅 ≥ 阈值），而非数值回归；
  2. 预测窗口延长到 24h，让纯外推能力下降，给云图留出增值空间。

用法:
    # 三模型对比 + 5 折交叉验证（推荐）
    python scripts/train_ri.py --compare --cv 5

    # 单模型，指定 RI 阈值
    python scripts/train_ri.py --set satellite.ri.threshold_ms=10

    # 快速调试（CPU）
    python scripts/train_ri.py --compare --cv 2 --set device=cpu satellite.ri.epochs=5
"""
from __future__ import annotations

import argparse
import json
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
from src.data.ri_events import RIConfig, build_ri_samples, class_weights  # noqa: E402
from src.data.sat_intensity import SCALAR_FEATURES, fit_scalar_scaler  # noqa: E402
from src.models.ri_model import build_ri_model  # noqa: E402


# ---------------------------------------------------------------------------
# 评估指标（不依赖 sklearn，避免额外依赖）
# ---------------------------------------------------------------------------
def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC = 随机取一正一负样本，正样本得分更高的概率（Mann-Whitney U）。"""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # 处理并列：使用平均秩
    allv = np.concatenate([pos, neg])
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """平均精度（AP）：正样本稀有时比 AUC 更能反映实际可用性。"""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    recall = tp / ys.sum()
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def threshold_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    """给定阈值下的 POD/FAR/CSI。"""
    pred = score >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    pod = tp / (tp + fn) if (tp + fn) else float("nan")     # 命中率
    far = fp / (tp + fp) if (tp + fp) else float("nan")     # 空报率
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")  # 临界成功指数
    return {"thr": float(thr), "POD": pod, "FAR": far, "CSI": csi,
            "tp": tp, "fp": fp, "fn": fn}


def best_csi_threshold(y: np.ndarray, score: np.ndarray) -> dict:
    """在验证分数上扫描阈值，返回 CSI 最高的那个（部署时用验证集定阈值）。"""
    qs = np.quantile(score, np.linspace(0.5, 0.995, 60))
    best = None
    for thr in np.unique(qs):
        m = threshold_metrics(y, score, thr)
        if best is None or (m["CSI"] == m["CSI"] and m["CSI"] > best["CSI"]):
            best = m
    return best or {"thr": float("nan"), "POD": float("nan"), "FAR": float("nan"),
                    "CSI": float("nan"), "tp": 0, "fp": 0, "fn": 0}


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
class RIDataset(Dataset):
    def __init__(self, bundle: dict, idx: np.ndarray, scaler: dict | None,
                 with_img: bool, use_cnn: bool):
        self.s = bundle["s"][idx]
        self.y = bundle["y"][idx]
        self.delta = bundle["delta"][idx]
        self.with_img = with_img and "p" in bundle
        self.use_cnn = use_cnn
        self.p = bundle["p"][idx] if (self.with_img and use_cnn) else None
        self.pfeat = bundle["pfeat"][idx] if (self.with_img and not use_cnn) else None
        if scaler is not None:
            mean = np.asarray(scaler["mean"], np.float32)
            std = np.asarray(scaler["std"], np.float32)
            self.s = (self.s - mean) / std
            if self.pfeat is not None:
                fm = np.asarray(scaler["img_mean"], np.float32)
                fs = np.asarray(scaler["img_std"], np.float32)
                self.pfeat = (self.pfeat - fm) / fs
        if self.p is not None:
            self.p = self.p / 255.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict:
        item = {
            "s": torch.from_numpy(self.s[i].astype(np.float32)),
            "y": torch.tensor(float(self.y[i]), dtype=torch.float32),
        }
        # 显式检查属性，避免 None 掩盖问题
        if getattr(self, "p", None) is not None:
            item["p"] = torch.from_numpy(self.p[i].astype(np.float32))
        if getattr(self, "pfeat", None) is not None:
            item["pfeat"] = torch.from_numpy(self.pfeat[i].astype(np.float32))
        return item


def _resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if pref == "cuda":
            raise RuntimeError("要求 cuda 但当前不可用")
        return torch.device("cpu")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# 单模型训练
# ---------------------------------------------------------------------------
def run_training(cfg, bundle: dict, splits: dict, mname: str) -> dict:
    sc = cfg.satellite
    rc = sc.ri
    tr = rc.train          # 训练超参都在 satellite.ri.train 下
    device = _resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))

    use_cnn = mname == "cnn_transformer"
    with_img = mname in ("cnn_transformer", "feature_transformer")
    is_persist = mname == "persistence"

    scaler = fit_scalar_scaler(bundle["s"][splits["train"]], SCALAR_FEATURES)
    if "pfeat" in bundle:
        # 注意变量名不要用 tr（那是训练超参配置）
        tr_feat = bundle["pfeat"][splits["train"]].reshape(-1, bundle["pfeat"].shape[-1])
        m_ = np.nanmean(tr_feat, axis=0)
        s_ = np.nanstd(tr_feat, axis=0)
        scaler["img_mean"] = m_.tolist()
        scaler["img_std"] = np.where(s_ < 1e-6, 1.0, s_).tolist()

    datasets = {k: RIDataset(bundle, splits[k], scaler, with_img, use_cnn)
                for k in ("train", "val", "test")}
    bs = int(tr.batch_size)
    loaders = {k: DataLoader(v, batch_size=bs, shuffle=(k == "train"),
                             num_workers=int(tr.num_workers))
               for k, v in datasets.items()}

    ytr = bundle["y"][splits["train"]]
    pos_rate = ytr.mean()
    tag = {"cnn_transformer": "CNN+云图", "feature_transformer": "特征+云图",
           "gru": "纯时序GRU", "persistence": "持续法"}[mname]

    # 持续法：直接打分，不训练
    if is_persist:
        val_score = bundle["s"][splits["val"], -1, 0] - bundle["s"][splits["val"], -2, 0]
        test_score = bundle["s"][splits["test"], -1, 0] - bundle["s"][splits["test"], -2, 0]
        return _finish(tag, 0, bundle["y"][splits["val"]], val_score,
                       bundle["y"][splits["test"]], test_score, pos_rate, [], mname)

    patch_chans = int(bundle["p"].shape[2]) if "p" in bundle else 3
    img_dim = int(bundle["pfeat"].shape[-1]) if "pfeat" in bundle else 0
    model = build_ri_model(rc.model, bundle["s"].shape[-1], img_dim,
                           bundle["s"].shape[1], patch_chans,
                           name=mname).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'#' * 60}\n# {tag} 参数量={n_param:,} 正样本率={pos_rate:.1%}\n{'#' * 60}")

    w_neg, w_pos = class_weights(ytr)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr.lr),
                            weight_decay=float(tr.weight_decay))
    epochs = int(tr.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = bool(tr.amp) and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    pos_w = torch.tensor([w_pos], device=device)
    neg_w = torch.tensor([w_neg], device=device)

    def forward(b):
        s = b["s"].to(device)
        if use_cnn:
            return model(s, p=b["p"].to(device))
        if with_img:
            return model(s, pfeat=b["pfeat"].to(device))
        return model(s)

    @torch.no_grad()
    def score_all(loader):
        model.eval()
        sc_, yy = [], []
        for b in loader:
            sc_.append(torch.sigmoid(forward(b)).cpu().numpy())
            yy.append(b["y"].numpy())
        return np.concatenate(yy).astype(int), np.concatenate(sc_)

    best_val_auc, best_state, best_epoch, bad = -1.0, None, -1, 0
    history = []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        run, seen = 0.0, 0
        for b in loaders["train"]:
            y = b["y"].to(device)
            w = torch.where(y > 0.5, pos_w, neg_w)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logit = forward(b)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, y, weight=w)
            if amp_scaler is not None:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.grad_clip))
                amp_scaler.step(opt)
                amp_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.grad_clip))
                opt.step()
            run += loss.detach().item() * y.size(0)
            seen += y.size(0)
        sched.step()

        yv, sv = score_all(loaders["val"])
        auc = roc_auc(yv, sv)
        ap = pr_auc(yv, sv)
        history.append({"epoch": epoch, "train_loss": run / max(1, seen),
                        "val_auc": auc, "val_pr_auc": ap,
                        "seconds": round(time.time() - t0, 1)})
        print(f"[epoch {epoch:2d}] loss={run / max(1, seen):.4f} "
              f"val_AUC={auc:.4f} val_PR-AUC={ap:.4f} {history[-1]['seconds']}s")

        if auc > best_val_auc + 1e-5:
            best_val_auc, best_epoch, bad = auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tr.early_stop_patience):
                print(f"[早停] epoch {epoch}，最佳 epoch {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yv, sv = score_all(loaders["val"])
    yt, st = score_all(loaders["test"])
    return _finish(tag, n_param, yv, sv, yt, st, pos_rate, history, mname,
                   best_epoch=best_epoch)


def _finish(tag, n_param, yv, sv, yt, st, pos_rate, history, mname, best_epoch=0) -> dict:
    """在验证集上选 CSI 最优阈值，再应用到测试集。"""
    best_thr = best_csi_threshold(yv, sv)
    thr = best_thr["thr"]
    tm = threshold_metrics(yt, st, thr) if np.isfinite(thr) else best_csi_threshold(yt, st)
    res = {
        "tag": tag, "model": mname, "n_params": n_param,
        "pos_rate": float(pos_rate),
        "test_auc": roc_auc(yt, st),
        "test_pr_auc": pr_auc(yt, st),
        "test_threshold": float(thr),
        "test_POD": tm["POD"], "test_FAR": tm["FAR"], "test_CSI": tm["CSI"],
        "n_test": int(len(yt)), "n_test_pos": int(yt.sum()),
        "best_epoch": best_epoch, "history": history,
    }
    print(f"[{tag}] 测试集: AUC={res['test_auc']:.4f} PR-AUC={res['test_pr_auc']:.4f} "
          f"POD={tm['POD']:.3f} FAR={tm['FAR']:.3f} CSI={tm['CSI']:.3f}")
    return res


def kfold_by_storm(bundle: dict, n_folds: int, seed: int) -> list[dict]:
    names = bundle["name"]
    uniq = np.array(sorted(set(names.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)
    out = []
    for k in range(n_folds):
        test_names = set(folds[k].tolist())
        rest = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        # 验证集用于选 CSI 阈值。RI 正样本稀少（4-12%），15% 的验证集里
        # 可能只有几个正样本，阈值不稳定，因此提高到 25%。
        n_val = max(1, int(len(rest) * 0.25))
        val_names = set(rest[:n_val].tolist())
        train_names = set(rest[n_val:].tolist())
        mk = lambda st: np.where(np.isin(names, list(st)))[0]  # noqa: E731
        out.append({"train": mk(train_names), "val": mk(val_names), "test": mk(test_names)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="RI 快速增强事件预测")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--compare", action="store_true",
                    help="跑 cnn_transformer / feature_transformer / gru / persistence")
    ap.add_argument("--cv", type=int, default=0, help="K 折交叉验证折数")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    sc = cfg.satellite
    rc_cfg = sc.ri
    import pandas as pd

    scalars = pd.read_parquet(_data_path(cfg, cfg.data.processed_dir) / "sat_scalars.parquet")
    cache = _data_path(cfg, cfg.data.processed_dir) / patch_cache_name(cfg)
    if not cache.exists():
        print(f"[RI] 缺少云图缓存 {cache}，请先运行 build_sat_patches.py")
        return 1
    d = np.load(cache, allow_pickle=False)
    patches = {}
    for k, p in zip(d["keys"], d["patches"]):
        n, t = str(k).split("|", 1)
        patches[(n, pd.Timestamp(t))] = p
    print(f"[RI] 载入云图 {len(patches)} 帧")

    ri_cfg = RIConfig(
        in_len=int(rc_cfg.in_len), horizon=int(rc_cfg.horizon),
        interval_h=float(sc.interval_h),
        ri_threshold_ms=float(rc_cfg.threshold_ms),
        patch_size=int(sc.patch_size), use_patch=True,
    )
    bundle = build_ri_samples(scalars, ri_cfg, patches)

    models = (["cnn_transformer", "feature_transformer", "gru", "persistence"]
              if args.compare else [str(rc_cfg.model)])

    out_dir = ROOT / cfg.output.dir / "ri_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cv and args.cv > 1:
        folds = kfold_by_storm(bundle, args.cv, int(cfg.seed))
        per_fold = {}
        for fi, splits in enumerate(folds):
            print(f"\n{'=' * 70}\nCV fold {fi + 1}/{len(folds)}\n{'=' * 70}")
            per_fold[fi] = {m: run_training(cfg, bundle, splits, m) for m in models}
        _print_cv(per_fold, args.cv, models)
        tests_auc = paired_tests(per_fold, models, "test_auc")
        _print_paired(tests_auc)
        tests_pr = paired_tests(per_fold, models, "test_pr_auc")
        _print_paired(tests_pr)
        tests_csi = paired_tests(per_fold, models, "test_CSI")
        _print_paired(tests_csi)
        payload = {"n_folds": args.cv, "threshold_ms": ri_cfg.ri_threshold_ms,
                   "summary": _cv_summary(per_fold, models),
                   "paired_tests": {"auc": tests_auc, "pr_auc": tests_pr,
                                    "csi": tests_csi},
                   "per_fold": {str(k): {m: {
                       "test_auc": v[m]["test_auc"],
                       "test_pr_auc": v[m]["test_pr_auc"],
                       "test_POD": v[m]["test_POD"],
                       "test_FAR": v[m]["test_FAR"],
                       "test_CSI": v[m]["test_CSI"],
                   } for m in models} for k, v in per_fold.items()}}
        path = out_dir / f"cv{args.cv}_thr{ri_cfg.ri_threshold_ms:g}_summary.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nCV 结果已保存: {path}")
        return 0

    splits = kfold_by_storm(bundle, 5, int(cfg.seed))[0]
    results = {m: run_training(cfg, bundle, splits, m) for m in models}
    print("\n" + "=" * 70 + "\n结果\n" + "=" * 70)
    print(_table(results, models))
    path = out_dir / "summary.json"
    path.write_text(json.dumps({k: {kk: vv for kk, vv in v.items()}
                                for k, v in results.items()},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {path}")
    return 0


def paired_tests(per_fold: dict, models: list[str], metric: str = "test_auc") -> dict:
    """同一划分下各模型指标的配对比较。

    为什么用配对检验: 所有模型跑的是**同一套 K 折划分**，因此每折的难度是共同的。
    直接比较均值会低估检验效力；配对检验只关心"同一折下 A 是否稳定优于 B"，
    能消除折间难度差异，是评估两个模型差异的标准做法。

    同时给出参数检验(t)与非参数检验(Wilcoxon)，因为 K 折样本量小、指标分布未必正态。
    """
    from scipy import stats

    rank = {m: per_fold[0][m]["tag"] for m in models}
    out = {"metric": metric, "comparisons": []}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            va = np.array([per_fold[f][a][metric] for f in per_fold], dtype=float)
            vb = np.array([per_fold[f][b][metric] for f in per_fold], dtype=float)
            # 某些折的测试集可能没有正样本（RI 稀有），导致 AUC 无法计算。
            # 配对检验只在两者都有有限值的折上进行，否则整列会变 NaN。
            valid = np.isfinite(va) & np.isfinite(vb)
            n_dropped = int((~valid).sum())
            va, vb = va[valid], vb[valid]
            diff = va - vb
            n = len(diff)
            entry = {
                "model_a": rank[a], "model_b": rank[b],
                "mean_a": float(np.nanmean(va)) if n else float("nan"),
                "mean_b": float(np.nanmean(vb)) if n else float("nan"),
                "mean_diff": float(np.nanmean(diff)) if n else float("nan"),
                "n_folds": int(n),
                "n_dropped_folds": n_dropped,
                "a_wins": int((diff > 0).sum()),
            }
            # 配对 t 检验（双尾）
            if n > 1 and np.isfinite(diff).all():
                t_stat, p_t = stats.ttest_rel(va, vb)
                entry["t_stat"] = float(t_stat)
                entry["p_value_t"] = float(p_t)
                # Wilcoxon 符号秩（非参数，适合小样本）
                if np.any(diff != 0):
                    try:
                        w_stat, p_w = stats.wilcoxon(va, vb)
                        entry["wilcoxon_stat"] = float(w_stat)
                        entry["p_value_wilcoxon"] = float(p_w)
                    except Exception:
                        entry["p_value_wilcoxon"] = None
                else:
                    entry["p_value_wilcoxon"] = None
                # Cohen's d（配对版）：效应量，避免只看 p 值
                sd = float(np.std(diff, ddof=1))
                entry["cohen_d"] = float(np.mean(diff) / sd) if sd > 1e-9 else 0.0
            out["comparisons"].append(entry)
    return out


def _print_paired(tests: dict) -> None:
    print("\n" + "=" * 94)
    print(f"配对检验（指标: {tests['metric']}），配对 t 检验 + Wilcoxon 符号秩")
    print("=" * 94)
    print("| A vs B | A 均值 | B 均值 | 差值 | A 胜出折数 | t | p(t) | p(Wilcoxon) | Cohen's d |")
    print("|" + "---|" * 9)
    for c in tests["comparisons"]:
        p_t = c.get("p_value_t")
        p_w = c.get("p_value_wilcoxon")
        sig = ""
        if p_t is not None:
            sig = " ***" if p_t < 0.001 else (" **" if p_t < 0.01 else (" *" if p_t < 0.05 else ""))
        drop = c.get("n_dropped_folds", 0)
        drop_note = f" (-{drop}折)" if drop else ""
        print(f"| {c['model_a']} vs {c['model_b']} | {c['mean_a']:.4f} | {c['mean_b']:.4f} "
              f"| {c['mean_diff']:+.4f} | {c['a_wins']}/{c['n_folds']}{drop_note} "
              f"| {c.get('t_stat', float('nan')):+.3f} "
              f"| {p_t if p_t is not None else float('nan'):.4f}{sig} "
              f"| {p_w if p_w is not None else float('nan'):.4f} "
              f"| {c.get('cohen_d', float('nan')):+.2f} |")
    print("\n  p<0.05 视为显著（*）；Cohen's d: 0.2 小效应 / 0.5 中 / 0.8 大效应")
    print("  注意: K 折的 p 值有已知偏差（各折训练集重叠），应看作「证据强度」而非严格概率")


def _cv_summary(per_fold: dict, models: list[str]) -> dict:
    out = {}
    for m in models:
        def col(key):
            return np.array([per_fold[f][m][key] for f in per_fold], dtype=float)
        out[per_fold[0][m]["tag"]] = {
            "AUC_mean": float(np.nanmean(col("test_auc"))),
            "AUC_std": float(np.nanstd(col("test_auc"))),
            "PR_AUC_mean": float(np.nanmean(col("test_pr_auc"))),
            "PR_AUC_std": float(np.nanstd(col("test_pr_auc"))),
            "POD_mean": float(np.nanmean(col("test_POD"))),
            "FAR_mean": float(np.nanmean(col("test_FAR"))),
            "CSI_mean": float(np.nanmean(col("test_CSI"))),
            "CSI_std": float(np.nanstd(col("test_CSI"))),
            "AUC_per_fold": col("test_auc").tolist(),
        }
    return out


def _print_cv(per_fold: dict, n_folds: int, models: list[str]) -> None:
    print("\n" + "=" * 78)
    print(f"{n_folds} 折交叉验证汇总（RI 事件预测，正类稀少，重点看 AUC 与 PR-AUC）")
    print("=" * 78)
    print("| 模型 | AUC | PR-AUC | POD | FAR | CSI |")
    print("|" + "---|" * 6)
    for m in models:
        tag = per_fold[0][m]["tag"]
        a = np.array([per_fold[f][m]["test_auc"] for f in per_fold])
        p = np.array([per_fold[f][m]["test_pr_auc"] for f in per_fold])
        pod = np.array([per_fold[f][m]["test_POD"] for f in per_fold])
        far = np.array([per_fold[f][m]["test_FAR"] for f in per_fold])
        csi = np.array([per_fold[f][m]["test_CSI"] for f in per_fold])
        print(f"| {tag} | {np.nanmean(a):.4f}±{np.nanstd(a):.4f} "
              f"| {np.nanmean(p):.4f}±{np.nanstd(p):.4f} "
              f"| {np.nanmean(pod):.3f} | {np.nanmean(far):.3f} "
              f"| {np.nanmean(csi):.3f}±{np.nanstd(csi):.3f} |")


def _table(results: dict, models: list[str]) -> str:
    head = "| 模型 | AUC | PR-AUC | POD | FAR | CSI |"
    lines = [head, "|---|---|---|---|---|---|"]
    for m in models:
        r = results[m]
        lines.append(f"| {r['tag']} | {r['test_auc']:.4f} | {r['test_pr_auc']:.4f} "
                     f"| {r['test_POD']:.3f} | {r['test_FAR']:.3f} | {r['test_CSI']:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
