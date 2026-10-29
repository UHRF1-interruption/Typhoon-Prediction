"""命令行入口: 环境场对路径预测的增益（消融实验）。

对比三种配置，其余条件完全相同:
    baseline   仅轨迹特征（现有 8 维）
    +env_phys  轨迹特征 + 环境场
    env_only   仅环境场（检验环境场本身是否含路径信息）

用同一套按年份划分与多个随机种子，统计均值±标准差与配对检验，
避免单次运行的偶然性。

用法:
    # 用模拟环境场（无凭据，验证流程）
    python scripts/ablate_environment.py --env data/processed/env_synthetic.parquet

    # 用真实 ERA5
    python scripts/ablate_environment.py --env data/processed/env_era5.parquet
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.dataset import split_by_year  # noqa: E402
from src.data.environment import ENV_FEATURES  # noqa: E402
from src.data.features import build_samples, fit_feature_scaler  # noqa: E402
from src.training.losses import format_metrics_table  # noqa: E402
from src.training.trainer import train  # noqa: E402

def _align_timezones(tracks: pd.DataFrame, env: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把两张表的 time 列统一成「不带时区的 UTC 纳秒时间戳」。

    IBTrACS 读出来带 UTC 时区，环境场通常不带；直接 merge 会报
    "merging on datetime64[us, UTC] and datetime64[us]"。
    """
    tracks = tracks.copy()
    env = env.copy()
    for df in (tracks, env):
        ts = pd.to_datetime(df["time"], utc=True)
        df["time"] = ts.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    return tracks, env


BASE_TRACK_FEATURES = ["dlat", "dlon", "dist_prev", "sin_heading", "cos_heading",
                       "wind_ms", "dpres_hpa", "lat_norm"]


def split_by_storm_random(bundle: dict, test_ratio: float = 0.2,
                          val_ratio: float = 0.2, seed: int = 0) -> dict:
    """按台风随机划分（同台风的所有窗口只进一个集合）。

    为什么不用 split_by_year: 环境场往往只覆盖几个月，裁剪后所有样本都在
    同一年，按年份切分会得到空的验证/测试集。按台风切分在小样本下更合理，
    也避免了同一台风的相邻窗口跨集合造成的信息泄漏。
    """
    names = bundle["sid"]
    uniq = np.array(sorted(set(names.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
    test_names = set(uniq[:n_test].tolist())
    val_names = set(uniq[n_test:n_test + n_val].tolist())
    mask = lambda st: np.where(np.isin(names, list(st)))[0]  # noqa: E731
    out = {
        "train": np.where(~np.isin(names, list(test_names | val_names)))[0],
        "val": mask(val_names),
        "test": mask(test_names),
    }
    print(f"[split] 台风 train={n - n_test - n_val} val={n_val} test={n_test} | "
          f"样本 train={len(out['train'])} val={len(out['val'])} test={len(out['test'])}")
    return out


# 样本构造结果缓存：同一种特征组合只算一次（与 seed 无关，seed 只影响切分）
_BUNDLE_CACHE: dict = {}


def kfold_by_storm(bundle: dict, n_folds: int = 5, inner_val_ratio: float = 0.25,
                   seed: int = 0) -> list[dict]:
    """按台风的 K 折划分（每折轮流当测试集）。

    为什么 K 折比"随机抽 20% 当测试集"更好：
      * 每个台风恰好当一次测试集，避免某次划分恰好抽到难台风导致方差爆炸
      * 全部样本都参与测试，统计功效显著提高
      * 折间结果的离散度更小，配对检验更可靠

    做法：
      1. 把台风随机分成 K 组
      2. 第 i 折的测试集 = 第 i 组台风
      3. 剩余台风中再随机划 inner_val_ratio 比例做验证集（用于早停/选阈值）
      4. 同台风的全部窗口只进一个集合，无泄漏
    """
    names = bundle["sid"]
    uniq = np.array(sorted(set(names.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)
    mask = lambda st: np.where(np.isin(names, list(st)))[0]  # noqa: E731

    out = []
    for k in range(n_folds):
        test_names = set(folds[k].tolist())
        rest = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        n_val = max(1, int(round(len(rest) * inner_val_ratio))) if len(rest) > 2 else 0
        val_names = set(rest[:n_val].tolist())
        train_names = set(rest[n_val:].tolist())
        idx = {"train": mask(train_names), "val": mask(val_names), "test": mask(test_names)}
        print(f"[fold {k}] 台风 train={len(train_names)} val={len(val_names)} "
              f"test={len(test_names)} | 样本 "
              f"train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}")
        out.append(idx)
    return out


def build_bundle(tracks: pd.DataFrame, env: pd.DataFrame, use_track: bool,
                 use_env: bool, cfg, seed: int) -> tuple[dict, list[str]]:
    """构造指定特征组合的样本。环境场按 (sid, time) 对齐到轨迹。

    结果会缓存：同一特征组合在多个 seed 下复用，避免重复的滑窗构造
    （build_samples 在 5000 条轨迹上约需数秒，15 次训练能省下可观时间）。
    注意缓存键不含 seed —— seed 只影响后续的划分，不影响样本本身。
    """
    key = (use_track, use_env, int(cfg.data.in_len), int(cfg.data.out_len),
           float(cfg.data.interval_hours), len(tracks), len(env))
    if key in _BUNDLE_CACHE:
        bundle, feats = _BUNDLE_CACHE[key]
        # 划分依赖 seed，必须复制后重新划分
        b = dict(bundle)
        b["_splits"] = split_by_storm_random(b, seed=seed)
        print(f"[cache] 复用已构造的样本（特征 {len(feats)} 维）")
        return b, feats

    df = tracks.copy()
    feats: list[str] = []
    if use_track:
        feats += BASE_TRACK_FEATURES
    if use_env:
        env_cols = [c for c in ENV_FEATURES if c in env.columns]
        if not env_cols:
            raise ValueError(f"环境数据里没有可用特征，实际列: {list(env.columns)}")
        print(f"[bundle] 合并环境场 {len(env_cols)} 个特征 ...", flush=True)
        merged = pd.merge(df, env[["sid", "time"] + env_cols], on=["sid", "time"], how="left")
        missing = float(merged[env_cols].isna().all(axis=1).mean())
        print(f"[bundle] 环境缺失率 {missing:.1%}", flush=True)
        if missing > 0.5:
            raise ValueError(f"环境场对齐失败：{missing:.0%} 的轨迹点没有环境数据")
        df = merged
        feats += env_cols

    print(f"[bundle] 构造滑窗样本（特征 {len(feats)} 维）...", flush=True)
    bundle = build_samples(
        df, in_len=int(cfg.data.in_len), out_len=int(cfg.data.out_len),
        interval_hours=float(cfg.data.interval_hours), feature_names=feats,
    )
    # 按台风随机划分（环境场覆盖短、样本少时比按年份更稳）
    splits = split_by_storm_random(bundle, seed=seed)
    # 标准化参数只用训练集
    tr = np.nan_to_num(bundle["x"][splits["train"]], nan=0.0, posinf=0.0, neginf=0.0)
    spec = fit_feature_scaler(tr)
    spec.names = feats
    bundle["x"] = np.nan_to_num(bundle["x"], nan=0.0, posinf=0.0, neginf=0.0)
    bundle["_splits"] = splits
    _BUNDLE_CACHE[key] = ({k: v for k, v in bundle.items() if k != "_splits"}, feats)
    return bundle, feats


def run_one(cfg, tracks, env, use_track: bool, use_env: bool, tag: str,
            seed: int, overrides: list[str] | None = None,
            fixed_splits: dict | None = None,
            save_pred_for: str | None = None) -> dict:
    """训练一次并返回测试指标。

    注意: 必须沿用调用方已解析的 ovverides（如 train.epochs），
    不能重新加载默认配置，否则命令行 --set 会被静默丢弃。

    fixed_splits: 传入时直接使用给定划分（K 折模式下用），不再按 seed 随机划分。
    """
    cfg = load_config(str(ROOT / "configs" / "default.yaml"), overrides or [])
    cfg.seed = seed
    cfg.output.run_name = f"env_ablation_{tag}_s{seed}"
    bundle, feats = build_bundle(tracks, env, use_track, use_env, cfg, seed)
    splits = bundle.pop("_splits")
    if fixed_splits is not None:
        splits = fixed_splits
    tr = np.nan_to_num(bundle["x"][splits["train"]], nan=0.0, posinf=0.0, neginf=0.0)
    spec = fit_feature_scaler(tr)
    spec.names = feats

    from src.data.dataset import TrackWindowDataset  # noqa: F401
    result = train(cfg, bundle, spec, splits)
    m = result["test_metrics"]
    out = {
        "tag": tag, "seed": seed, "n_features": len(feats),
        "feature_names": feats,
        "mae_by_step": m["mae_by_step"].tolist(),
        "mae_overall": m["mae_overall"],
        "best_epoch": result["best_epoch"],
    }
    # 保存逐样本预测，供分层分析使用
    if save_pred_for is not None:
        pred = result.get("pred")
        # predict() 返回 dict: pred_abs / true_abs / last_abs / year，均为绝对坐标
        if isinstance(pred, dict) and "pred_abs" in pred:
            np.savez_compressed(
                save_pred_for,
                pred_abs=np.asarray(pred["pred_abs"], dtype=np.float32),
                true_abs=np.asarray(pred["true_abs"], dtype=np.float32),
                last_abs=np.asarray(pred["last_abs"], dtype=np.float32),
                year=np.asarray(pred["year"]),
                n_features=np.array([len(feats)]),
            )
            out["pred_path"] = str(save_pred_for)
    return out


def paired_tests(results: dict[str, list[dict]], a: str, b: str) -> dict:
    """对同一随机种子的两组结果做配对检验。"""
    from scipy import stats
    va = np.array([r["mae_overall"] for r in results[a]])
    vb = np.array([r["mae_overall"] for r in results[b]])
    diff = va - vb
    out = {"a": a, "b": b, "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
           "mean_diff": float(diff.mean()), "n": len(diff),
           "a_better_folds": int((diff < 0).sum())}
    if len(diff) > 1:
        t, p = stats.ttest_rel(va, vb)
        out["p_value"] = float(p)
        sd = diff.std(ddof=1)
        out["cohen_d"] = float(diff.mean() / sd) if sd > 1e-9 else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="环境场对路径预测的消融实验")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--env", required=True, help="环境场 parquet 路径")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--restrict-to-env", action="store_true", default=True,
                    help="只保留环境场覆盖时段内的轨迹点（默认开启，保证公平对比）")
    ap.add_argument("--full-tracks", dest="restrict_to_env", action="store_false",
                    help="使用全部轨迹（环境场会大量缺失，仅供诊断）")
    ap.add_argument("--folds", type=int, default=0,
                    help="K 折交叉验证的折数。>0 时忽略 --seeds，依次跑每一折。"
                         "推荐 5，方差远小于随机划分")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tracks = pd.read_parquet(_data_path(cfg, cfg.data.raw_dir) / f"tracks_{cfg.data.basin}_clean.parquet")
    env = pd.read_parquet(ROOT / args.env)
    tracks, env = _align_timezones(tracks, env)
    print(f"[ablate] 轨迹 {len(tracks)} 条 / 环境 {len(env)} 行")

    if args.restrict_to_env and len(env):
        # 三组配置必须用同一批轨迹，否则结果不可比。
        # 这里裁剪到环境场实际覆盖的时段与台风。
        lo, hi = env["time"].min(), env["time"].max()
        env_sids = set(env["sid"].unique())
        before_t, before_s = len(tracks), tracks["sid"].nunique()
        tracks = tracks[(tracks["time"] >= lo) & (tracks["time"] <= hi)
                        & tracks["sid"].isin(env_sids)].reset_index(drop=True)
        print(f"[ablate] 裁剪到环境场覆盖范围 {lo} ~ {hi}")
        print(f"[ablate] 轨迹 {before_t} -> {len(tracks)} 条, "
              f"台风 {before_s} -> {tracks['sid'].nunique()} 个")
        if tracks.empty:
            print("[错误] 裁剪后没有轨迹点，检查环境场与轨迹的时间/台风是否匹配")
            return 1

    configs = {
        "baseline(仅轨迹)": dict(use_track=True, use_env=False),
        "+环境场": dict(use_track=True, use_env=True),
        "仅环境场": dict(use_track=False, use_env=True),
    }

    results: dict[str, list[dict]] = {k: [] for k in configs}
    if args.folds and args.folds > 1:
        # ---- K 折交叉验证模式 ----
        print(f"\n[ablate] K 折交叉验证: {args.folds} 折")
        # 先用第一个配置构造一次样本以取得台风列表（三配置台风相同）
        first_cfg = next(iter(configs.values()))
        base_bundle, _ = build_bundle(tracks, env, seed=0, cfg=cfg, **first_cfg)
        folds = kfold_by_storm(base_bundle, n_folds=args.folds, seed=int(cfg.seed))
        for k, fold_splits in enumerate(folds):
            for tag, kw in configs.items():
                print(f"\n{'=' * 66}\n配置: {tag}  fold {k + 1}/{args.folds}\n{'=' * 66}")
                r = run_one(cfg, tracks, env, tag=tag, seed=k,
                            overrides=list(args.set), fixed_splits=fold_splits, **kw)
                results[tag].append(r)
                print(f"[{tag}] fold={k} 平均误差={r['mae_overall']:.1f} km")
        n_reps = args.folds
        rep_name = f"{args.folds} 折交叉验证"
    else:
        # ---- 随机划分多 seed 模式 ----
        for seed in args.seeds:
            for tag, kw in configs.items():
                print(f"\n{'=' * 66}\n配置: {tag}  seed={seed}\n{'=' * 66}")
                r = run_one(cfg, tracks, env, tag=tag, seed=seed,
                            overrides=list(args.set), **kw)
                results[tag].append(r)
                print(f"[{tag}] seed={seed} 平均误差={r['mae_overall']:.1f} km "
                      f"(特征 {r['n_features']} 维)")
        n_reps = len(args.seeds)
        rep_name = f"{len(args.seeds)} 个随机种子"

    # 汇总
    print("\n" + "=" * 78)
    print(f"消融结果（{rep_name}，测试集路径误差 km）")
    print("=" * 78)
    print("| 配置 | 特征数 | 平均误差 | 标准差 | 6h | 12h | 18h | 24h |")
    print("|" + "---|" * 8)
    for tag, rs in results.items():
        overall = np.array([r["mae_overall"] for r in rs])
        steps = np.array([r["mae_by_step"] for r in rs]).mean(axis=0)
        cells = " | ".join(f"{v:.1f}" for v in steps)
        print(f"| {tag} | {rs[0]['n_features']} | {overall.mean():.1f} "
              f"| {overall.std():.1f} | {cells} |")

    print("\n配对检验（同一随机种子，负值表示 A 更好）:")
    pairs = [("+环境场", "baseline(仅轨迹)"), ("+环境场", "仅环境场")]
    tests = []
    for a, b in pairs:
        if a in results and b in results:
            t = paired_tests(results, a, b)
            tests.append(t)
            p = t.get("p_value", float("nan"))
            print(f"  {a} vs {b}: 差值 {t['mean_diff']:+.1f} km, "
                  f"A 更好 {t['a_better_folds']}/{t['n']}, "
                  f"p={p:.4f}, d={t.get('cohen_d', float('nan')):+.2f}")

    # 输出目录包含 run_name，避免并行跑多 seed 时互相覆盖
    run_tag = str(cfg.output.run_name) if cfg.output.run_name else "env_ablation"
    out_dir = ROOT / cfg.output.dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"env_source": args.env, "seeds": args.seeds,
               "results": results, "paired_tests": tests}
    (out_dir / "ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_dir / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
