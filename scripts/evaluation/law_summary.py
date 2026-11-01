"""汇总「无标签物理律」两组实验（技术报告 §4.13），产出可复现的统计表。

输出 `outputs/physics_laws/law_summary.json`，供
`scripts/evaluation/plot_physics_law_figures.py` 出图，也供报告核对数字。

两组：
* A 组 `outputs/lawA_*`：全量 20094 窗口，year 划分，smooth+curvature。
  含 4 个种子（42 / 0 / 1 / 2）的 w=0 与 w=0.1。
* B 组 `outputs/lawB_*`：并入 9 维环境场（1644 窗口），storm 划分，三项全开。

口径
----
* 指标：测试集逐时效球面距离 MAE（km），与各自 `logs/summary.json` 对账。
* 配对检验：同一 seed 内、同一测试集 → 逐窗口配对差；按**台风**做聚类
  bootstrap（同一台风的相邻窗口高度相关，按窗口重采样会高估显著性）。
* 跨 seed：以 seed 为单位做配对 t 检验 + 符号检验（n=4，功效很低，只作方向参考）。

用法:
    python scripts/evaluation/law_summary.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config                            # noqa: E402
from src.data.dataset import (TrackWindowDataset, load_bundle,  # noqa: E402
                              split_by_storm_random, split_by_year)
from src.models.typhoon_transformer import build_model        # noqa: E402
from src.training.losses import error_km                      # noqa: E402
from src.training.physics import (KM_PER_DEG, SCALE_ACC_KM,   # noqa: E402
                                  SCALE_VEL_KM, TURN_ABS_DEG, _cos_lat)
from src.training.physics_laws import (_pred_velocity_km,     # noqa: E402
                                       build_ctx, fit_steer_k,
                                       steering_penalty)
from src.training.trainer import make_loader, predict         # noqa: E402

LEADS = [6, 12, 18, 24]
N_BOOT = 4000
RNG = np.random.default_rng(20261101)


def parse_run(name: str):
    """lawA_s0_w0.1 → ('A', 0, 0.1)；lawA_w0.0 → ('A', 42, 0.0)。"""
    m = re.match(r"law([AB])_s(\d+)_w([\d.]+)$", name)
    if m:
        return m.group(1), int(m.group(2)), float(m.group(3))
    m = re.match(r"law([AB])_w([\d.]+)$", name)
    if m:
        return m.group(1), 42, float(m.group(2))
    return None


def discover(pattern: str):
    out = []
    for d in sorted(ROOT.glob(pattern)):
        if not d.is_dir():
            continue
        key = parse_run(d.name)
        if key and (d / "checkpoints" / "best.pt").exists():
            out.append((key, d))
    return out


def kinematics(delta_deg, last_abs):
    """返回逐时效速度 (N,T,2) km、加速度模 (N,T-1)、转角 (N,T-1)。"""
    delta = torch.as_tensor(delta_deg, dtype=torch.float64)
    last = torch.as_tensor(last_abs, dtype=torch.float64)
    v = _pred_velocity_km(delta, last[:, 0])
    acc = v[:, 1:] - v[:, :-1]
    acc_mag = torch.sqrt((acc ** 2).sum(-1))
    v1, v2 = v[:, :-1], v[:, 1:]
    cosang = (v1 * v2).sum(-1) / (
        torch.sqrt((v1 ** 2).sum(-1) + 1e-6) * torch.sqrt((v2 ** 2).sum(-1) + 1e-6))
    ang = torch.rad2deg(torch.arccos(torch.clamp(cosang, -1.0, 1.0)))
    return v.numpy(), acc_mag.numpy(), ang.numpy()


def run_one(d: Path, bundle, spec, splits):
    """推理一个 run，返回逐样本误差与运动学；同时与 summary.json 对账。"""
    cfg = load_config(d / "used_config.yaml")
    model = build_model(cfg, feat_dim=bundle["x"].shape[-1],
                        in_len=bundle["x"].shape[1], out_len=bundle["y"].shape[1])
    model.load_state_dict(torch.load(d / "checkpoints" / "best.pt",
                                     map_location="cpu", weights_only=False)["model"])
    ds = TrackWindowDataset(bundle, spec, splits["test"])
    pred = predict(model, make_loader(ds, 256, False, 0), torch.device("cpu"))
    summary = json.loads((d / "logs" / "summary.json").read_text(encoding="utf-8"))
    dist = error_km(pred["pred_abs"], pred["true_abs"])
    ref = np.asarray(summary["test"]["mae_by_step"])
    if not np.allclose(dist.mean(0), ref, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{d.name} 复现失败: {dist.mean(0)} vs {ref}")

    delta = pred["pred_abs"] - pred["last_abs"][:, None, :]
    delta[..., 1] = (delta[..., 1] + 180) % 360 - 180
    v, acc, ang = kinematics(delta, pred["last_abs"])
    return {"pred": pred, "dist": dist, "summary": summary,
            "v": v, "acc": acc, "ang": ang}


def bootstrap_ci(diff, idx_by_storm: list[np.ndarray], n_boot: int = N_BOOT):
    """按台风重采样的均值差 CI 与双侧 p（p=2·min(单侧比例)）。"""
    k = len(idx_by_storm)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.integers(0, k, k)
        ids = np.concatenate([idx_by_storm[q] for q in pick])
        boots[b] = diff[ids].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return [round(float(lo), 3), round(float(hi), 3)], round(float(min(p, 1.0)), 4)


def summarize_run(res):
    dist = res["dist"]
    mae = [round(float(x), 3) for x in dist.mean(0)]
    return {
        "best_epoch": res["summary"]["best_epoch"],
        "val_mae": round(float(res["summary"]["best_val_mae_km"]), 3),
        "mae_by_step": mae,
        "mae_overall": round(float(dist.mean()), 3),
        "acc_mean_km": round(float(res["acc"].mean()), 2),
        "turn_mean_deg": round(float(res["ang"].mean()), 3),
        "smooth_norm": round(float(np.sqrt((res["acc"] ** 2).mean())) / SCALE_ACC_KM, 4),
        "step_km": [round(float(np.linalg.norm(res["v"][:, t], axis=-1).mean()), 2)
                    for t in range(res["v"].shape[1])],
    }


def main() -> int:
    global N_BOOT
    ap = argparse.ArgumentParser(description="汇总无标签物理律实验（§4.13）")
    ap.add_argument("--out", default="outputs/physics_laws")
    ap.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap 次数")
    args = ap.parse_args()
    N_BOOT = args.boot

    base = load_config(ROOT / "configs" / "default.yaml")
    bundle, spec = load_bundle(ROOT / base.data.processed_dir)
    splits = split_by_year(bundle, list(base.data.val_years), list(base.data.test_years))
    sid = bundle["sid"][splits["test"]]
    storms = np.unique(sid)
    idx_by_storm = [np.where(sid == s)[0] for s in storms]
    print(f"[A] test windows={len(sid)}, storms={len(storms)}")

    out = {"A": {"n_windows": int(len(sid)), "n_storms": int(len(storms)),
                 "runs": {}, "per_seed": {}, "cross_seed": {}},
           "B": {}}

    # ---- 真值运动学（A 组测试集） ----
    delta_t = bundle["y_abs"][splits["test"]] - bundle["last_abs"][splits["test"]][:, None, :]
    delta_t[..., 1] = (delta_t[..., 1] + 180) % 360 - 180
    vt, acc_t, ang_t = kinematics(delta_t, bundle["last_abs"][splits["test"]])
    out["A"]["truth"] = {
        "acc_mean_km": round(float(acc_t.mean()), 2),
        "turn_mean_deg": round(float(ang_t.mean()), 3),
        "step_km": [round(float(np.linalg.norm(vt[:, t], axis=-1).mean()), 2)
                    for t in range(vt.shape[1])],
    }

    # ---- A 组 ----
    runs_a, errs = {}, {}
    for (grp, seed, w), d in discover("outputs/lawA_*"):
        if grp != "A":
            continue
        res = run_one(d, bundle, spec, splits)
        runs_a[(seed, w)] = summarize_run(res)
        errs[(seed, w)] = res["dist"]
        print(f"  {d.name:<18} best={runs_a[(seed, w)]['best_epoch']:>3} "
              f"val={runs_a[(seed, w)]['val_mae']:>7.2f} mean={runs_a[(seed, w)]['mae_overall']:>7.2f}")
    out["A"]["runs"] = {f"s{s}_w{w}": r for (s, w), r in runs_a.items()}

    seeds = sorted({s for s, _ in runs_a})
    for s in seeds:
        if (s, 0.0) not in errs or (s, 0.1) not in errs:
            continue
        diff = errs[(s, 0.1)] - errs[(s, 0.0)]
        row = {}
        for t, h in enumerate(LEADS):
            ci, p = bootstrap_ci(diff[:, t], idx_by_storm)
            row[f"{h}h"] = {"delta": round(float(diff[:, t].mean()), 3),
                            "ci95": ci, "p_boot": p}
        row["all"] = {"delta": round(float(diff.mean()), 3)}
        out["A"]["per_seed"][str(s)] = row

    paired = [s for s in seeds if (s, 0.0) in runs_a and (s, 0.1) in runs_a]
    for idx, label in enumerate(["6h", "12h", "18h", "24h", "all"]):
        key = "mae_by_step" if label != "all" else "mae_overall"
        a = np.array([runs_a[(s, 0.0)][key][idx] if label != "all" else runs_a[(s, 0.0)][key]
                      for s in paired])
        b = np.array([runs_a[(s, 0.1)][key][idx] if label != "all" else runs_a[(s, 0.1)][key]
                      for s in paired])
        d = b - a
        tt = stats.ttest_rel(b, a)
        out["A"]["cross_seed"][label] = {
            "base_mean": round(float(a.mean()), 3), "base_sd": round(float(a.std(ddof=1)), 3),
            "law_mean": round(float(b.mean()), 3), "law_sd": round(float(b.std(ddof=1)), 3),
            "delta": [round(float(x), 3) for x in d],
            "mean_delta": round(float(d.mean()), 3),
            "sd_delta": round(float(d.std(ddof=1)), 3),
            "t": round(float(tt.statistic), 3), "p": round(float(tt.pvalue), 4),
            "p_sign": round(float(stats.binomtest(int((d < 0).sum()), len(d), 0.5).pvalue), 4),
            "n_better": int((d < 0).sum()), "n": len(d),
        }
        o = out["A"]["cross_seed"][label]
        print(f"  {label:>4}: {o['base_mean']:.2f}±{o['base_sd']:.2f} → "
              f"{o['law_mean']:.2f}±{o['law_sd']:.2f}  Δ={o['mean_delta']:+.2f} "
              f"p={o['p']:.3f} ({o['n_better']}/{o['n']})")

    # ---- B 组（环境场 + storm 划分）----
    env_path = ROOT / "data_ext" / "env_era5.parquet"
    runs_b = discover("outputs/lawB_*")
    if runs_b and env_path.exists():
        spec_mod = importlib.util.spec_from_file_location(
            "train_entry", ROOT / "scripts" / "path" / "train.py")
        train_entry = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(train_entry)
        b_bundle, b_spec = train_entry._build_with_env(base, ROOT / base.data.processed_dir, env_path)
        b_splits = split_by_storm_random(b_bundle, seed=int(base.seed), verbose=False)
        b_storms = np.unique(b_bundle["sid"][b_splits["test"]])
        b_idx = [np.where(b_bundle["sid"][b_splits["test"]] == s)[0] for s in b_storms]
        print(f"[B] test windows={len(b_splits['test'])}, storms={len(b_storms)}")

        ctx = build_ctx(b_spec.names, b_spec.mean, b_spec.std, float(base.data.interval_hours))
        tr = b_splits["train"]
        k, r2 = fit_steer_k(
            torch.tensor(b_bundle["y"][tr][:, 0], dtype=torch.float32),
            torch.tensor((b_bundle["x"][tr] - b_spec.mean) / b_spec.std, dtype=torch.float32),
            torch.tensor(b_bundle["last_abs"][tr, 0], dtype=torch.float32), ctx)

        out["B"] = {"n_windows": int(len(b_splits["test"])), "n_storms": int(len(b_storms)),
                    "k": round(float(k), 4), "k_r2": round(float(r2), 4), "runs": {}}
        b_errs, b_res = {}, {}
        for (grp, _seed, w), d in runs_b:
            if grp != "B":
                continue
            res = run_one(d, b_bundle, b_spec, b_splits)
            stat = summarize_run(res)
            delta = res["pred"]["pred_abs"] - res["pred"]["last_abs"][:, None, :]
            delta[..., 1] = (delta[..., 1] + 180) % 360 - 180
            with torch.no_grad():
                st, _ = steering_penalty(torch.as_tensor(delta, dtype=torch.float32),
                                         torch.as_tensor((b_bundle["x"][b_splits["test"]]
                                                          - b_spec.mean) / b_spec.std,
                                                         dtype=torch.float32),
                                         torch.as_tensor(res["pred"]["last_abs"],
                                                         dtype=torch.float32)[:, 0],
                                         ctx, k=torch.tensor(k))
            stat["steering_pen"] = round(float(st), 4)
            _last = res["pred"]["last_abs"]
            _clat = np.clip(np.cos(np.deg2rad(_last[:, 0])), 0.15, None)
            _d = res["pred"]["pred_abs"] - _last[:, None, :]
            _d[..., 1] = (_d[..., 1] + 180) % 360 - 180
            stat["cum_step_km"] = [round(float(np.hypot(
                _d[:, t, 0] * KM_PER_DEG, _d[:, t, 1] * KM_PER_DEG * _clat).mean()), 1)
                for t in range(4)]
            out["B"]["runs"][f"w{w}"] = stat
            b_errs[w] = res["dist"]
            b_res[w] = res
            print(f"  {d.name:<12} mean={stat['mae_overall']:>7.2f} steering={stat['steering_pen']:.3f}")

        # 真值 + 引导流先验
        truth_abs = b_bundle["y_abs"][b_splits["test"]]
        last_abs = b_bundle["last_abs"][b_splits["test"]]
        dt = truth_abs - last_abs[:, None, :]
        dt[..., 1] = (dt[..., 1] + 180) % 360 - 180
        vtt, acc_tt, ang_tt = kinematics(dt, last_abs)
        x_te = (b_bundle["x"][b_splits["test"]] - b_spec.mean) / b_spec.std
        u = torch.as_tensor(x_te[:, -1, ctx.u_idx] * float(ctx.std[ctx.u_idx])
                            + float(ctx.mean[ctx.u_idx]), dtype=torch.float64)
        v = torch.as_tensor(x_te[:, -1, ctx.v_idx] * float(ctx.std[ctx.v_idx])
                            + float(ctx.mean[ctx.v_idx]), dtype=torch.float64)
        per = 3.6 * ctx.interval_h
        lat = torch.as_tensor(last_abs[:, 0], dtype=torch.float64)
        lon = torch.as_tensor(last_abs[:, 1], dtype=torch.float64)
        clat = _cos_lat(lat)
        pts = []
        for _ in range(4):
            pts.append(torch.stack([lat, lon], -1))
            lat = lat + k * v * per / KM_PER_DEG
            lon = lon + k * u * per / (KM_PER_DEG * clat)
        steer_pred = torch.stack(pts, 1).numpy()
        steer_err = error_km(steer_pred, truth_abs)

        out["B"]["truth"] = {
            "mae_by_step": [round(float(x), 3) for x in steer_err.mean(0)],
            "mae_overall": round(float(steer_err.mean()), 3),
            "acc_mean_km": round(float(acc_tt.mean()), 2),
            "turn_mean_deg": round(float(ang_tt.mean()), 3),
            "cum_step_km": [round(float(np.hypot(
                (truth_abs[:, t, 0] - last_abs[:, 0]) * KM_PER_DEG,
                ((truth_abs[:, t, 1] - last_abs[:, 1] + 180) % 360 - 180) * KM_PER_DEG
                * np.clip(np.cos(np.deg2rad(last_abs[:, 0])), 0.15, None)).mean()), 1)
                for t in range(4)],
        }

        # 24h 误差分解（沿轨/横轨）
        def decompose(res):
            last = res["pred"]["last_abs"]
            d = res["pred"]["pred_abs"] - last[:, None, :]
            d[..., 1] = (d[..., 1] + 180) % 360 - 180
            clat_np = np.clip(np.cos(np.deg2rad(last[:, 0])), 0.15, None)
            dy = d[:, -1, 0] * KM_PER_DEG
            dx = d[:, -1, 1] * KM_PER_DEG * clat_np
            ty = dt[:, -1, 0] * KM_PER_DEG
            tx = dt[:, -1, 1] * KM_PER_DEG * clat_np
            ey, ex = dy - ty, dx - tx
            # 沿轨 / 横轨：直接做点积与叉积，避免 atan2 的轴序混淆
            nu = np.hypot(ty, tx)
            along = (ey * ty + ex * tx) / nu
            cross = (ex * ty - ey * tx) / nu
            return {"pred_disp_km": round(float(np.hypot(dy, dx).mean()), 1),
                    "true_disp_km": round(float(np.hypot(ty, tx).mean()), 1),
                    "along": round(float(along.mean()), 1),
                    "cross_abs": round(float(np.abs(cross).mean()), 1)}

        for w, res in b_res.items():
            out["B"]["runs"][f"w{w}"]["decomp24"] = decompose(res)
        if 0.0 in b_errs and 0.3 in b_errs:
            diff = b_errs[0.3] - b_errs[0.0]
            out["B"]["vs_w0.0"] = {}
            for t, h in enumerate(LEADS):
                ci, p = bootstrap_ci(diff[:, t], b_idx)
                out["B"]["vs_w0.0"][f"{h}h"] = {"delta": round(float(diff[:, t].mean()), 3),
                                                "ci95": ci, "p_boot": p}
            ci_all, p_all = bootstrap_ci(diff.mean(1), b_idx)
            out["B"]["vs_w0.0"]["all"] = {"delta": round(float(diff.mean()), 3),
                                          "ci95": ci_all, "p_boot": p_all}
    else:
        print("[B] 跳过（无 lawB_* 目录或缺少 data_ext/env_era5.parquet）")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "law_summary.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
