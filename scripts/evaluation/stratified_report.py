"""分层评估：按海域、强度分档看模型误差。

核心问题：「环境场/云图是不是在某些子集上有效，只是被全局平均掩盖？」

数据来源: data/processed/samples.npz
  last_abs (N,2)   预报起始时的绝对位置 (lat, lon)
  y_abs    (N,4,2) 未来 4 步绝对位置
  y        (N,4,2) 未来 4 步增量
  sid      (N,)    台风编号
  feature_names    特征名（含 wind_ms 等）

由于该文件不含模型预测，本脚本做两件事：
  1. 计算「持续法」的分层误差，建立各海域/强度档的基线难度
  2. 若有模型逐样本预测（--pred），则做分层对比

用法:
    # 只看基线难度
    python scripts/evaluation/stratified_report.py

    # 有模型预测时做分层对比
    python scripts/evaluation/stratified_report.py --pred model_preds.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.data.features import haversine_km  # noqa: E402
from src.evaluation.stratified import (  # noqa: E402
    classify_intensity, classify_region, compare_stratified, stratified_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="分层评估")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--pred", default=None,
                    help="模型逐样本预测 npz（含 pred_abs (N,4,2)）")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    proc = resolve_path(cfg, cfg.data.processed_dir, ROOT)
    d = np.load(proc / "samples.npz", allow_pickle=True)

    last = d["last_abs"].astype(float)          # (N,2)
    y_abs = d["y_abs"].astype(float)            # (N,4,2)
    names = [str(x) for x in d["feature_names"]]
    lat0, lon0 = last[:, 0], last[:, 1]

    print(f"[strat] 样本 {len(last)} 个")
    print(f"[strat] 特征 {names}")

    # ---------- 24h 误差（第 4 步）----------
    def err24(true_abs, pred_abs=None):
        """预测的 24h 位置误差（km）。pred_abs=None 时用持续法。"""
        if pred_abs is None:
            pred_abs = np.repeat(last[:, None, :], 4, axis=1)
        return haversine_km(pred_abs[:, -1, 0], pred_abs[:, -1, 1],
                            true_abs[:, -1, 0], true_abs[:, -1, 1])

    # ---------- 分组标签 ----------
    region = classify_region(lat0, lon0)
    region = np.where(lon0 > 180, "远洋(菲东)", region)  # 修正：西太经度>180 归远洋

    if "wind_ms" in names:
        wi = names.index("wind_ms")
        # 训练时做过标准化，这里用起点经纬度重算不方便；
        # 改用 lat_norm 无法反推，故从原始轨迹取强度
        tracks = pd.read_parquet(resolve_path(cfg, cfg.data.raw_dir, ROOT)
                                 / f"tracks_{cfg.data.basin}_clean.parquet")
        tracks["time"] = pd.to_datetime(tracks["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        print("  [提示] 强度分档将从轨迹表按 sid 匹配")
        wind0 = None
    else:
        wind0 = None

    # ---------- 1) 持续法基线难度 ----------
    print("\n" + "=" * 66)
    print("持续法的 24h 误差（各分组难度基线）")
    print("=" * 66)
    err_persist = err24(y_abs)
    tab_region = stratified_report(err_persist, region, title="按海域")

    # 强度分档：从轨迹表取起点强度
    try:
        tracks["sid"] = tracks["sid"].astype(str)
        key = pd.DataFrame({"sid": d["sid"].astype(str),
                            "lat": np.round(lat0, 1), "lon": np.round(lon0, 1)})
        # 用近似位置匹配起点强度（轨迹 6h 间隔，位置容差 0.3°）
        tr = tracks[["sid", "lat", "lon", "wind"]].copy()
        tr["lat"] = tr["lat"].round(1); tr["lon"] = tr["lon"].round(1)
        # 同一台风可能在同一位置出现多次（6h 间隔内的重复观测），
        # 先去重，避免 merge 后行数膨胀（产生笛卡尔积）
        tr = tr.drop_duplicates(subset=["sid", "lat", "lon"], keep="first")
        merged = key.merge(tr, on=["sid", "lat", "lon"], how="left")
        if len(merged) != len(key):
            print(f"  [警告] 匹配后行数 {len(merged)} != 样本数 {len(key)}，截断")
            merged = merged.iloc[:len(key)]
        w = merged["wind"].to_numpy(float)
        kt_to_ms = 0.514444
        inten = classify_intensity(w * kt_to_ms)
        n_ok = np.isfinite(w).sum()
        print(f"\n  强度匹配: {n_ok}/{len(w)} ({n_ok/len(w):.0%})")
        if n_ok > 100:
            stratified_report(err_persist, inten, title="按强度分档")
    except Exception as e:
        print(f"  [跳过强度分档] {e}")

    # ---------- 2) 模型预测的分层对比 ----------
    if args.pred:
        m = np.load(args.pred, allow_pickle=True)
        key = "pred_abs" if "pred_abs" in m else list(m.keys())[0]
        pred_abs = m[key].astype(float)
        if len(pred_abs) != len(y_abs):
            print(f"[错误] 预测样本数 {len(pred_abs)} != 样本数 {len(y_abs)}")
            return 1
        err_model = err24(y_abs, pred_abs)
        print("\n" + "=" * 66)
        print("模型 vs 持续法（分层）")
        print("=" * 66)
        print(f"全局: 持续法 {err_persist.mean():.1f} km, 模型 {err_model.mean():.1f} km")
        compare_stratified(err_persist, err_model, region,
                           name_a="持续法", name_b="模型")
        out = ROOT / "outputs" / "stratified" / "region_compare.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        compare_stratified(err_persist, err_model, region, "持续法", "模型").to_csv(
            out, index=False, encoding="utf-8")
        print(f"\n已保存: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
