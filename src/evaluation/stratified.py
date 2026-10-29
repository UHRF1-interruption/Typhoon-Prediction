"""分层评估：把误差按海域、强度、移动速度等维度拆开看。

为什么需要分层
--------------
全局平均 MAE 会掩盖不同子集的巨大差异。例如：
  * 南海台风受地形影响大，菲东台风多在开阔洋面
  * 弱台风（<25 m/s）的强度变化小，MAE 天然低
  * 快速移动的台风路径更接近直线，持续法都能做好

按子集分别评估可以回答：「环境场/云图是不是在某些子集上其实有效，
只是被全局平均掩盖了？」
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------- 海域划分（按台风中心位置）----------
SEA_REGIONS = {
    "南海":        dict(lat=(5, 25), lon=(105, 122)),
    "吕宋以东":     dict(lat=(5, 22), lon=(122, 135)),
    "远洋(菲东)":   dict(lat=(5, 30), lon=(135, 180)),
    "日本以南":     dict(lat=(25, 35), lon=(125, 145)),
    "华东近海":     dict(lat=(25, 35), lon=(118, 130)),
}


def classify_region(lat, lon) -> np.ndarray:
    """按经纬度划分海域。未落入选定区域归为「其他」。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float) % 360.0
    out = np.full(lat.shape, "其他", dtype=object)
    for name, box in SEA_REGIONS.items():
        m = ((lat >= box["lat"][0]) & (lat <= box["lat"][1])
             & (lon >= box["lon"][0]) & (lon <= box["lon"][1]))
        # 后定义的区域不覆盖已分配的（按字典顺序优先）
        out[m & (out == "其他")] = name
    return out


def classify_intensity(wind_ms, bins=(0, 25, 33, 45, 200),
                       labels=("弱(<25)", "中(25-33)", "强(33-45)", "超强(>45)")):
    """按风速（m/s）分档。默认用中国标准：TS/STS/TY/STY。"""
    w = np.asarray(wind_ms, dtype=float)
    return pd.cut(w, bins=bins, labels=labels, right=False).astype(object)


def stratified_report(errors: np.ndarray, labels: np.ndarray,
                      title: str = "", min_n: int = 20) -> pd.DataFrame:
    """按标签分组统计误差。

    errors: 每个样本的绝对误差（km 或 m/s）
    labels: 分组标签
    """
    df = pd.DataFrame({"err": np.asarray(errors, dtype=float),
                       "grp": np.asarray(labels, dtype=object)})
    rows = []
    for name, sub in df.groupby("grp", sort=False):
        if len(sub) < min_n:
            continue
        rows.append({
            "分组": str(name),
            "样本数": len(sub),
            "MAE": float(sub["err"].mean()),
            "中位数": float(sub["err"].median()),
            "P90": float(sub["err"].quantile(0.90)),
            "标准差": float(sub["err"].std()),
        })
    out = pd.DataFrame(rows).sort_values("样本数", ascending=False)
    if title:
        print(f"\n=== {title} ===")
        print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return out


def compare_stratified(err_a: np.ndarray, err_b: np.ndarray, labels: np.ndarray,
                       name_a: str = "A", name_b: str = "B",
                       min_n: int = 20) -> pd.DataFrame:
    """在两个配置间做分层对比（配对，同一批样本）。

    返回每个分组的 A/B MAE 与差值，并给出组内配对 t 检验。
    """
    from scipy import stats

    df = pd.DataFrame({"a": np.asarray(err_a, float), "b": np.asarray(err_b, float),
                       "grp": np.asarray(labels, object)})
    rows = []
    for name, sub in df.groupby("grp", sort=False):
        if len(sub) < min_n:
            continue
        diff = sub["b"].to_numpy() - sub["a"].to_numpy()
        if len(diff) > 1 and diff.std(ddof=1) > 1e-9:
            t, p = stats.ttest_rel(sub["b"], sub["a"])
            d = diff.mean() / diff.std(ddof=1)
        else:
            t = p = d = float("nan")
        rows.append({
            "分组": str(name), "样本数": len(sub),
            f"{name_a}": float(sub["a"].mean()),
            f"{name_b}": float(sub["b"].mean()),
            "差值": float(diff.mean()),
            "相对%": float(diff.mean() / max(1e-9, sub["a"].mean()) * 100),
            "B更好": int((diff < 0).sum()),
            "p值": float(p), "d": float(d),
        })
    out = pd.DataFrame(rows)
    print(f"\n=== 分层对比（{name_a} vs {name_b}）===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}")
          if len(out) else "  (样本不足)")
    return out
