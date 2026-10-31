"""把微波 patch 附加到强度任务的滑动窗口样本上。

设计要点（都来自对数据的实测）
------------------------------
1. **微波是离散观测，不是连续序列。** 每台风中位 44 次观测、相邻间隔中位
   2.4 h，且观测时刻相对 3h 网格有约 30 min 的偏移。若要求 8 步窗口每步
   都恰好有微波（±1.5h），可用样本只剩 21 个；±3h 才有 2260 个。

2. 因此**不把微波做成序列，而是取窗口末端之前的一个"锚观测"**：
       锚 = 该台风在 [窗口末端-3h, 窗口末端] 内最近的一次微波观测
   并把「锚与窗口末端的时间差 dt（0-3h）」作为额外输入，让模型自学时效衰减。
   这比强行插值到网格上更诚实（微波本来就是快照）。

3. 对齐靠 `names` 字段：微波 npz 由 annotate_mw_names.py 补入了台风名，
   与红外、标量表同一命名体系。没有 names 的旧文件会被跳过并提示。

输出 bundle 结构（与 sat_intensity.build_intensity_samples 兼容，额外三项）：
    s        (N, in_len, F)        标量序列（原样复用）
    y        (N, out_len)          未来强度
    name     (N,)                  台风名
    mw       (N, C, S, S)          锚观测的微波 patch（float32, K）
    mw_dt    (N,)                  锚与窗口末端的时间差（小时，0-3）
    mw_mask  (N, C)                通道可信度（0=原始, 1=缺失/已替代）
    has_mw   (N,)  bool            是否带微波（False 表示只做了标量回退）
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_microwave(proc_dir: Path, pattern: str = "sat_patches_mw64_*.npz"):
    """读取所有微波 npz，返回 {台风名(大写): [(time, patch, channel_mask, info), ...]}。

    只接受带 names 字段的文件；缺 names 时提示先跑 annotate_mw_names.py。
    """
    files = sorted(glob.glob(str(Path(proc_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"{proc_dir} 下没有 {pattern}")
    # 关键：**只存索引，不在载入时复制 patch**。
    # npz 是压缩包，np.load 的数组是惰性解压的；逐帧取 d["patches"][i]
    # 会把整幅数组展开，8319 帧 x 4x64x64 约 1.3 GB，加上解压中间对象
    # 会把进程撑爆（实测被系统 SIGTERM）。改为记录 (时间, 文件, 行号)，
    # 取用时按文件整块加载一次（见 attach_microwave 的 _PatchStore）。
    out: dict[str, list] = {}
    n_frames = 0
    no_names = []
    for f in files:
        d = np.load(f)
        if "names" not in d.files:
            no_names.append(Path(f).name)
            continue
        for i, k in enumerate(d["keys"]):
            nm = str(d["names"][i]).upper()
            if not nm:
                continue
            ts = str(k).split("|", 1)[1]
            out.setdefault(nm, []).append(
                (pd.Timestamp(ts, tz="UTC"), str(f), int(i)))
            n_frames += 1
    if no_names:
        print(f"[mw] 跳过 {len(no_names)} 个缺 names 的文件: {no_names}")
        print("[mw] 请先运行: python scripts/data/annotate_mw_names.py")
    for v in out.values():
        v.sort(key=lambda x: x[0])
    print(f"[mw] 载入 {n_frames} 帧, 覆盖 {len(out)} 个台风")
    return out


class _PatchStore:
    """按文件缓存已解压的 patch 数组，避免逐帧解压。

    npz 每次取 d["patches"][i] 都要把整个压缩数组展开一次；这里按文件
    整块载入并做引用计数式淘汰（只保留最近一个文件），内存峰值约
    一个文件的解压大小（最大约 200 MB）。
    """

    def __init__(self):
        self._cur_file = None
        self._arr = None
        self._mask = None
        self.shape = None

    def _load(self, f):
        if f == self._cur_file:
            return
        d = np.load(f)
        self._arr = d["patches"]
        self._mask = d["channel_mask"]
        self._cur_file = f
        if self.shape is None:
            self.shape = (self._arr.shape[1], self._arr.shape[2])

    def get(self, f, i):
        self._load(f)
        return self._arr[i], self._mask[i]


def attach_microwave(bundle: dict, mw_index: dict, max_dt_h: float = 3.0):
    """给已有 bundle 的每个样本挂上锚观测。

    匹配规则：对样本的「最后输入时刻 t_end」，取该台风在
    [t_end - max_dt_h, t_end] 内**最晚**的一次微波观测。
    没有落到窗口内的观测则 has_mw=False，并把 mw 全置 0。

    为什么取"最晚"而不是"最近"：t_end 之后还有未来 12h 的标签，
    用 t_end 之后的观测会泄漏。只允许向前看 3h（即不晚于 t_end）。
    """
    names = np.asarray(bundle["name"]).astype(str)
    t_end = pd.to_datetime(np.asarray(bundle["end_time"]), utc=True)
    n = len(names)

    store = _PatchStore()
    # 先用第一个可用样本确定 patch 形状
    f0, i0 = next(iter(mw_index.values()))[0][1], next(iter(mw_index.values()))[0][2]
    store._load(f0)
    c, s = store.shape
    mw = np.zeros((n, c, s, s), np.float32)
    dt = np.full(n, np.nan, np.float32)
    mask = np.ones((n, c), np.uint8)      # 默认 1=不可用
    has = np.zeros(n, bool)

    win = pd.Timedelta(hours=max_dt_h)
    for i in range(n):
        lst = mw_index.get(names[i].upper())
        if not lst:
            continue
        te = t_end[i]
        # 找到 time <= te 的最后一个（lst 已按时间排序）
        lo, hi = 0, len(lst)
        while lo < hi:
            mid = (lo + hi) // 2
            if lst[mid][0] <= te:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        if j < 0:
            continue
        t_obs = lst[j][0]
        if te - t_obs > win:
            continue
        patch, cmask = store.get(lst[j][1], lst[j][2])
        mw[i] = patch
        mask[i] = cmask
        dt[i] = (te - t_obs).total_seconds() / 3600.0
        has[i] = True

    bundle = dict(bundle)
    bundle["mw"] = mw
    bundle["mw_dt"] = dt
    bundle["mw_mask"] = mask
    bundle["has_mw"] = has
    print(f"[mw] 样本 {n}, 带微波 {int(has.sum())} ({has.mean():.1%}), "
          f"dt 中位 {np.nanmedian(dt[has]) if has.any() else float('nan'):.2f} h")
    return bundle


def subset_with_mw(bundle: dict) -> dict:
    """只保留带微波的样本（用于纯微波对照）。"""
    keep = np.where(bundle["has_mw"])[0]
    out = {}
    for k, v in bundle.items():
        if isinstance(v, np.ndarray):
            out[k] = v[keep]
        else:
            out[k] = v
    return out


# 微波手工统计量：用少量物理量替代 CNN。
# 依据（AIES 2026 的 dropout 实验）：36.64 与 89.0 GHz 亮温最关键，
# 且 <89 GHz 的贡献更显著。这里取"中心区/眼墙/外圈"三层 × 各通道均值差，
# 加冷云面积——即台风强度分析里最常看的几个量。
MW_FEATURE_NAMES = [
    "mw_36v_center", "mw_36h_center", "mw_89v_center", "mw_89h_center",
    "mw_89v_eyewall", "mw_36v_eyewall",
    "mw_89v_eyeminus_env", "mw_36v_eyeminus_env",
    "mw_89v_min", "mw_36v_min",
    "mw_cold_frac_200", "mw_cold_frac_240",
    "mw_89v_std", "mw_36v_std",
    "mw_dt",          # 锚观测与窗口末端的时间差
]


def patch_statistics(p: np.ndarray, dt: float) -> np.ndarray:
    """单帧微波 patch (C,S,S) -> MW_FEATURE_NAMES 向量。

    通道序假定 [36.5V, 36.5H, 89.0V, 89.0H]（build_patch_microwave.py 的输出）。
    通道数与顺序不同时会退化为按前两/四个通道取，不做强校验——
    stats 只用于"有没有信号"的判定，不追求物理精确。
    """
    s = p.shape[-1]
    c0, c1 = int(s * 0.19), int(s * 0.81)          # 中心区约 ±0.19S
    eye = p[:, c0:c1, c0:c1]
    ring0, ring1 = int(s * 0.31), int(s * 0.50)    # 眼墙环带
    yy, xx = np.mgrid[0:s, 0:s]
    cy = cx = (s - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx)
    ring = (r >= ring0) & (r < ring1)
    env = r >= 0.5 * s

    def m(ch, mask):
        v = p[ch][mask]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else np.nan

    def center(ch):
        v = eye[ch][np.isfinite(eye[ch])]
        return float(v.mean()) if v.size else np.nan

    def std_all(ch):
        v = p[ch][np.isfinite(p[ch])]
        return float(v.std()) if v.size else np.nan

    def mn(ch):
        v = p[ch][np.isfinite(p[ch])]
        return float(v.min()) if v.size else np.nan

    nch = p.shape[0]
    g = lambda ch, f: f(ch) if ch < nch else np.nan  # noqa: E731
    f = [
        g(0, center), g(1, center), g(2, center), g(3, center),
        g(2, lambda ch: m(ch, ring)), g(0, lambda ch: m(ch, ring)),
        g(2, center) - g(2, lambda ch: m(ch, env)),
        g(0, center) - g(0, lambda ch: m(ch, env)),
        g(2, mn), g(0, mn),
        float(np.nanmean((p[2] < 200.0).astype(float))) if nch > 2 else np.nan,
        float(np.nanmean((p[2] < 240.0).astype(float))) if nch > 2 else np.nan,
        g(2, std_all), g(0, std_all),
        float(dt),
    ]
    return np.asarray(f, dtype=np.float32)


def attach_mw_features(bundle: dict, index: dict) -> dict:
    """给 bundle 补 mwfeat (N, F) 与 has_mw。用法与 attach_microwave 相同，
    但只在内存里算统计量，不留 patch，内存占用极小。

    index 为 load_microwave 返回的 {(name): [(time, file, row), ...]}。
    """
    store = _PatchStore()
    f0, i0 = next(iter(index.values()))[0][1], next(iter(index.values()))[0][2]
    store._load(f0)
    nfeat = len(MW_FEATURE_NAMES)
    names = np.asarray(bundle["name"]).astype(str)
    t_end = pd.to_datetime(np.asarray(bundle["end_time"]), utc=True)
    n = len(names)
    feat = np.full((n, nfeat), np.nan, np.float32)
    has = np.zeros(n, bool)
    win = pd.Timedelta(hours=3.0)
    for i in range(n):
        lst = index.get(names[i].upper())
        if not lst:
            continue
        te = t_end[i]
        lo, hi = 0, len(lst)
        while lo < hi:
            mid = (lo + hi) // 2
            if lst[mid][0] <= te:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        if j < 0:
            continue
        t_obs = lst[j][0]
        dt = (te - t_obs).total_seconds() / 3600.0
        if dt > win.total_seconds() / 3600.0:
            continue
        p, _ = store.get(lst[j][1], lst[j][2])
        feat[i] = patch_statistics(p, dt)
        has[i] = True
    out = dict(bundle)
    out["mwfeat"] = feat
    out["has_mw"] = has
    print(f"[mwfeat] 样本 {n}, 带微波 {int(has.sum())} ({has.mean():.1%}), "
          f"特征 {nfeat} 维")
    return out
