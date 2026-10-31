"""命令行入口: 从 TC PRIMED 被动微波过境文件提取以台风中心为中心的 patch。

与 build_sat_patches.py（红外）并列：本脚本处理**微波**，输出格式与其兼容，
便于把微波 patch 作为独立通道或独立分支接入现有强度模型。

为什么需要二维插值
------------------
TC PRIMED 的 x/y 是**圆锥扫描投影的曲线坐标**：x 随行变化、y 随列变化
（实测 x 跨行不同、y 跨列不同），且沿轨方向存在观测间隙。
因此不能像规则经纬网格那样直接切片，必须把散点 (x, y, TB) 用
`scipy.interpolate.griddata` 插值到以台风中心为原点的规则公里网格上。

通道选择
--------
默认 36.5V/H + 89.0V/H 共 4 通道。依据（AIES 2026, Razin et al. 对流/层云分类）：
按 dropout 实验，36.64 与 89.0 GHz 亮温是最重要的预测因子，且 <89 GHz 的信息
贡献更显著。89 GHz 对冰散射敏感（深对流），37 对液态水发射敏感（低层降水）。

注意：89 GHz 在 AMSR2/GMI 上分成 A/B 两个扫描段（S5/S6），需要合并后一起插值。

用法:
    # 单个文件测试
    python scripts/data/build_patch_microwave.py --limit 5

    # 指定年份与输出
    python scripts/data/build_patch_microwave.py \
        --seasons 2022 2023 --out data/processed/sat_patches_mw64.npz

    # 只保留离台风中心近的样本（默认 300 km，沿用 manifest 的筛选标准）
    python scripts/data/build_patch_microwave.py --max-dist-km 200

输出: data/processed/sat_patches_mw64.npz
    keys    (N,)      "<台风名>|<时间>"，与红外缓存同一命名规则
    patches (N, 4, 64, 64) float32   通道序 [36.5V, 36.5H, 89.0V, 89.0H]，单位 K
    meta    (N,)  JSON 字符串：sid / atcf / instrument / 有效像素比例 / 中心距离
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
import pandas as pd  # noqa: E402
import netCDF4 as nc  # noqa: E402

from src.config import load_config  # noqa: E402

# 目标通道：(频率 GHz, 极化)。不同传感器的频率命名不同，见下方 TOL。
#   AMSR2 : 36.5 / 89.0（89 分 A/B 两段）
#   GMI   : 36.64 / 89.0
#   SSMIS : 37.0 / 91.665（无 36.5，最接近的是 37.0）
TARGET_CHANNELS = [(36.5, "V"), (36.5, "H"), (89.0, "V"), (89.0, "H")]
CHANNEL_LABELS = ["mw36.5V", "mw36.5H", "mw89.0V", "mw89.0H"]
# 频率匹配容差（GHz）。SSMIS 的 37.0 与 36.5 差 0.5，91.665 与 89.0 差 2.7，需放宽容差
FREQ_TOL = 3.0
# 亮温物理下限（K）。低于此值视为定标残差，截断后再入训练。
TB_FLOOR_K = 100.0


def _scan_channels(pm):
    """扫描所有组，返回 [(freq, pol, [变量名...])]，同一 (freq,pol) 的多段合并。

    不能按固定组名（S4/S5/S6）取：实测三种传感器的分组方式完全不同。
    """
    from collections import defaultdict
    found = defaultdict(list)
    for gname in sorted(pm.groups.keys()):
        grp = pm.groups[gname]
        for v in grp.variables:
            if not v.startswith("TB_"):
                continue
            body = v[3:]                      # 例如 "36.5V" / "A89.0V" / "36.64H" / "183.31_3.0V"
            pol = body[-1]
            if pol not in ("V", "H"):
                continue
            num = body[:-1]
            num = num.lstrip("AB")            # 去掉 A/B 分段前缀
            try:
                freq = float(num.split("_")[0])   # "183.31_3.0" -> 183.31
            except ValueError:
                continue
            found[(freq, pol)].append((gname, v))
    return found


def _pick_channel(found, target_freq, pol):
    """在扫描结果里找与目标频率最接近的通道（容差内）。"""
    best = None
    for (f, p), items in found.items():
        if p != pol:
            continue
        d = abs(f - target_freq)
        if d <= FREQ_TOL and (best is None or d < best[0]):
            best = (d, f, items)
    return best


def _channel_empty(pm, items, min_points: int = 50) -> bool:
    """该通道在文件里是否几乎没有有效数据。

    只读第一个变量的元信息（netCDF4 读一维切片即可，不必载入整个数组）。
    """
    for gname, vname in items:
        try:
            var = pm.groups[gname].variables[vname]
            tb = np.ma.filled(var[:].astype(np.float32), np.nan)
            if np.isfinite(tb).sum() >= min_points:
                return False
        except Exception:  # noqa: BLE001
            continue
    return True


def _collect_points(pm, items):
    """从若干 (组名, 变量名) 收集散点 (x, y, TB)。"""
    xs, ys, ts = [], [], []
    for gname, vname in items:
        grp = pm.groups[gname]
        tb = grp.variables[vname][:].astype(np.float32)
        x = grp.variables["x"][:].astype(np.float32)
        y = grp.variables["y"][:].astype(np.float32)
        tb = np.ma.filled(np.ma.masked_invalid(tb), np.nan)
        xs.append(np.asarray(x).ravel())
        ys.append(np.asarray(y).ravel())
        ts.append(np.asarray(tb).ravel())
    if not xs:
        return None
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ts))


def extract_patch_mw(path: Path, half_km: float = 128.0, out_size: int = 64,
                     min_valid: float = 0.5):
    """从单个 TC PRIMED 文件提取 (4, out_size, out_size) 的 patch。

    返回 (patch, info)。patch 中无法插值的格点为 NaN（由调用方决定填充策略）。
    info 含台风位置、时间、仪器与有效比例。
    """
    from scipy.interpolate import griddata

    with nc.Dataset(path) as ds:
        pm = ds.groups["passive_microwave"]
        st = ds.groups["overpass_storm_metadata"]
        lat = float(st.variables["storm_latitude"][0])
        lon = float(st.variables["storm_longitude"][0])
        it = st.variables.get("intensity")
        inten = float(it[0]) if it is not None else float("nan")

        # 网格：以台风中心为原点，±half_km，out_size 点
        g = np.linspace(-half_km, half_km, out_size, dtype=np.float32)
        gx, gy = np.meshgrid(g, g)

        found = _scan_channels(pm)
        n_clip = [0]
        chans = []
        valid_ratios = []
        used = []      # 实际使用的通道描述，逐通道
        subst = []     # 每通道是否发生"极化替代"（0=原样, 1=用了另一极化）
        for (tfreq, tpol) in TARGET_CHANNELS:
            hit = _pick_channel(found, tfreq, tpol)
            substituted = 0
            # SSMIS 的 37V 在部分轨道整场缺失（实测 0/14850），而 37H 正常。
            # 用同频率的另一极化**填充**而不是留 NaN，否则整帧会被 min_valid 丢弃
            # （2020 年因此丢了 133 帧 = 22%）。
            # 但替代会破坏通道物理语义，所以逐通道记一个 mask（subst）。
            # 训练端可据此：(a) 丢弃该通道；(b) 作为"该通道不可信"的辅助输入。
            if hit is not None and _channel_empty(pm, hit[2]):
                alt = _pick_channel(found, tfreq, "H" if tpol == "V" else "V")
                if alt is not None and not _channel_empty(pm, alt[2]):
                    hit = alt
                    substituted = 1
            if hit is None:
                chans.append(np.full((out_size, out_size), np.nan, np.float32))
                valid_ratios.append(0.0)
                used.append("missing")
                subst.append(1)
                continue
            _, real_freq, items = hit
            pts = _collect_points(pm, items)
            if pts is None:
                chans.append(np.full((out_size, out_size), np.nan, np.float32))
                valid_ratios.append(0.0)
                used.append("empty")
                subst.append(1)
                continue
            x, y, tb = pts
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(tb)
            if ok.sum() < 100:
                chans.append(np.full((out_size, out_size), np.nan, np.float32))
                valid_ratios.append(0.0)
                used.append("fewpts")
                subst.append(1)
                continue
            grid = griddata((x[ok], y[ok]), tb[ok], (gx, gy), method="linear")
            holes = ~np.isfinite(grid)
            if holes.any():
                near = griddata((x[ok], y[ok]), tb[ok], (gx, gy), method="nearest")
                grid[holes] = near[holes]
            # 物理下限：被动微波亮温理论上不会低于约 100 K
            # （最强冰散射的 89 GHz 约 100-120 K；再低即为定标残差）。
            # TC PRIMED 源文件里确有 51-55 K 的像素（实测 19-24 个/文件，
            # 占 0.02%），保留会让归一化被极端值带偏，故截断并计入统计。
            lo = np.isfinite(grid) & (grid < TB_FLOOR_K)
            if lo.any():
                n_clip[0] += int(lo.sum())
                grid[lo] = TB_FLOOR_K
            chans.append(grid.astype(np.float32))
            valid_ratios.append(float(np.isfinite(grid).mean()))
            used.append(f"{real_freq:g}{tpol}:{items[0][1]}")
            subst.append(substituted)

        patch = np.stack(chans)
        info = {
            "lat": lat, "lon": lon, "intensity_kt": inten,
            "valid_ratio": min(valid_ratios) if valid_ratios else 0.0,
            "per_channel_valid": valid_ratios,
            "used_channels": used,
            "channel_mask": subst,     # 每通道 0=原始, 1=已被替代/缺失
            "n_clipped": n_clip[0],    # 被抬到 TB_FLOOR_K 的像素数
        }
    return patch, info


def main() -> int:
    ap = argparse.ArgumentParser(description="从 TC PRIMED 提取微波 patch")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--src", default=None,
                    help="TC PRIMED 目录，默认 data/raw/tcprimed")
    ap.add_argument("--manifest", default=None,
                    help="清单，默认 <src>/manifest.csv")
    ap.add_argument("--out", default=None,
                    help="输出 npz，默认 data/processed/sat_patches_mw<s>.npz")
    ap.add_argument("--seasons", nargs="*", type=int, default=None,
                    help="只处理这些年份（默认全部）")
    ap.add_argument("--half-km", type=float, default=128.0, help="半边长（km）")
    ap.add_argument("--size", type=int, default=64, help="输出边长相点数")
    ap.add_argument("--max-dist-km", type=float, default=300.0,
                    help="只保留与本地台风距离小于该值的文件（沿用 manifest 标准）")
    ap.add_argument("--min-valid", type=float, default=0.5,
                    help="有效插值像素比例低于此值则丢弃该帧")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（调试）")
    ap.add_argument("--workers", type=int, default=4, help="并行进程数")
    args = ap.parse_args()

    global _HALF_KM, _SIZE, _MIN_VALID
    _HALF_KM, _SIZE, _MIN_VALID = args.half_km, args.size, args.min_valid

    cfg = load_config(args.config, args.set)
    src = Path(args.src) if args.src else _data_path(cfg, cfg.data.raw_dir) / "tcprimed"
    man = Path(args.manifest) if args.manifest else src / "manifest.csv"

    if not man.exists():
        print(f"[错误] 找不到清单 {man}")
        print("       先运行 scripts/data/fetch_tcprimed.py，或把 manifest.csv 放到该目录")
        return 1
    df = pd.read_csv(man)
    df = df[df["key"].str.startswith("v01r01/final/")]
    df["year"] = df["key"].str.split("/").str[2].astype(int)
    if args.seasons:
        df = df[df["year"].isin(args.seasons)]
    df = df[pd.to_numeric(df["size_mb"], errors="coerce").fillna(0) >= 0]
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    years = sorted(df["year"].unique().tolist())
    print(f"[输入] {len(df)} 个文件, 年份 {years}, 台风 {df['sid'].nunique()}")

    # 输出文件名**自动带年份与尺度**，避免分年运行时互相覆盖。
    # 多年轻则用 2020-2022 形式；只有一年就用该年。
    if len(years) == 1:
        ysuffix = f"_{years[0]}"
    elif len(years) <= 3:
        ysuffix = "_" + "-".join(str(y) for y in years)
    else:
        ysuffix = f"_{years[0]}-{years[-1]}"
    if args.out:
        out = Path(args.out)
    else:
        out = _data_path(cfg, cfg.data.processed_dir) / \
            f"sat_patches_mw{args.size}{ysuffix}.npz"
    print(f"[输出] {out}")

    files_dir = src / "files"
    tasks = []
    for r in df.itertuples(index=False):
        p = files_dir / r.key.split("/")[-1]
        if p.exists():
            tasks.append((p, r.sid, r.atcf_id, r.instrument))
    print(f"[存在] 磁盘上找到 {len(tasks)} 个")
    if not tasks:
        return 1

    keys, patches, metas = [], [], []
    fails: dict[str, int] = {}
    t0 = time.time()
    if args.workers > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            it = ex.map(_work_trampoline, tasks, chunksize=8)
            results = _iter_with_progress(it, len(tasks), t0)
            for res in results:
                _consume(res, keys, patches, metas, fails)
    else:
        for i, t in enumerate(tasks, 1):
            _consume(_work_trampoline(t), keys, patches, metas, fails)
            if i % 200 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  {i}/{len(tasks)}  已用 {el / 60:.1f} 分  "
                      f"预计还需 {el / i * (len(tasks) - i) / 60:.1f} 分", flush=True)

    if not keys:
        print("[错误] 没有成功提取任何帧")
        if fails:
            print("[失败统计]", fails)
        return 1
    arr = np.stack(patches).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    # channel_mask (N, 4) uint8：0=该通道是原始观测，1=缺失或已用另一极化替代。
    # 训练端据此决定丢弃通道或作为可信度输入；避免"36.5V 里混着 H"却无标记。
    cmask = np.array([m.get("channel_mask", [0, 0, 0, 0]) for m in metas],
                     dtype=np.uint8)
    np.savez_compressed(out, keys=np.array(keys), patches=arr,
                        channel_mask=cmask,
                        meta=np.array([json.dumps(m, ensure_ascii=False) for m in metas]))
    n_sub = int((cmask == 1).sum())
    print(f"[mask] 有替代/缺失的通道共 {n_sub} 个 "
          f"({n_sub / cmask.size * 100:.1f}% of {cmask.size})")
    print(f"\n[完成] {len(keys)} 帧, 形状 {arr.shape}, "
          f"{out.stat().st_size / 1e6:.1f} MB -> {out}")
    v = arr[np.isfinite(arr)]
    if v.size:
        print(f"[值域] {v.min():.1f} - {v.max():.1f} K, 中位 {np.median(v):.1f} K, "
              f"NaN {np.isnan(arr).mean() * 100:.2f}%")
    if fails:
        print(f"[失败] {sum(fails.values())} 帧: {fails}")
    print(f"[耗时] {(time.time() - t0) / 60:.1f} 分钟")
    return 0


# 模块级函数，供 ProcessPoolExecutor pickle
_HALF_KM = 128.0
_SIZE = 64
_MIN_VALID = 0.5


def _work_trampoline(t):
    import re
    path, sid, atcf, instr = t
    try:
        patch, info = extract_patch_mw(path, _HALF_KM, _SIZE, _MIN_VALID)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", None
    if info["valid_ratio"] < _MIN_VALID:
        return None, "low_valid", None
    m = re.search(r"_(\d{14})\.nc$", path.name)
    s = m.group(1)
    ts = f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}"
    return (f"{sid}|{ts}", patch,
            {**info, "sid": sid, "atcf": atcf, "instrument": instr, "file": path.name})


def _consume(res, keys, patches, metas, fails):
    """res 为 (key, patch, info)，失败时为 (None, 错误串, None)。"""
    key = res[0]
    if key is None:
        err = str(res[1])
        name = err.split(":")[0] if ":" in err else err
        fails[name] = fails.get(name, 0) + 1
        return
    keys.append(key)
    patches.append(res[1])
    metas.append(res[2])


def _iter_with_progress(it, n, t0):
    for i, res in enumerate(it, 1):
        if i % 200 == 0 or i == n:
            el = time.time() - t0
            print(f"  {i}/{n}  已用 {el / 60:.1f} 分  "
                  f"预计还需 {el / i * (n - i) / 60:.1f} 分", flush=True)
        yield res


if __name__ == "__main__":
    raise SystemExit(main())
