"""命令行入口: 下载 TC PRIMED 被动微波过境文件（支持断点续传）。

TC PRIMED（Razin et al. 2023）把 GMI / AMSR2 / SSMIS / ATMS / MHS / AMSU-B / TMI
的被动微波亮温做了交叉定标，并按台风切好、重网格到以台风中心为原点的规则网格，
每个文件另带 GPROF 降水和红外场。是"微波 → 强度/RI"路线最省事的数据源。

为什么单独写这个脚本
--------------------
官方 tcprimedapi 本身可用，但它把整段输出缓冲到内存、且没有断点续传；
本脚本用纯标准库直连 S3 公开桶，支持 Range 续传与状态持久化，
适合几十 GB 的长时间下载。

匹配逻辑（踩过坑，勿简化）
--------------------------
TC PRIMED 的编号（WP032020）与 IBTrACS 的 sid（2020213N15131）**没有对应关系**，
且 TC PRIMED 收录 10-15 kt 的弱扰动（IBTrACS 只到 35 kt），
同期多个系统的时间范围还会互相重叠。因此：
  1. 先下载每个系统的少量样本文件；
  2. 用文件内 overpass_storm_metadata 的真实经纬度，与本地轨迹插值位置比对；
  3. 多票表决确定该系统对应哪个本地台风；
  4. 匹配不上的（多为弱扰动）直接丢弃。

用法:
    # 先看会下载什么（不实际下载）
    python scripts/data/fetch_tcprimed.py --seasons 2022 --dry-run

    # 正式下载（中断后重跑同一条命令即自动续传）
    python scripts/data/fetch_tcprimed.py --seasons 2022 2023 2024 --min-intensity 64

    # 只处理指定台风
    python scripts/data/fetch_tcprimed.py --seasons 2022 --sids 2022123N10140

输出: data/raw/tcprimed/
    files/*.nc      过境文件
    manifest.csv    每个文件一行：本地 sid / 时间 / atcf 编号 / 仪器 / 大小
    state.json      断点续传状态（done / bad）
    samples.log, all.log

依赖: pandas, pyarrow, netCDF4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUCKET = "https://noaa-nesdis-tcprimed-pds.s3.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) tcprimed-fetch/1.1"

DEFAULT_INSTRUMENTS = ("AMSR2", "GMI", "SSMIS")

ENV_RE = re.compile(
    r"TCPRIMED_v\d+r\d+-final_(?P<atcf>[A-Z]{2}\d{2}\d{4})_env_"
    r"s(?P<s>\d{14})_e(?P<e>\d{14})\.nc$"
)
OVERPASS_RE = re.compile(
    r"TCPRIMED_v\d+r\d+-final_(?P<atcf>[A-Z]{2}\d{2}\d{4})_"
    r"(?P<instr>[A-Z0-9-]+)_(?P<plat>[A-Z0-9]+)_\d+_(?P<time>\d{14})\.nc$"
)

def _data_path(cfg, value):
    """把配置路径解析成绝对路径，支持 TYPHOON_DATA_ROOT 挂载外部数据。"""
    from src.config import resolve_path
    return resolve_path(cfg, value, ROOT)


_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# HTTP / S3
# --------------------------------------------------------------------------
def _open(url: str, timeout: int = 120, extra: dict | None = None):
    headers = {"User-Agent": UA}
    if extra:
        headers.update(extra)
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                  timeout=timeout)


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """列出前缀下所有对象 (key, size)，自动翻页。"""
    out: list[tuple[str, int]] = []
    token = None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        with _open(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{S3_NS}Contents"):
            k = c.findtext(f"{S3_NS}Key") or ""
            s = int(c.findtext(f"{S3_NS}Size") or 0)
            if s > 0:
                out.append((k, s))
        if (root.findtext(f"{S3_NS}IsTruncated") or "").lower() != "true":
            break
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not token:
            break
    return out


def parse_key(key: str):
    m = OVERPASS_RE.search(key)
    if not m:
        return None
    ts = time.strptime(m.group("time"), "%Y%m%d%H%M%S")
    return {
        "atcf_id": m.group("atcf"),
        "instrument": m.group("instr"),
        "platform": m.group("plat"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", ts),
    }


# --------------------------------------------------------------------------
# 读取过境文件的台风坐标（用于匹配）
# --------------------------------------------------------------------------
def read_storm_pos(path: Path):
    """返回 (lat, lon, intensity_kt)，读不到返回 None。"""
    try:
        import netCDF4 as nc
        with nc.Dataset(path) as ds:
            st = ds.groups["overpass_storm_metadata"]
            lat = float(st.variables["storm_latitude"][0])
            lon = float(st.variables["storm_longitude"][0])
            it = st.variables.get("intensity")
            inten = float(it[0]) if it is not None else float("nan")
            return lat, lon, inten
    except Exception:
        return None


# --------------------------------------------------------------------------
# 断点续传下载
# --------------------------------------------------------------------------
def download_resumable(key: str, size: int, dest: Path, retries: int = 5,
                       backoff: float = 3.0) -> str:
    """带 Range 续传的单文件下载。

    返回 "ok" / "skip" / "fail: xxx"。
    .part 文件在同一次运行被中断后仍保留，下次从已有字节继续。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and dest.stat().st_size == size:
        return "skip"

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:                      # .part 反而更大，丢弃重来
            part.unlink()
            have = 0
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            url = f"{BUCKET}/{urllib.parse.quote(key)}"
            with _open(url, extra=headers, timeout=120) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            got = part.stat().st_size
            if got != size:
                raise IOError(f"大小不符 {got} != {size}")
            part.replace(dest)
            return "ok"
        except Exception as e:  # noqa: BLE001
            # 注意：这里**不删** .part，保留以便下次续传
            if attempt == retries:
                return f"fail: {type(e).__name__}: {e}"
            time.sleep(backoff * attempt)
    return "fail"


# --------------------------------------------------------------------------
# 状态文件（断点续传的第二重保障）
# --------------------------------------------------------------------------
class State:
    """断点续传状态。path=None 表示只放内存（dry-run 用，不写盘）。"""

    def __init__(self, path: Path | None):
        self.path = path
        self.data = {"done": {}, "bad": {}}
        if path is not None and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        self.data.setdefault("done", {})
        self.data.setdefault("bad", {})
        self._lock = threading.Lock()

    def mark_done(self, key: str, size: int) -> None:
        with self._lock:
            self.data["done"][key] = size

    def mark_bad(self, key: str, reason: str) -> None:
        with self._lock:
            self.data["bad"][key] = reason

    def is_done(self, key: str, size: int) -> bool:
        return self.data["done"].get(key) == size

    def save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)


# --------------------------------------------------------------------------
# 匹配
# --------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def build_matches(rows: list[dict], tracks, files_dir: Path,
                  tol_km: float, sample_n: int) -> list[dict]:
    """确定每个 atcf 系统对应哪个本地台风。

    实测教训：TC PRIMED 的系统生命史**长于** IBTrACS 的台风生命史
    （收录 15 kt 的扰动阶段）。例如 2020 年 WP03 系统从 7/31 开始，
    而对应本地台风 2020213N15131 从 7/30 开始、7/31 已北移到 52°N，
    系统早期样本与本地轨迹相距数千公里。只用一个早期样本会判错。

    做法：对每个系统，用它已有的**全部**样本文件各自做位置比对，
    统计每个本地台风被命中的次数，取票数最多且距离达标者。
    """
    import numpy as np
    import pandas as pd

    tr = tracks.sort_values(["sid", "t"])
    per_sid = {sid: g for sid, g in tr.groupby("sid")}

    by_atcf: dict[str, list[dict]] = {}
    for r in rows:
        by_atcf.setdefault(r["atcf_id"], []).append(r)

    assign: dict[str, str | None] = {}
    log(f"[匹配] {len(by_atcf)} 个系统，用已下载的样本做位置投票 ...")
    for atcf, rs in sorted(by_atcf.items()):
        rs.sort(key=lambda r: r["utc"])
        votes: dict[str, int] = {}
        dists: dict[str, list[float]] = {}
        checked = 0
        for r in rs[: max(sample_n, 1) * 4]:      # 多试几个，覆盖系统演变
            f = files_dir / r["key"].split("/")[-1]
            if not f.exists():
                continue
            pos = read_storm_pos(f)
            if pos is None:
                continue
            lat, lon, _ = pos
            t = pd.Timestamp(r["utc"])
            best_sid, best_d = None, float("inf")
            # 时间宽松到 ±48h：TC PRIMED 收录扰动阶段，观测常早于 IBTrACS
            # 起点几小时（实测 WP062020 样本早 9h，位置差 < 0.1°）。
            # 端点处用线性外推坐标，而不是夹到端点值，否则距离被系统性放大。
            slack_h = 48
            for sid, g in per_sid.items():
                if t < g["t"].iloc[0] - pd.Timedelta(hours=slack_h):
                    continue
                if t > g["t"].iloc[-1] + pd.Timedelta(hours=slack_h):
                    continue
                # 统一用 Unix 秒。坑很多，踩过两次：
                #   pandas 3.x 的 datetime64 默认精度是 **微秒**（不是纳秒），
                #   所以 astype("int64") 得到的是微秒；若再除以 1e9 当纳秒，
                #   时间会小 1000 倍，外推坐标飞出地球（表现为"全部匹配失败"）。
                # 最稳的做法：逐个 ts.timestamp()，不依赖内部存储单位。
                tt = np.array([x.timestamp() for x in g["t"]], dtype=float)
                ot = t.timestamp()
                if len(tt) >= 2:
                    vlat = np.gradient(g["lat"].to_numpy(), tt)
                    vlon = np.gradient(g["lon"].to_numpy(), tt)
                    if ot < tt[0]:
                        glat = g["lat"].iloc[0] + vlat[0] * (ot - tt[0])
                        glon = g["lon"].iloc[0] + vlon[0] * (ot - tt[0])
                    elif ot > tt[-1]:
                        glat = g["lat"].iloc[-1] + vlat[-1] * (ot - tt[-1])
                        glon = g["lon"].iloc[-1] + vlon[-1] * (ot - tt[-1])
                    else:
                        glat = float(np.interp(ot, tt, g["lat"].to_numpy()))
                        glon = float(np.interp(ot, tt, g["lon"].to_numpy()))
                else:
                    continue
                if glon < 0:
                    glon += 360
                olon = lon if lon >= 0 else lon + 360
                d = haversine_km(lat, olon, glat, glon)
                if d < best_d:
                    best_d, best_sid = d, sid
            if best_sid is None:
                continue
            checked += 1
            if best_d < tol_km:
                votes[best_sid] = votes.get(best_sid, 0) + 1
                dists.setdefault(best_sid, []).append(best_d)
        if votes:
            sid = max(votes, key=lambda k: votes[k])
            med = float(np.median(dists[sid]))
            assign[atcf] = sid
            if checked:
                log(f"    {atcf} -> {sid}  ({votes[sid]}/{checked} 命中, 中位距离 {med:.0f} km)")
        else:
            assign[atcf] = None
            if checked:
                log(f"    {atcf} -> 未匹配（{checked} 个样本都超出 {tol_km:.0f} km）")

    out = []
    for r in rows:
        sid = assign.get(r["atcf_id"])
        if sid:
            out.append({**r, "sid": sid})
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="按本地轨迹下载 TC PRIMED 过境文件（支持断点续传）")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tracks", default=None,
                    help="本地轨迹 parquet，默认 data/raw/tracks_WP_clean.parquet")
    ap.add_argument("--out", default=None,
                    help="输出目录，默认 data/raw/tcprimed")
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--basin", default="WP")
    ap.add_argument("--min-intensity", type=float, default=0.0,
                    help="只保留本地峰值风速 >= 该值(kt)的台风（建议 64 排除弱系统）")
    ap.add_argument("--instruments", nargs="+", default=list(DEFAULT_INSTRUMENTS))
    ap.add_argument("--tol-km", type=float, default=500.0, help="位置匹配容差")
    ap.add_argument("--match-samples", type=int, default=5,
                    help="每个系统最多下载几个文件用于位置投票（默认 5，太少先下再看结果）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sids", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="只列清单，不下载")
    ap.add_argument("--no-resume", action="store_true", help="忽略 state，强制校验全部文件")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        log("[错误] 需要 pandas。请运行: python -m pip install pandas pyarrow netCDF4")
        return 1

    if args.tracks is None or args.out is None:
        from src.config import load_config
        cfg = load_config(args.config, args.set)
        raw_dir = _data_path(cfg, cfg.data.raw_dir)
        if args.tracks is None:
            args.tracks = str(raw_dir / "tracks_WP_clean.parquet")
        if args.out is None:
            args.out = str(raw_dir / "tcprimed")
    log(f"[路径] 轨迹 {args.tracks}")
    log(f"[路径] 输出 {args.out}")
    if not Path(args.tracks).exists():
        log(f"[错误] 找不到轨迹文件 {args.tracks}")
        return 1

    out = Path(args.out)
    files_dir = out / "files"
    if args.dry_run:
        # dry-run 不写任何文件；同时提前检查目录可写，避免跑到一半才发现
        if not out.exists():
            parent = out.parent
            if not parent.exists() or not os.access(parent, os.W_OK):
                log(f"[注意] {parent} 不可写，正式下载前需要先挂载/授权")
    else:
        out.mkdir(parents=True, exist_ok=True)
        files_dir.mkdir(parents=True, exist_ok=True)
    state = State(out / "state.json" if not args.dry_run else None)
    if args.no_resume:
        state.data = {"done": {}, "bad": {}}

    # ---- 1. 本地轨迹 ----
    tr = pd.read_parquet(args.tracks)
    for col in ("sid", "season", "lat", "lon", "time"):
        if col not in tr.columns:
            log(f"[错误] 轨迹缺少列: {col}")
            return 1
    tr["t"] = pd.to_datetime(tr["time"], utc=True)
    if args.sids:
        tr = tr[tr["sid"].isin(args.sids)]
    if args.min_intensity > 0 and "wind" in tr.columns:
        peak = tr.groupby("sid")["wind"].max()
        tr = tr[tr["sid"].isin(peak[peak >= args.min_intensity].index)]
    seasons = sorted(set(args.seasons))
    tr = tr[tr["season"].isin(seasons)]
    if tr.empty:
        log("[错误] 过滤后没有台风")
        return 1
    log(f"[本地] {tr['sid'].nunique()} 个台风 / {len(tr)} 点 / 年份 {seasons}")

    # ---- 2. 列文件 ----
    log("[S3] 列出 TC PRIMED 文件 ...")
    rows: list[dict] = []
    for y in seasons:
        keys = list_keys(f"v01r01/final/{y}/{args.basin}/")
        n0 = len(rows)
        for k, size in keys:
            p = parse_key(k)
            if p is None or p["instrument"] not in args.instruments:
                continue
            rows.append({**p, "key": k, "size": size})
        log(f"    {y}: {len(keys)} 对象, 目标仪器 {len(rows) - n0}")
    if not rows:
        log("[错误] 没有目标文件")
        return 1

    # ---- 3. 先下匹配样本，再决定完整清单 ----
    
    sample_keys = {}
    for r in sorted(rows, key=lambda r: r["utc"]):
        lst = sample_keys.setdefault(r["atcf_id"], [])
        if len(lst) < args.match_samples:
            lst.append(r)
    sample_flat = [r for v in sample_keys.values() for r in v]
    log(f"[匹配样本] 需要 {len(sample_flat)} 个文件做位置校验")

    if not args.dry_run:
        _download_batch(sample_flat, files_dir, state, args.workers, "samples")

    matched = build_matches(rows, tr, files_dir, args.tol_km, args.match_samples)
    if not matched:
        if args.dry_run:
            log("")
            log("[说明] dry-run 未下载样本文件，无法做位置投票，所以匹配结果为空。")
            log(f"       清单预览：{len(rows)} 个候选文件、{len(sample_keys)} 个系统。")
            log("       去掉 --dry-run 跑一次即可得到完整匹配清单与下载。")
            total = sum(r["size"] for r in rows) / 2**30
            log(f"       若全部下载约 {total:.1f} GB。")
            return 0
        log("[错误] 没有任何系统匹配到本地台风")
        return 1
    log(f"[匹配] 命中 {len(matched)} 文件 / {len({r['sid'] for r in matched})} 台风")

    matched.sort(key=lambda r: (r["sid"], r["utc"]))
    if args.max_files:
        matched = matched[: args.max_files]
        log(f"[截断] 只取前 {args.max_files} 个")

    # manifest 按年份分开写，**不能**共用一个文件名：
    # 曾经用 --seasons 分三次跑，每次都覆盖同一个 manifest.csv，
    # 最后只剩最后一次的记录，磁盘上有 9401 个文件而清单只有 3519 行。
    # 现在按年份落盘，并额外维护一份全量合并清单。
    _write_manifest(out, matched, args.seasons)
    total = sum(r["size"] for r in matched) / 2**30
    done = sum(1 for r in matched
               if state.is_done(r["key"], r["size"])
               or (files_dir / r["key"].split("/")[-1]).exists()
               and (files_dir / r["key"].split("/")[-1]).stat().st_size == r["size"])
    log(f"[清单] {man}: {len(matched)} 文件 / {total:.2f} GB（已完成 {done}）")
    if args.dry_run:
        log("[dry-run] 未下载")
        return 0

    # ---- 4. 全量下载（断点续传）----
    _download_batch(matched, files_dir, state, args.workers, "all")
    log(f"[完成] 状态文件 {state.path}")
    bad = len(state.data["bad"])
    if bad:
        log(f"[注意] {bad} 个文件失败，重跑同一命令会继续重试")
        return 2
    return 0


def _download_batch(rows: list[dict], files_dir: Path, state: "State",
                    workers: int, tag: str) -> None:
    todo = []
    for r in rows:
        dest = files_dir / r["key"].split("/")[-1]
        if state.is_done(r["key"], r["size"]):
            continue
        if r["key"] in state.data["bad"]:
            continue
        todo.append((r["key"], r["size"], dest))
    if not todo:
        log(f"[{tag}] 全部已完成，跳过")
        return
    log(f"[{tag}] 待下载 {len(todo)} 个，{workers} 线程")
    ok = skip = fail = 0
    t0 = time.time()
    lock = threading.Lock()

    def work(k, s, d):
        st = download_resumable(k, s, d)
        with lock:
            if st == "ok":
                state.mark_done(k, s)
            elif st == "skip":
                state.mark_done(k, s)
            else:
                state.mark_bad(k, st)
                state.save()
        return st

    with open(files_dir.parent / f"{tag}.log", "a", encoding="utf-8") as logf:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, k, s, d): k for k, s, d in todo}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                st = fut.result()
                logf.write(f"{st}\t{futs[fut]}\n")
                if st.startswith("fail"):
                    fail += 1
                elif st == "ok":
                    ok += 1
                else:
                    skip += 1
                if i % 20 == 0 or i == len(futs):
                    el = time.time() - t0
                    rate = i / el if el else 0
                    eta = (len(futs) - i) / rate if rate else 0
                    log(f"  {i}/{len(futs)}  新下 {ok} 跳过 {skip} 失败 {fail}  "
                        f"已用 {el / 60:.1f} 分  预计还需 {eta / 60:.1f} 分")
        state.save()
    log(f"[{tag}] 完成：新下 {ok}，跳过 {skip}，失败 {fail}，{(time.time() - t0) / 60:.1f} 分钟")


MANIFEST_FIELDS = ["sid", "utc", "atcf_id", "instrument", "platform", "size_mb", "key"]


def _write_manifest(out: Path, rows: list[dict], seasons: list[int]) -> None:
    """按年份分别写 manifest_<year>.csv，并合并成全量 manifest.csv。"""
    by_year: dict[str, list[dict]] = {}
    for r in rows:
        y = r["key"].split("/")[2]        # .../final/<year>/WP/...
        by_year.setdefault(y, []).append(r)
    for y, rs in by_year.items():
        p = out / f"manifest_{y}.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for r in sorted(rs, key=lambda x: (x["sid"], x["utc"])):
                w.writerow({"sid": r["sid"], "utc": r["utc"], "atcf_id": r["atcf_id"],
                            "instrument": r["instrument"], "platform": r["platform"],
                            "size_mb": f"{r['size'] / 2**20:.1f}", "key": r["key"]})
        log(f"[清单] {p.name}: {len(rs)} 行")

    # 合并：读回所有 manifest_*.csv（含以前运行的年份），去重后写 manifest.csv
    merged: dict[str, dict] = {}
    for p in sorted(out.glob("manifest_*.csv")):
        with open(p, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                merged[row["key"]] = row
    man = out / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for k in sorted(merged):
            w.writerow(merged[k])
    log(f"[清单] manifest.csv（全量合并）: {len(merged)} 行")


if __name__ == "__main__":
    sys.exit(main())
