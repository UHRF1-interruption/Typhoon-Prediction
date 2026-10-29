"""命令行入口: 从卫星数据包提取逐时次标量表。

产出 `sat_scalars.parquet`，是强度 / RI 任务的输入之一。
每行对应压缩包里的一个 NetCDF 文件；默认对同一 (台风, 时次) 只保留
优先级最高的卫星，与 build_sat_patches.py 的规则严格一致。

用法:
    python scripts/data/build_sat_scalars.py
    python scripts/data/build_sat_scalars.py --limit 100       # 调试
    python scripts/data/build_sat_scalars.py --out /abs/x.parquet

输出列: tcid, name, storm_key, time, satellite, minute,
        cent_lat, cent_lon, cent_prs, wind_spd, vza
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="提取卫星标量表")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None,
                    help="输出 parquet（默认 <processed_dir>/sat_scalars.parquet）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（调试）")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false",
                    help="保留同一时次的所有卫星（默认只保留优先级最高的一颗）")
    ap.add_argument("--sat-priority", default="FY4-B,FY4-A,FY2-H,FY2-G,FY2-F,FY2-E,FY2-C",
                    help="卫星优先级（逗号分隔，靠前优先）")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    raw_dir = resolve_path(cfg, cfg.data.raw_dir, ROOT) / "bsts"
    out_path = (Path(args.out) if args.out
                else resolve_path(cfg, cfg.data.processed_dir, ROOT) / "sat_scalars.parquet")

    from src.data.fysatty import extract_scalars, list_satellite_files

    zip_list = raw_dir / "zip_list.txt"
    if zip_list.exists():
        paths = [l.strip() for l in zip_list.read_text().splitlines() if l.strip()]
        paths = [p for p in paths if Path(p).exists()]
    else:
        zip_dir = resolve_path(cfg, cfg.satellite.zip_dir, ROOT)
        paths = sorted(str(p) for p in zip_dir.glob("*.zip"))
    if not paths:
        print(f"[scalars] 没有找到数据包（查了 {raw_dir} 与 {cfg.satellite.zip_dir}）")
        return 1

    print(f"[scalars] 扫描 {len(paths)} 个压缩包的目录索引 ...", flush=True)
    files = list_satellite_files(paths)

    if args.dedup:
        # 同一 (台风, 时次) 可能有多颗卫星观测。只保留优先级最高的一颗：
        #   * 与 build_sat_patches.py 一致，保证标量表与云图 patch 严格对齐
        #   * 文件数约减到 1/3
        order = [x.strip() for x in args.sat_priority.split(",") if x.strip()]
        prio = {name: i for i, name in enumerate(order)}
        files = files.copy()
        files["_prio"] = files["satellite"].map(prio).fillna(len(prio)).astype(int)
        before = len(files)
        files = (files.sort_values(["storm_key", "time", "_prio"])
                 .drop_duplicates(subset=["storm_key", "time"], keep="first")
                 .drop(columns="_prio").reset_index(drop=True))
        print(f"[scalars] 去重: {before} -> {len(files)} 个文件"
              f"（每时次保留优先级最高的卫星）", flush=True)
    if args.limit:
        files = files.iloc[: args.limit]
    print(f"[scalars] 待提取 {len(files)} 个文件", flush=True)

    rows, fail_detail = [], []
    t0 = time.time()
    for i, row in enumerate(files.itertuples(index=False), 1):
        try:
            sc = extract_scalars(row.zip_path, row.member)
            rows.append({
                "tcid": row.tcid, "name": row.name, "storm_key": row.storm_key,
                "time": row.time, "satellite": row.satellite, "minute": int(row.minute),
                "cent_lat": sc["cent_lat"], "cent_lon": sc["cent_lon"],
                "cent_prs": sc["cent_prs"], "wind_spd": sc["wind_spd"],
                "vza": sc["vza"],
            })
        except Exception as exc:
            fn = str(row.member).split("/")[-1]
            fail_detail.append((fn, type(exc).__name__, str(exc)[:120]))
            if len(fail_detail) <= 10:
                print(f"[scalars] 失败 {fn[:60]}: {type(exc).__name__}", flush=True)
        if i % 2000 == 0 or i == len(files):
            el = time.time() - t0
            eta = el / i * (len(files) - i)
            print(f"[scalars] {i}/{len(files)}  已用 {el/60:.1f} 分钟, "
                  f"预计还需 {eta/60:.1f} 分钟", flush=True)

    if not rows:
        print("[scalars] 没有成功提取任何记录")
        return 1

    df = pd.DataFrame(rows).sort_values(["storm_key", "time"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\n[scalars] 完成: {len(df)} 条, {df['storm_key'].nunique()} 台风, "
          f"失败 {len(fail_detail)}")
    print(f"[scalars] 时间范围 {df['time'].min()} ~ {df['time'].max()}")
    print(f"[scalars] 卫星分布 {dict(df['satellite'].value_counts())}")
    print(f"[scalars] 已保存 -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")
    print(f"[scalars] 总耗时 {(time.time()-t0)/60:.1f} 分钟")

    if fail_detail:
        types = Counter(t for _, t, _ in fail_detail)
        print(f"\n[scalars] 失败类型统计: {dict(types)}")
        print("[scalars] 失败明细:")
        for fn, et, msg in fail_detail[:20]:
            print(f"  {fn[:72]}")
            print(f"    {et}: {msg[:90]}")
        if len(fail_detail) > 20:
            print(f"  ... 其余 {len(fail_detail)-20} 个省略")
        if types.get("CorruptFileError"):
            print("\n  提示: CorruptFileError 表示源文件 HDF5 元数据损坏，"
                  "数据不可恢复，只能跳过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
