"""命令行入口: 构建环境场并附加到轨迹上。

用法:
    # 无凭据：生成物理合理的模拟环境场（用于打通流程）
    python scripts/build_environment.py --source synthetic

    # 有 ERA5 凭据：下载真实数据并提取
    python scripts/build_environment.py --source era5 --years 2019 2020 2021

    # 只检查数据源可用性
    python scripts/build_environment.py --check

输出: data/processed/env_{source}.parquet
      包含 time/sid/lat/lon + ENV_FEATURES 各列
"""
from __future__ import annotations

import argparse
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

from src.config import load_config  # noqa: E402
from src.data.environment import ENV_FEATURES, synthesize_environment  # noqa: E402


def check_sources() -> int:
    """检查 ERA5 数据源与凭据可用性。"""
    import socket
    print("=" * 60)
    print("ERA5 数据源检查")
    print("=" * 60)

    for host in ["cds.climate.copernicus.eu", "data.ecmwf.int"]:
        try:
            socket.create_connection((host, 443), timeout=8).close()
            print(f"  网络  {host}: 可连接")
        except Exception as exc:
            print(f"  网络  {host}: 不可达 ({type(exc).__name__})")

    try:
        import cdsapi
        print(f"  cdsapi: 已安装 {getattr(cdsapi, '__version__', 'unknown')}")
    except ImportError:
        print("  cdsapi: 未安装")
        print("         安装: .venv/bin/python -m pip install "
              "-i https://mirrors.cernet.edu.cn/pypi/web/simple cdsapi")

    rc = Path.home() / ".cdsapirc"
    if rc.exists():
        txt = rc.read_text()
        has_key = "key:" in txt and "url:" in txt
        print(f"  凭据 ~/.cdsapirc: {'已配置' if has_key else '格式不完整'}")
    else:
        print("  凭据 ~/.cdsapirc: 未配置")
        print("         需在 https://cds.climate.copernicus.eu 注册并同意 ERA5 许可，")
        print("         然后把 URL 和 API key 写入 ~/.cdsapirc")
        print("         没有凭据时可先用 --source synthetic 打通流程")
    print("=" * 60)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="构建台风环境场数据")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--source", choices=["synthetic", "era5"], default="synthetic")
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="只检查数据源")
    args = ap.parse_args()

    if args.check:
        return check_sources()

    cfg = load_config(args.config, args.set)
    tracks_path = _data_path(cfg, cfg.data.raw_dir) / f"tracks_{cfg.data.basin}_clean.parquet"
    if not tracks_path.exists():
        print(f"[env] 缺少轨迹数据 {tracks_path}，请先运行 prepare_data.py")
        return 1
    tracks = pd.read_parquet(tracks_path)
    print(f"[env] 轨迹 {len(tracks)} 条, {tracks['sid'].nunique()} 个台风")

    if args.source == "synthetic":
        print("[env] 使用物理模拟环境场（规则生成，用于验证管线）")
        frames = []
        for i, (sid, g) in enumerate(tracks.groupby("sid")):
            env = synthesize_environment(g, seed=int(cfg.seed) + i)
            env["sid"] = sid
            frames.append(env)
        env_df = pd.concat(frames, ignore_index=True)
        out = _data_path(cfg, cfg.data.processed_dir) / "env_synthetic.parquet"
        env_df.to_parquet(out, index=False)
        print(f"[env] 已保存 {len(env_df)} 行 -> {out}")
        print("[env] 环境特征:", ", ".join(ENV_FEATURES))
        print("\n[env] 各特征统计:")
        for c in ENV_FEATURES:
            col = env_df[c]
            print(f"  {c:16s} mean={col.mean():8.2f} std={col.std():8.2f} "
                  f"range=[{col.min():.2f}, {col.max():.2f}]")
        return 0

    # ERA5 真实数据
    from src.data.environment import ERA5Request, extract_point_env, fetch_era5

    years = args.years or sorted(tracks["time"].dt.year.unique().tolist())
    months = sorted(tracks["time"].dt.month.unique().tolist())
    print(f"[env] 下载 ERA5: 年份 {years}, 月份 {months}")
    files = fetch_era5(ERA5Request(out_dir=_data_path(cfg, cfg.data.raw_dir) / "era5",
                                   years=[int(y) for y in years],
                                   months=[int(m) for m in months]))
    print(f"[env] 获得 {len(files)} 个文件")

    # 按月批量提取
    import xarray as xr

    rows = []
    tracks["time"] = pd.to_datetime(tracks["time"])
    for year in years:
        for month in months:
            pl_path = _data_path(cfg, cfg.data.raw_dir) / "era5" / f"era5_pl_{year}{month:02d}.nc"
            sl_path = _data_path(cfg, cfg.data.raw_dir) / "era5" / f"era5_sl_{year}{month:02d}.nc"
            if not pl_path.exists():
                continue
            ds_pl = xr.open_dataset(pl_path)
            ds_sl = xr.open_dataset(sl_path) if sl_path.exists() else xr.Dataset()
            sub = tracks[(tracks["time"].dt.year == year) & (tracks["time"].dt.month == month)]
            for row in sub.itertuples(index=False):
                try:
                    ts = pd.Timestamp(row.time)
                    pl_t = ds_pl.sel(time=ts, method="nearest")
                    sl_t = ds_sl.sel(time=ts, method="nearest") if len(ds_sl) else xr.Dataset()
                    vec = extract_point_env(pl_t, sl_t, ts, float(row.lat), float(row.lon))
                    rows.append({"sid": row.sid, "time": ts,
                                 **dict(zip(ENV_FEATURES, vec.tolist()))})
                except Exception:
                    continue
            ds_pl.close()
            if len(ds_sl):
                ds_sl.close()
            print(f"[env] {year}-{month:02d} 提取 {len(sub)} 条轨迹")

    if not rows:
        print("[env] 未提取到任何环境数据")
        return 1
    env_df = pd.DataFrame(rows)
    out = _data_path(cfg, cfg.data.processed_dir) / "env_era5.parquet"
    env_df.to_parquet(out, index=False)
    print(f"[env] 已保存 {len(env_df)} 行 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
