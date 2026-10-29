"""命令行入口: 数据准备（下载 + 预处理）。

用法:
  python scripts/prepare_data.py                              # 下载 WP 子集并预处理
  python scripts/prepare_data.py --set data.source=synthetic  # 离线冒烟测试
  python scripts/prepare_data.py --download-only              # 只下载，可反复运行续传

国内到 NCEI 很慢，默认单次下载时间上限 300 秒。若下载被中断，直接
重新运行同一命令即可——已完成的分块会被跳过（续传）。
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

from src.config import dump_config, load_config  # noqa: E402
from src.data.dataset import preprocess, save_bundle  # noqa: E402
from src.data.ibtracs import DownloadInterrupted, download_ibtracs, load_ibtracs, summarize  # noqa: E402
from src.data.synthetic import generate_synthetic_storms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="下载并预处理台风轨迹数据")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置，如 data.basin=WP")
    ap.add_argument("--download-only", action="store_true", help="只下载不预处理")
    ap.add_argument("--no-time-budget", action="store_true",
                    help="不限单次下载时长（默认 300s），适合放到后台慢慢下")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    raw_dir = _data_path(cfg, cfg.data.raw_dir)
    processed_dir = _data_path(cfg, cfg.data.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("步骤 1/3: 获取原始数据")
    print("=" * 70)
    source = str(cfg.data.source).lower()
    if source == "cma":
        # 本地 CMA 台风最佳路径（无需联网）：直接解析已解压的 CH*SAT.txt
        from src.data.cma import combine_cma_files, summarize_cma
        cma_dir = _data_path(cfg, cfg.data.cma_dir)
        files = sorted(Path(cma_dir).rglob("*SAT.txt"))
        if not files:
            print(f"[cma] 未在 {cma_dir} 找到 *SAT.txt 文件。"
                  f"请先解压 CHSATdata.zip 到该目录。")
            return 1
        df = combine_cma_files(files)
        print("[cma]", summarize_cma(df))
        parquet_path = raw_dir / "tracks_cma_clean.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"[cma] 已保存 {parquet_path}")

        print("=" * 70)
        print("步骤 2/3: 构造滑动窗口样本")
        print("=" * 70)
        bundle, spec = preprocess(cfg, df)
        save_bundle(bundle, spec, processed_dir)
        dump_config(cfg, processed_dir / "used_config.yaml")
        print("\n完成。下一步: python scripts/train.py")
        return 0
    elif source == "synthetic":
        syn = cfg.data.synthetic
        df = generate_synthetic_storms(
            n_storms=int(syn.n_storms),
            n_years=int(syn.n_years),
            start_year=int(syn.get("start_year", 1995)),
            points_per_storm=int(syn.points_per_storm),
            interval_hours=float(cfg.data.interval_hours),
            seed=int(cfg.seed),
        )
        raw_path = raw_dir / "synthetic.parquet"
        df.to_parquet(raw_path, index=False)
        print(f"[synthetic] 已保存 {raw_path}")
    else:
        budget = None if args.no_time_budget else float(cfg.data.get("download_time_budget_s") or 0) or None
        try:
            csv_path = download_ibtracs(
                raw_dir,
                version=str(cfg.data.ibtracs_version),
                base_url=str(cfg.data.ibtracs_base_url),
                suffix=str(cfg.data.ibtracs_suffix),
                workers=int(cfg.data.get("download_workers", 24)),
                time_budget_s=budget,
            )
        except DownloadInterrupted as exc:
            print("\n[download] 下载未完成。%s" % exc)
            print("[download] 请直接重新运行本命令继续下载（不会重传已完成部分）。")
            return 2

        if args.download_only:
            print(f"\n[download] 仅下载模式完成: {csv_path}")
            return 0

        print("=" * 70)
        print("步骤 2/3: 解析与过滤")
        print("=" * 70)
        df = load_ibtracs(csv_path, basin=str(cfg.data.basin),
                          min_wind_knots=float(cfg.data.min_wind_knots))
        print("[load]", summarize(df))
        parquet_path = raw_dir / f"tracks_{cfg.data.basin}_clean.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"[load] 清洗后的长表已保存 {parquet_path}")

    print("=" * 70)
    print("步骤 3/3: 构造滑动窗口样本并拟合标准化参数")
    print("=" * 70)
    bundle, spec = preprocess(cfg, df)
    save_bundle(bundle, spec, processed_dir)

    dump_config(cfg, processed_dir / "used_config.yaml")
    print("\n完成。下一步: python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
