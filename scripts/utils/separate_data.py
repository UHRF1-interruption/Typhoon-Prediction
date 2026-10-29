"""把 data/ 目录从项目里剥离到外部位置（数据与代码分离）。

用法:
    # 预览将要做什么
    python scripts/utils/separate_data.py --target /mnt/datasets/typhoon

    # 实际执行（移动数据，项目内只留一个软链接）
    python scripts/utils/separate_data.py --target /mnt/datasets/typhoon --apply

    # 只移动不建软链接（完全分离，改用环境变量挂载）
    python scripts/utils/separate_data.py --target /mnt/datasets/typhoon --apply --no-symlink

三种使用方式（任选其一，优先级从高到低）:
  1. export TYPHOON_DATA_ROOT=/mnt/datasets/typhoon     # 临时/多数据集切换
  2. 配置文件里写 data_root: /mnt/datasets/typhoon
  3. 保留 <项目>/data 软链接指向外部目录               # 最省事，不改任何命令
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def dir_size(p: Path) -> str:
    try:
        out = shutil.disk_usage(p) if False else None
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except Exception:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f}{unit}"
        total /= 1024
    return f"{total:.1f}PB"


def main() -> int:
    ap = argparse.ArgumentParser(description="把 data/ 剥离到外部位置")
    ap.add_argument("--target", required=True,
                    help="数据要放到的外部目录，例如 /mnt/datasets/typhoon/data")
    ap.add_argument("--apply", action="store_true", help="实际执行（默认只预览）")
    ap.add_argument("--no-symlink", action="store_true",
                    help="不创建 <项目>/data 软链接（完全分离，需用环境变量挂载）")
    ap.add_argument("--copy", action="store_true",
                    help="复制而非移动（保留项目内副本，慎用，占双倍空间）")
    args = ap.parse_args()

    src = ROOT / "data"
    dst = Path(args.target).expanduser().resolve()

    if not src.exists():
        print(f"[错误] 找不到 {src}")
        return 1
    if dst.exists() and any(dst.iterdir()):
        print(f"[错误] 目标目录非空: {dst}")
        print("       请换一个空目录，或先手动清空")
        return 1

    size = dir_size(src)
    action = "复制" if args.copy else "移动"
    print("=" * 66)
    print(f"数据分离计划")
    print("=" * 66)
    print(f"  源目录 : {src}  ({size})")
    print(f"  目标   : {dst}")
    print(f"  操作   : {action}")
    print(f"  软链接 : {'不创建' if args.no_symlink else f'{src} -> {dst}'}")
    print("=" * 66)
    print()
    print("分离后的使用方式:")
    if args.no_symlink:
        print(f"  export TYPHOON_DATA_ROOT={dst}")
        print("  然后在同一终端里运行所有脚本")
    else:
        print("  无需任何改动，原有命令照常使用（软链接自动指向外部数据）")
        print(f"  想切到别的数据集时: export TYPHOON_DATA_ROOT=<另一个目录>")
    print()

    if not args.apply:
        print("这是预览。确认无误后加 --apply 实际执行。")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    if args.copy:
        print(f"[1/3] 复制 {src} -> {dst} ...")
        shutil.copytree(src, dst)
        print(f"[2/3] 复制完成（源目录保留）")
    else:
        print(f"[1/3] 移动 {src} -> {dst} ...")
        try:
            shutil.move(str(src), str(dst))
        except OSError as exc:
            print(f"[错误] 移动失败（可能跨文件系统）: {exc}")
            print("       可以改用 --copy，或先手动 rsync 再删原目录")
            return 1
        print(f"[2/3] 移动完成")

    if not args.no_symlink:
        if src.exists() or src.is_symlink():
            # --copy 模式源目录仍在，需要先移除才能建链接
            if args.copy:
                backup = src.with_name(src.name + "_old")
                if backup.exists():
                    print(f"[警告] {backup.name} 已存在，未删除原 data/ 目录")
                    print(f"       请手动删除 {src} 后重试，或将 {backup} 换名")
                else:
                    src.rename(backup)
                    print(f"[3/4] 原 data/ 已改名为 {backup.name}（确认无误后可删除）")
            else:
                print(f"[警告] {src} 仍存在，跳过软链接")
        if not src.exists() and not src.is_symlink():
            src.symlink_to(dst, target_is_directory=True)
            print(f"[4/4] 已创建软链接: {src} -> {dst}")

    print()
    print("完成。验证一下:")
    print("  .venv/bin/python scripts/data/era5_env.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
