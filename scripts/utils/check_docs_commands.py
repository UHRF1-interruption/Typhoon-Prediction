"""校验文档里的命令是否与脚本实际接受的参数一致。

为什么需要
----------
本项目已多次出现**文档与代码脱节**（都是真实事故）：

  · 文档写 `train_ri.py --threshold 10`，但脚本没有 `--threshold`，
    正确写法是 `--set satellite.ri.threshold_ms=10` —— 用户照着跑直接报错。
  · 文档写 `train_ri.py --cv 10 --threshold 10`，漏了 `--compare`，
    而当时不加 `--compare` 会因 `str(rc_cfg.model)` 拿到整个配置节点而崩溃。
  · 文档写 `fetch_tcprimed.py --seasons ...`，但参数名曾经改过。

人工检查靠不住，本脚本把文档里的命令抽出来，逐个与 `脚本 --help` 的
真实参数表比对。

做法
----
1. 扫描 README.md / docs/*.md / outputs/*.md 的代码块与命令行
2. 拼接反斜杠续行，跳过注释行与占位符行
3. 对每个 `scripts/**.py` 命令，用正则抽出所有 `--flag`（去掉 =值）
4. 运行 `python <script> --help`，解析 "options:" 段里定义的参数
5. 报告：不存在的参数、找不到的脚本

局限（诚实说明）
----------------
* 只校验**参数名是否存在**，不校验取值是否合理、参数组合是否合法
* `--set key=value` 里的 key 不校验（那是配置项，与 argparse 无关）
* 少数脚本没有 argparse（如 check_gpu.py 直接自检），会被标为 skipped
* shell 变量、`<哈希>` 之类占位符只做跳过，不展开

用法:
    python scripts/utils/check_docs_commands.py
    python scripts/utils/check_docs_commands.py --strict   # 有告警即非零退出
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ["README.md", "outputs/INDEX.md"] + \
       [str(p.relative_to(ROOT)) for p in sorted((ROOT / "docs").glob("*.md"))]

# 从 --help 里认参数：形如 "  -h, --help" 或 "  --config CONFIG" 或 "  --set [SET ...]"
_OPT_LINE = re.compile(r"^\s{1,8}(?:-\w,\s*)?(--[A-Za-z][\w-]*)")
# 从命令里抽参数：--flag 或 --flag=value
_TOKEN_FLAG = re.compile(r"--([A-Za-z][\w-]*)")


def _help_flags(script: Path) -> set[str] | None:
    """返回脚本接受的 --flag 集合；不支持 --help 时返回 None。"""
    try:
        r = subprocess.run([sys.executable, str(script), "--help"],
                           capture_output=True, text=True, timeout=120,
                           cwd=str(ROOT))
    except (subprocess.TimeoutExpired, OSError):
        return None
    out = (r.stdout or "") + (r.stderr or "")
    if "usage:" not in out:
        return None
    flags = set()
    for line in out.splitlines():
        m = _OPT_LINE.match(line)
        if m:
            flags.add(m.group(1))
    return flags or None


def _iter_commands(text: str):
    """产出 (行号, 命令字符串)。处理反斜杠续行、跳过注释。"""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        start = i + 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            i += 1
            continue
        cmd = stripped
        # 续行：当前行以 \ 结尾则拼接下一行（最多 6 行，防跑飞）
        k = 0
        while cmd.endswith("\\") and i + 1 < len(lines) and k < 6:
            cmd = cmd[:-1].rstrip() + " " + lines[i + 1].strip()
            i += 1
            k += 1
        yield start, cmd
        i += 1


def _scripts_in(cmd: str) -> list[str]:
    return re.findall(r"(scripts/[\w/]+\.py)", cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="校验文档命令与脚本参数是否一致")
    ap.add_argument("--strict", action="store_true", help="有告警则以非零退出")
    args = ap.parse_args()

    cache: dict[str, set[str] | None] = {}
    problems: list[str] = []
    skipped: list[str] = []
    n_cmd = n_checked = 0

    for doc in DOCS:
        p = ROOT / doc
        if not p.exists():
            continue
        for lineno, cmd in _iter_commands(p.read_text(encoding="utf-8")):
            scripts = _scripts_in(cmd)
            if not scripts:
                continue
            n_cmd += 1
            tokens = {f"--{m}" for m in _TOKEN_FLAG.findall(cmd)}
            for sp in scripts:
                spath = ROOT / sp
                if not spath.exists():
                    problems.append(f"{doc}:{lineno}  脚本不存在: {sp}")
                    continue
                if sp not in cache:
                    cache[sp] = _help_flags(spath)
                allowed = cache[sp]
                if allowed is None:
                    skipped.append(f"{sp}（无 --help）")
                    continue
                n_checked += 1
                unknown = sorted(tokens - allowed)
                if unknown:
                    problems.append(
                        f"{doc}:{lineno}  {sp} 不接受: {', '.join(unknown)}"
                        f"\n{'':>10}该脚本可接受: {', '.join(sorted(allowed))}")

    print(f"[扫描] {n_cmd} 条含脚本路径的命令，校验了 {n_checked} 次")
    if skipped:
        print(f"[跳过] {len(set(skipped))} 个脚本没有 --help："
              f"{', '.join(sorted(set(skipped)))}")
    if problems:
        print(f"\n[发现 {len(problems)} 处不一致]")
        for x in problems:
            print(f"  ✗ {x}")
        return 1 if args.strict else 0
    print("\n[通过] 文档里的参数名与脚本一致 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
