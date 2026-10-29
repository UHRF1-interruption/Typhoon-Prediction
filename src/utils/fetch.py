"""多线程分块下载器。

背景：从国内访问 NCEI 只有约 20 KB/s，且经常中途断开；而 CERNET 镜像
能到 5 MB/s。单连接 TCP 受 RTT/丢包限制，因此这里用"多个 Range 分块并发
下载 + 每块失败重试"的方式提高吞吐，并对不支持 Range 的服务器自动回退到
单连接下载。

用法（命令行）:
    python src/utils/fetch.py <url> <输出路径> [--workers 16] [--chunk 1MB]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) typhoon-transformer/1.0",
    "Accept": "*/*",
}


def _head(url: str, timeout: float) -> tuple[int, bool, str | None]:
    """返回 (文件大小, 是否支持 Range, ETag)。"""
    req = Request(url, method="HEAD", headers=DEFAULT_HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        size = int(resp.headers.get("Content-Length") or 0)
        accept_ranges = (resp.headers.get("Accept-Ranges") or "").lower() == "bytes"
        return size, accept_ranges, resp.headers.get("ETag")


def _remote_size(url: str, timeout: float) -> int:
    """用 Range: bytes=0-0 读取 Content-Range 得到总大小（部分服务器 HEAD 不返回长度）。"""
    headers = {**DEFAULT_HEADERS, "Range": "bytes=0-0"}
    with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
        content_range = resp.headers.get("Content-Range") or ""
        if "/" in content_range:
            try:
                return int(content_range.rsplit("/", 1)[1])
            except ValueError:
                pass
        return int(resp.headers.get("Content-Length") or 0)


def _download_chunk(url: str, start: int, end: int, target: Path, part: Path,
                    timeout: float, retries: int, progress: dict, lock: threading.Lock,
                    idx: int) -> None:
    """下载 [start, end] 区间到临时分块文件，完成后写入目标文件的对应位置。"""
    expected = end - start + 1
    attempt = 0
    while attempt < retries:
        have = part.stat().st_size if part.exists() else 0
        if have >= expected:
            break
        attempt += 1
        headers = dict(DEFAULT_HEADERS)
        headers["Range"] = f"bytes={start + have}-{end}"
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp, part.open("ab") as fh:
                while True:
                    chunk = resp.read(1 << 15)
                    if not chunk:
                        break
                    fh.write(chunk)
                    with lock:
                        progress["done"] = progress["done"] + len(chunk)
        except Exception as exc:  # 网络抖动：记录后重试
            with lock:
                progress["errors"] = progress["errors"] + 1
                if progress["errors"] % 10 == 1:
                    print(f"\n[chunk {idx}] {type(exc).__name__}: {str(exc)[:70]} "
                          f"(已收 {have / 1e6:.1f}MB / {expected / 1e6:.1f}MB, 第 {attempt} 次)", flush=True)
            time.sleep(min(2 * attempt, 10))

    have = part.stat().st_size if part.exists() else 0
    if have < expected:
        raise RuntimeError(f"分块 {idx} 下载不完整: {have}/{expected} bytes")

    # 按偏移写回目标文件
    with target.open("r+b") as dst, part.open("rb") as src:
        dst.seek(start)
        while True:
            buf = src.read(1 << 20)
            if not buf:
                break
            dst.write(buf)
    part.unlink(missing_ok=True)


def fetch(url: str, dest: str | Path, workers: int = 24,
          chunk_size: int = 1 << 20, timeout: float = 20, retries: int = 8,
          time_budget_s: float | None = None) -> Path:
    """把 url 下载到 dest，返回本地路径。已完整下载则跳过。

    流程:
      1. 先检查 dest 是否已存在且大小正确（用户手动放入的文件会被直接复用）；
      2. HEAD/Range 探测文件总大小与是否支持分块；
      3. 支持则多线程按块并发下载，每块失败自动重试并续传；
      4. 块下载完成后校验大小，再原子改名为目标文件。

    续传: 已完成的块保存在 ``<dest>.tmp.parts/``，任何一次中断后重跑都会跳过
    已完成块；``time_budget_s`` 到期时抛出 DownloadInterrupted（已完成块不丢）。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1) 目标文件已存在就直接复用（支持离线 / 用户自己拷贝的文件），
    #    大小校验在拿到远端大小后补做；拿不到远端大小时按"非空即用"。
    if dest.exists() and dest.stat().st_size > 0:
        have = dest.stat().st_size
        print(f"[fetch] 已存在本地文件: {dest.name} ({have / 1e6:.1f}MB)，检查完整性",
              flush=True)
        try:
            remote = _head(url, timeout)[0] or _remote_size(url, timeout)
        except Exception:
            remote = 0
        if remote and have != remote:
            print(f"[fetch] 大小与远端不符（本地 {have / 1e6:.1f}MB / 远端 "
                  f"{remote / 1e6:.1f}MB），将重新下载", flush=True)
            backup = dest.with_suffix(dest.suffix + ".size-mismatch")
            dest.replace(backup)
            print(f"[fetch] 原文件已改名保留: {backup.name}", flush=True)
        else:
            if remote:
                print(f"[fetch] 本地文件完整，跳过下载: {dest}", flush=True)
            else:
                print(f"[fetch] 无法核对远端大小，直接使用本地文件: {dest}", flush=True)
            return dest

    size, ranges_ok = 0, False
    try:
        size, ranges_ok, _etag = _head(url, timeout)
    except Exception as exc:
        print(f"[fetch] HEAD 失败({type(exc).__name__})，改用 Range 探测", flush=True)
    if not size:
        try:
            size = _remote_size(url, timeout)
            ranges_ok = True
        except Exception as exc:
            print(f"[fetch] 大小探测失败({type(exc).__name__})，将单连接下载", flush=True)

    if size:
        print(f"[fetch] 目标 {size / 1e6:.1f}MB, Range支持={ranges_ok}", flush=True)
        if dest.exists() and dest.stat().st_size == size:
            print(f"[fetch] 已完整存在，跳过: {dest}")
            return dest

    if not (ranges_ok and size and workers > 1):
        return _single_stream(url, dest, size, timeout)

    tmp_target = dest.with_suffix(dest.suffix + ".tmp")
    part_dir = tmp_target.with_suffix(".parts")
    part_dir.mkdir(exist_ok=True)
    same_layout = tmp_target.exists() and tmp_target.stat().st_size == size

    if not same_layout:
        for stale in part_dir.glob("*.part"):
            stale.unlink()
        with tmp_target.open("wb") as fh:
            fh.truncate(size)
    else:
        kept = len(list(part_dir.glob("*.part")))
        print(f"[fetch] 续传已有临时文件（已完成 {kept} 块）", flush=True)

    chunks = [(s, min(s + chunk_size - 1, size - 1)) for s in range(0, size, chunk_size)]
    pending = [(i, (a, b)) for i, (a, b) in enumerate(chunks)
               if not ((part_dir / f"{i:05d}.part").exists()
                       and (part_dir / f"{i:05d}.part").stat().st_size == b - a + 1)]
    print(f"[fetch] 分块下载: {len(pending)}/{len(chunks)} 块待下载, 并发 {workers}", flush=True)

    progress = {"done": 0, "errors": 0}
    lock = threading.Lock()
    queue = list(pending)
    queue_lock = threading.Lock()
    stop = threading.Event()
    deadline = (time.time() + time_budget_s) if time_budget_s else None
    timer = None
    if deadline:
        timer = threading.Timer(time_budget_s, stop.set)
        timer.daemon = True
        timer.start()

    def worker() -> None:
        while not stop.is_set():
            with queue_lock:
                if not queue:
                    return
                idx, (start, end) = queue.pop(0)
            part = part_dir / f"{idx:05d}.part"
            try:
                _download_chunk(url, start, end, tmp_target, part, timeout,
                                retries, progress, lock, idx)
            except Exception as exc:
                if stop.is_set():      # 用户中断 / 时间预算用完，属于正常退出
                    return
                print(f"\n[fetch] 分块 {idx} 最终失败: {exc}", flush=True)
                stop.set()
                return

    workers_n = min(workers, max(1, len(pending))) if pending else 1
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers_n)]
    t0 = time.time()
    for th in threads:
        th.start()

    while any(th.is_alive() for th in threads):
        time.sleep(1.0)
        with lock:
            done = progress["done"]
        elapsed = max(time.time() - t0, 1e-6)
        print(f"\r[fetch] 本轮新下载 {done / 1e6:.1f}MB ({done / elapsed / 1024:.0f} KB/s)",
              end="", flush=True)
    print()

    if timer:
        timer.cancel()

    # 每个分块在 _download_chunk 内完成后会立即写回 tmp 并删除分块文件，
    # 因此只要没有残留的分块文件就说明全部完成。
    leftover = list(part_dir.glob("*.part"))
    done_blocks = len(chunks) - len(leftover)

    if stop.is_set() or leftover:
        raise DownloadInterrupted(
            f"下载未完成（{done_blocks}/{len(chunks)} 块），已完成的块已落盘，重跑可续传")

    part_dir.rmdir()
    if tmp_target.stat().st_size != size:
        raise RuntimeError(f"文件大小不符: {tmp_target.stat().st_size} != {size}")
    tmp_target.replace(dest)
    print(f"[fetch] 完成: {dest} ({dest.stat().st_size / 1e6:.1f}MB, {time.time() - t0:.0f}s)",
          flush=True)
    return dest


def _single_stream(url: str, dest: Path, size: int, timeout: float) -> Path:
    """单连接下载（服务器不支持 Range 时的兜底）。"""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    done = tmp.stat().st_size if tmp.exists() else 0
    print(f"[fetch] 单连接下载（从 {done / 1e6:.1f}MB 开始）", flush=True)
    headers = dict(DEFAULT_HEADERS)
    if done:
        headers["Range"] = f"bytes={done}-"
    with urlopen(Request(url, headers=headers), timeout=timeout) as resp, tmp.open("ab") as fh:
        total = int(resp.headers.get("Content-Length") or 0) + done
        while True:
            buf = resp.read(1 << 16)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if total:
                print(f"\r[fetch] {done / 1e6:.1f}/{total / 1e6:.1f}MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="多线程分块下载")
    ap.add_argument("url")
    ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1 << 20, help="分块大小(字节)")
    args = ap.parse_args(argv)
    fetch(args.url, args.output, workers=args.workers, chunk_size=args.chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
