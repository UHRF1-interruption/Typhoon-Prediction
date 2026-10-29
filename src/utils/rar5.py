"""不依赖 unrar 的最小 RAR5 读取器（仅支持 STORE / 简单头结构）。

本机没有 unrar/bsdtar，7z 也不带 RAR5 编解码器，因此这里直接按 RAR5 规范
解析头部并读取文件内容。只用于读取本工程收到的 CMA 数据包，不建议通用化。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

MAGIC = b"\x52\x61\x72\x21\x1a\x07\x01\x00"
HTYPE_NAMES = {1: "MAIN", 2: "FILE", 3: "SERVICE", 4: "ENCRYPT", 5: "ENDARC"}


def _vint(buf: bytes, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def list_entries(path: str | Path) -> list[dict]:
    """列出压缩包内所有文件条目及原始数据区间（offset/size）。"""
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"不是 RAR5 文件: {path}")
    pos = 8
    entries: list[dict] = []
    while pos < len(data):
        hdr_start = pos
        pos += 4                              # 头部 CRC32
        hsize, pos = _vint(data, pos)
        hdr_body_start = pos
        htype, pos = _vint(data, pos)
        hflags, pos = _vint(data, pos)
        data_size = 0
        if hflags & 0x0001:                   # 有数据区
            data_size, pos = _vint(data, pos)
        if hflags & 0x0002:                   # 有 extra 区
            _extra_len, pos = _vint(data, pos)

        if htype == 2:                        # FILE 头
            dflags, pos = _vint(data, pos)
            unp_size, pos = _vint(data, pos)
            _attr, pos = _vint(data, pos)
            if dflags & 0x0002:               # mtime
                pos += 4
            crc_field = None
            if dflags & 0x0004:               # CRC32
                crc_field = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            comp_info, pos = _vint(data, pos)
            host_os, pos = _vint(data, pos)
            name_len, pos = _vint(data, pos)
            name = data[pos:pos + name_len].decode("utf-8", "replace")
            pos += name_len
            method = (comp_info >> 7) & 0x7
            solid = bool(comp_info & 0x40)
            dict_size = (1 << ((comp_info & 0x3F) + 17)) if (comp_info & 0x3F) else 0
            # 数据区紧跟头部
            data_off = hdr_body_start + hsize
            entries.append({
                "name": name, "type": htype, "unp_size": unp_size,
                "data_size": data_size, "data_off": data_off,
                "method": method, "solid": solid, "dict_size": dict_size,
                "crc32": crc_field, "host_os": host_os,
            })

        pos = hdr_body_start + hsize
        if htype == 5:
            break
    return entries


def extract_stored(path: str | Path, out_dir: str | Path) -> list[Path]:
    """解出未压缩(method=0)的条目。"""
    data = Path(path).read_bytes()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for e in list_entries(path):
        if e["method"] != 0:
            continue
        body = data[e["data_off"]: e["data_off"] + e["data_size"]]
        target = out_dir / Path(e["name"]).name
        target.write_bytes(body)
        written.append(target)
    return written


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=" * 60)
        print(p)
        for e in list_entries(p):
            print(f"  {e['name']:<40} unp={e['unp_size']:>10} data={e['data_size']:>10} "
                  f"method={e['method']} solid={e['solid']} dict={e['dict_size']}")
