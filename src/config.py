"""简化的配置系统：YAML 加载 + 点号访问 + 命令行覆盖。

不依赖 omegaconf，只用 PyYAML（conda base 通常自带；没有则 pip install pyyaml）。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


class Config(dict):
    """支持 cfg.train.epochs 这种点号访问的嵌套字典。"""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - 错误路径
            raise AttributeError(f"配置项不存在: {name}") from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        self.pop(name, None)

    def to_dict(self) -> dict:
        def _unwrap(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: _unwrap(v) for k, v in node.items()}
            if isinstance(node, (list, tuple)):
                return [_unwrap(v) for v in node]
            return node

        return _unwrap(self)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_value(text: str) -> Any:
    """把命令行字符串解析成 bool/int/float/list/str。"""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(part) for part in inner.split(",") if part.strip()]
    return text


def _set_dotted(tree: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"无法在非字典节点上设置: {dotted}")
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """读取 YAML 配置，并应用形如 key.sub=value 的覆盖项。"""
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if overrides:
        patch: dict = {}
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"覆盖项格式应为 key=value，收到: {item}")
            key, _, value = item.partition("=")
            _set_dotted(patch, key.strip(), _parse_value(value))
        raw = _deep_merge(raw, patch)

    return Config(raw)


def dump_config(cfg: Config, path: str | Path) -> None:
    """把实际生效的配置写盘，方便复现实验。"""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# 数据根目录解析：支持把数据与代码分离（挂载外部数据集目录）
# ---------------------------------------------------------------------------
DATA_ROOT_ENV = "TYPHOON_DATA_ROOT"


def resolve_data_root(cfg: "Config", project_root: "Path | None" = None) -> Path:
    """解析数据根目录，优先级：

      1. 环境变量 TYPHOON_DATA_ROOT（最灵活，便于临时挂载不同数据集）
      2. 配置文件里的 data_root 字段
      3. 默认 <项目根目录>/data

    这样训练时既能用项目自带数据，也能指向外部挂载盘。
    """
    import os
    from pathlib import Path as _Path

    env = os.environ.get(DATA_ROOT_ENV)
    if env:
        return _Path(env).expanduser().resolve()
    configured = cfg.to_dict().get("data_root") if isinstance(cfg, dict) else None
    if configured:
        p = _Path(str(configured)).expanduser()
        if not p.is_absolute() and project_root is not None:
            p = _Path(project_root) / p
        return p.resolve()
    base = _Path(project_root) if project_root is not None else _Path.cwd()
    return (base / "data").resolve()


def resolve_path(cfg: "Config", value: str | Path, project_root: "Path | None" = None) -> Path:
    """把配置里的路径解析成绝对路径。

    - 绝对路径：原样返回
    - 以 "data/" 开头：相对于 resolve_data_root() 的上级解析
      （例如 "data/raw" -> <data_root>/raw）
    - 其它相对路径：相对于项目根目录
    """
    from pathlib import Path as _Path

    p = _Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    root = resolve_data_root(cfg, project_root)
    text = str(p).replace("\\", "/")
    if text.startswith("data/"):
        return (root / text[len("data/"):]).resolve()
    if text == "data":
        return root
    base = _Path(project_root) if project_root is not None else _Path.cwd()
    return (base / p).resolve()
