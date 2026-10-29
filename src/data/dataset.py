"""PyTorch Dataset 与数据打包。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FeatureSpec, build_samples, fit_feature_scaler


class TrackWindowDataset(Dataset):
    """滑窗样本数据集。

    返回:
      x            (in_len, F)  已标准化特征
      y_delta      (out_len, 2) 目标增量
      last_abs     (2,)         最后观测点 [lat, lon]
      valid_mask   (in_len,)    bool, True 表示真实观测（非 padding）
      y_abs        (out_len, 2) 目标绝对坐标（仅评估用）
    """

    def __init__(self, bundle: dict, spec: FeatureSpec, indices: np.ndarray | None = None):
        self.bundle = bundle
        self.spec = spec
        self.indices = np.arange(len(bundle["x"])) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        x = self.bundle["x"][j]
        x = (x - self.spec.mean) / self.spec.std
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y_delta": torch.from_numpy(self.bundle["y"][j].astype(np.float32)),
            "last_abs": torch.from_numpy(self.bundle["last_abs"][j].astype(np.float32)),
            "valid_mask": torch.from_numpy(self.bundle["valid_mask"][j].astype(bool)),
            "y_abs": torch.from_numpy(self.bundle["y_abs"][j].astype(np.float32)),
            "year": int(self.bundle["year"][j]),
        }


def collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    out = {}
    for key in keys:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = torch.tensor(values)
    return out


def split_by_year(bundle: dict, val_years: list[int], test_years: list[int]) -> dict[str, np.ndarray]:
    """按年份划分训练/验证/测试，避免同一场台风泄漏到不同集合。

    注意这是样本级切分。同一条长轨迹的相邻窗口可能跨越年份边界，
    实际业务评估建议改成"风暴级切分"（见 split_by_storm）。
    """
    years = bundle["year"]
    val_lo, val_hi = val_years
    test_lo, test_hi = test_years
    is_test = (years >= test_lo) & (years <= test_hi)
    is_val = (years >= val_lo) & (years <= val_hi)
    is_train = ~is_test & ~is_val
    return {
        "train": np.where(is_train)[0],
        "val": np.where(is_val)[0],
        "test": np.where(is_test)[0],
    }


def split_by_storm(bundle: dict, val_years: list[int], test_years: list[int],
                   seed: int = 42) -> dict[str, np.ndarray]:
    """严格按风暴划分：验证/测试年份的风暴整条进对应集合，训练集里再留 10% 风暴做验证。

    这比按年份切样本更严格，因为同一条轨迹的滑动窗口之间高度相关。
    """
    years = bundle["year"]
    sids = bundle["sid"]
    val_lo, val_hi = val_years
    test_lo, test_hi = test_years

    unique_sids = np.unique(sids)
    storm_year = {}
    for sid in unique_sids:
        storm_year[sid] = int(years[sids == sid].min())

    val_storms = [s for s, y in storm_year.items() if val_lo <= y <= val_hi]
    test_storms = [s for s, y in storm_year.items() if test_lo <= y <= test_hi]
    train_pool = [s for s, y in storm_year.items() if not (val_lo <= y <= val_hi) and not (test_lo <= y <= test_hi)]

    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool)
    n_holdout = max(1, int(len(train_pool) * 0.1)) if train_pool else 0
    holdout = train_pool[:n_holdout]
    train_storms = train_pool[n_holdout:]

    def mask_for(storm_list: list[str]) -> np.ndarray:
        if not storm_list:
            return np.array([], dtype=int)
        return np.where(np.isin(sids, storm_list))[0]

    return {
        "train": mask_for(train_storms),
        "val": np.concatenate([mask_for(val_storms), mask_for(holdout)]).astype(int),
        "test": mask_for(test_storms),
    }


# ----------------------------------------------------------------------------
# 预处理流水线：原始长表 -> 打包的 npz
# ----------------------------------------------------------------------------
def preprocess(cfg, df) -> tuple[dict, FeatureSpec]:
    """执行预处理并返回 (bundle, spec)。spec 只由训练集拟合。"""
    names = list(cfg.features.list)
    bundle = build_samples(
        df,
        in_len=int(cfg.data.in_len),
        out_len=int(cfg.data.out_len),
        interval_hours=float(cfg.data.interval_hours),
        feature_names=names,
    )
    splits = split_by_year(bundle, list(cfg.data.val_years), list(cfg.data.test_years))
    train_idx = splits["train"]
    if len(train_idx) == 0:
        raise ValueError("训练集为空，请检查 val_years/test_years 是否覆盖了全部年份")

    # 训练集里出现的缺失/极端值必须在拟合标准化参数前清理，否则 mean/std 会变成 NaN
    train_x = np.nan_to_num(bundle["x"][train_idx], nan=0.0, posinf=0.0, neginf=0.0)
    spec = fit_feature_scaler(train_x)
    spec.names = names
    bundle["x"] = np.nan_to_num(bundle["x"], nan=0.0, posinf=0.0, neginf=0.0)

    # 尽早暴露"划分后某个集合为空"的问题，否则训练时会静默地得到 nan 指标
    empty = [k for k in ("train", "val", "test") if len(splits[k]) == 0]
    if empty:
        years = bundle["year"]
        raise ValueError(
            f"以下划分集合为空: {empty}。数据实际年份范围 {int(years.min())}-{int(years.max())}，"
            f"请检查 data.val_years={list(cfg.data.val_years)} / data.test_years={list(cfg.data.test_years)}"
        )
    bundle["val_years"] = np.array(list(cfg.data.val_years), dtype=np.int32)
    bundle["test_years"] = np.array(list(cfg.data.test_years), dtype=np.int32)
    bundle["interval_hours"] = np.array(float(cfg.data.interval_hours))
    return bundle, spec


def save_bundle(bundle: dict, spec: FeatureSpec, out_dir: str | Path, prefix: str = "samples") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in bundle.items() if isinstance(v, np.ndarray)}
    npz_path = out_dir / f"{prefix}.npz"
    np.savez_compressed(npz_path, **arrays)
    import json
    (out_dir / f"{prefix}_spec.json").write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[preprocess] 已保存 {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    return npz_path


def load_bundle(out_dir: str | Path, prefix: str = "samples") -> tuple[dict, FeatureSpec]:
    out_dir = Path(out_dir)
    import json
    data = np.load(out_dir / f"{prefix}.npz", allow_pickle=False)
    bundle = {k: data[k] for k in data.files}
    spec = FeatureSpec.from_dict(json.loads((out_dir / f"{prefix}_spec.json").read_text(encoding="utf-8")))
    return bundle, spec
