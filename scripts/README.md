# 脚本索引

所有脚本均从项目根目录运行，用 `.venv/bin/python` 执行。

---

## data/ — 数据预处理

把原始数据转换成模型可用的样本或特征表。

| 脚本 | 作用 | 产出 |
|---|---|---|
| `prepare_data.py` | 轨迹 → 滑窗样本 | `samples.npz`, `samples_spec.json` |
| `build_sat_patches.py` | 卫星云图 → patch 缓存 + 标量表 | `sat_patches.npz`, `sat_scalars.parquet` |
| `build_environment.py` | 环境场入口（真实 ERA5 / 合成冒烟） | `env_*.parquet` |
| `era5_env.py` | 从 ERA5 分块提取 9 维环境特征（v1） | `env_era5.parquet` |
| `era5_env_v2.py` | 提取 v2/v3 环境特征（深层引导、多半径圈） | `env_era5_v2.parquet` |
| `era5_terrain.py` | 从 ERA5 静态场提取 7 维地形特征 | `env_terrain.parquet` |

```bash
# 轨迹样本
.venv/bin/python scripts/data/prepare_data.py

# 环境场（先检查数据完整性）
.venv/bin/python scripts/data/era5_env.py --check
.venv/bin/python scripts/data/era5_env.py

# 地形（需 ERA5 目录下有 geopotential_at_surface/0.0 与 land_sea_mask/0.0）
.venv/bin/python scripts/data/era5_terrain.py

# 卫星 patch
.venv/bin/python scripts/data/build_sat_patches.py
```

---

## path/ — 路径预测

| 脚本 | 作用 |
|---|---|
| `train.py` | 单模型训练 |
| `compare.py` | 四模型对比（持续法 / 线性 / GRU / Transformer） |
| `baseline_report.py` | 业务预报误差基线（21 个方法，2024-2025） |
| `ablate_environment.py` | 环境场消融（支持 `--folds K` 做 K 折） |
| `run_ablation_gpu.py` | 多 seed 并行执行器（比串行快 2-3 倍） |
| `stratified_ablation.py` | 分层消融（按海域/纬度拆分） |
| `latitude_adaptive.py` | 多种折划分的稳健性检验（推荐用于最终结论） |

**推荐的环境场评估流程**：

```bash
# 单次 K 折（快速看趋势）
.venv/bin/python scripts/path/ablate_environment.py \
    --env data/processed/env_era5.parquet --folds 5 --set device=cuda train.epochs=60

# 多种折划分（结论可靠，25 个 job）
.venv/bin/python scripts/path/latitude_adaptive.py \
    --env data/processed/env_era5.parquet --folds 5 \
    --fold-seeds 0 42 7 123 2024 --device cuda --epochs 60
```

**注意**：环境场消融必须用 **≥60 epoch**。用 30 epoch 会因训练不足得出"无效"的错误结论。

---

## intensity/ — 强度预测

| 脚本 | 作用 |
|---|---|
| `train_intensity.py` | 训练与 K 折。支持 `--warmcore`（暖心）、`--terrain`（地形）、`--fold-seeds`（多划分） |
| `rerun_ablations.py` | 三变体统一重跑（base / +暖心 / +地形），每个变体训练 4 个模型 |
| `compare_segments.py` | 按地形分段训练独立模型（已验证无效） |
| `compare_segment_approaches.py` | one-hot 段别 vs 残差修正（残差修正边际有效） |

```bash
# 标准训练 / K 折
.venv/bin/python scripts/intensity/train_intensity.py --cv 5 --set device=cuda

# 带特征变体
.venv/bin/python scripts/intensity/train_intensity.py --cv 5 --warmcore --fold-seeds 42 0 7
.venv/bin/python scripts/intensity/train_intensity.py --cv 5 --terrain data/processed/env_terrain.parquet

# 三变体对比（推荐，一次给出跨变体表格）
.venv/bin/python scripts/intensity/rerun_ablations.py --device cuda --folds 5 --fold-seeds 42 0 7
```

---

## ri/ — 快速增强事件

| 脚本 | 作用 |
|---|---|
| `train_ri.py` | RI 二分类，10 折交叉验证，支持阈值 10 / 15 m/s |

```bash
.venv/bin/python scripts/ri/train_ri.py --cv 10 --threshold 10
.venv/bin/python scripts/ri/train_ri.py --cv 10 --threshold 15
```

---

## evaluation/ — 分层评估

| 脚本 | 作用 |
|---|---|
| `stratified_report.py` | 按海域、强度分档统计误差（揭示子集差异） |

```bash
.venv/bin/python scripts/evaluation/stratified_report.py
```

实测各海域难度差异达 1.9 倍（南海 374 km vs 日本以南 701 km）。

---

## utils/ — 工具

| 脚本 | 作用 |
|---|---|
| `check_gpu.py` | GPU / CUDA 环境自检 |
| `separate_data.py` | 把 `data/` 移到外部盘并建软链接（`--target DIR --apply`） |

---

## 通用约定

- **路径解析**：所有脚本通过 `src/config.py` 的 `resolve_path()` 读取数据路径，
  自动支持 `TYPHOON_DATA_ROOT` 环境变量与 `data_root` 配置
- **设备选择**：`--set device=cuda|cpu|auto`
- **输出位置**：`outputs/<run_name>/`，含 `logs/`、`checkpoints/`、`*.json`
- **随机种子**：`--seeds 0 1 2` 或 `--fold-seeds 42 0 7`（后者用于 K 折）
