# 台风路径与强度预测：Transformer 基线工程

从零搭建的台风预测系统，覆盖**路径回归、强度回归、快速增强（RI）事件分类**三个任务，
并在 IBTrACS 轨迹、ARCO-ERA5 环境场、风云卫星云图上做了系统性消融实验。

> 完整实验记录与方法论分析见 [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

---

## 核心结果

| 任务 | 评估范围 | 最佳模型 | 指标 | vs 持续法 |
|---|---|---|---|---|
| 路径（24h） | 全量 1420 台风 | Transformer | **208 km** | 567 km（好 63%） |
| 路径（24h） | 环境场子集 128 台风 | Transformer + 环境场 | **320 km** | 330 km（好 3.1%） |
| 强度（12h） | 224 台风，15 job | 纯时序 GRU | **1.53 m/s** | 2.17 m/s（好 29%） |
| RI（AUC） | 224 台风，10 折 | GRU | **0.879 / 0.904**（阈值 10/15） | 0.808 / 0.846 |

两行"路径"不可直接比较：全量行训练数据多 15 倍，误差自然更低。

### 消融结论

| 尝试 | 结果 | 显著性 |
|---|---|---|
| **环境场 → 路径** | **−10.1 km（−3.1%）** | **p=0.0005，25 job** |
| 卫星云图 → 强度 | +0.11 m/s | 15 job **0/15 胜**，p=3.9e-08，**显著更差** |
| 暖心结构 → 强度 | +0.01 m/s | p=0.42，无变化 |
| 地形特征 → 强度 | +0.01 m/s | p=0.55，无变化 |
| 纬度自适应分区 | +0.8 km vs 全程环境场 | p=0.53，无差异 |

**三点重要提醒**：

1. 环境场的收益**依赖充分训练**（≥60 epoch）。用 30 epoch 时效应消失甚至反转为
   +2.6 km —— 这是本文早期错误结论的来源。
2. 所有特征消融都用 **多种折划分**（5 折 × 3~5 种划分）验证。单一划分会给出
   不可靠结论（同一配置在不同划分下可从 −10 km 变到 +3 km）。
3. **卫星云图这条路已经走完，不要再试**。四个独立方向都无效：数据扩容
   （90→224 台风）、加大 dropout、把视野从 ±64 km 扩到 ±128 km、把亮度归一化
   改成 z-score 并新增 8 维圈层特征——**每次都是 0/15 全负，ΔMAE 稳定在
   +0.11 m/s**。详见技术报告 §4.1.2。云图特征本身携带信号（单变量相关达
   −0.55），但信息已被 `cent_prs` 与强度历史覆盖，属于冗余。

---

## 数据

原始数据（约 374 GB ERA5 + 轨迹 + 卫星）**不在本仓库**，通过挂载使用。

数据根目录解析优先级：

| 优先级 | 方式 | 说明 |
|---|---|---|
| 1 | 环境变量 `TYPHOON_DATA_ROOT` | 临时切换、多数据集共存 |
| 2 | `configs/default.yaml` 的 `data_root` | 固定路径 |
| 3 | `<项目>/data`（默认） | 软链接或实际目录 |

```bash
# 挂载到外部目录后，三种用法任选
export TYPHOON_DATA_ROOT=/mnt/datasets/typhoon

# 或建软链接（之后所有命令无需改动）
.venv/bin/python scripts/utils/separate_data.py --target /mnt/datasets/typhoon --apply
```

所需目录结构、各文件格式与来源说明见 [docs/DATASETS.md](docs/DATASETS.md)，
仓库内的 [data.sample/](data.sample/) 提供目录骨架。

### 数据构成

| 数据 | 来源 | 规模 |
|---|---|---|
| 轨迹 | IBTrACS v4 西北太平洋 | 1420 台风 / 72206 记录 |
| 环境场 | ARCO-ERA5（Google 公开，免注册） | 3421 时次 / 17105 文件 |
| 卫星云图 | 风云 FY-2C/E/F/G/H、FY-4A/B | **224 台风 / 17193 帧**（2005-2024） |

> 卫星数据中 203 帧（1.17%）为源文件 HDF5 元数据损坏，无法恢复；已自动跳过，
> 87% 的台风数据完整。

---

## 快速开始

```bash
# 环境（复用系统科学栈，避免重复安装 numpy/torch 等）
python -m venv .venv --system-site-packages
.venv/bin/pip install -i https://mirrors.cernet.edu.cn/pypi/web/simple -r requirements.txt

# 自检（14 项）
.venv/bin/python tests/test_core.py
```

### 路径预测

```bash
# 1) 轨迹 -> 滑窗样本
.venv/bin/python scripts/data/prepare_data.py

# 2) 训练
.venv/bin/python scripts/path/train.py --set device=cuda

# 3) 四模型对比（持续法 / 线性 / GRU / Transformer）
.venv/bin/python scripts/path/compare.py

# 4) 业务预报基线（EC / EC AIFS / 中央气象台等 21 个方法）
.venv/bin/python scripts/path/baseline_report.py
```

### 环境场实验

```bash
# 1) 从本地 ERA5 分块提取环境特征
.venv/bin/python scripts/data/era5_env.py --check    # 先验证数据完整性
.venv/bin/python scripts/data/era5_env.py            # 提取 -> env_era5.parquet

# 2) 消融：有/无环境场（K 折 × 多划分，最严谨）
.venv/bin/python scripts/path/ablate_environment.py \
    --env data/processed/env_era5.parquet --folds 5 --set device=cuda

# 3) 多种折划分的稳健性检验（推荐，25 个 job）
.venv/bin/python scripts/path/latitude_adaptive.py \
    --env data/processed/env_era5.parquet --folds 5 \
    --fold-seeds 0 42 7 123 2024 --device cuda --epochs 60

# 4) 并行跑多个 seed（比串行快 2-3 倍）
.venv/bin/python scripts/path/run_ablation_gpu.py \
    --env data/processed/env_era5.parquet --seeds 0 1 2 3 4 --workers 3 --device cuda
```

### 强度与 RI

```bash
# 1) 卫星云图 -> patch 缓存（边长由 satellite.patch_size 决定，默认 64 即 ±128km）
#    缓存文件名自动带尺寸：sat_patches64.npz，换尺寸不会覆盖旧缓存
#    耗时约 17 分钟，产出约 264 MB
.venv/bin/python scripts/data/build_sat_patches.py

# 2) 强度预测（5 折 × 多划分，可加 --warmcore / --terrain）
.venv/bin/python scripts/intensity/train_intensity.py --cv 5 --fold-seeds 42 0 7 --set device=cuda

# 3) 三变体统一重跑（base / +暖心 / +地形，每个变体训练 4 个模型）
.venv/bin/python scripts/intensity/rerun_ablations.py --device cuda --folds 5 --fold-seeds 42 0 7

# 4) RI 事件（10 折，支持阈值 10 / 15 m/s）
.venv/bin/python scripts/ri/train_ri.py --cv 10 --threshold 10
```

**内存提示**：64px patch 下每个样本的云图占 375 KB，显存/内存随 batch 线性增长
（实测 batch=16/32/64/128 对应整进程峰值 0.78/0.97/1.33/2.01 GB）。
默认 `batch_size: 32`，16 GB 内存机器若仍 OOM 可降到 16。

### 两种 seed 不要混用

| 参数 | 控制什么 | 什么时候用 |
|---|---|---|
| `--fold-seeds 42 0 7` | **数据切分**：哪些台风进测试集 | 结论性实验**必须**用多种划分 |
| `--set seed=0 1 2` | **模型初始化 + 批次打乱顺序** | 要衡量训练随机性时用，3~5 个足够 |

`seed` 同时也是 `--fold-seeds` 的默认值，所以单独改 `seed` 会连切分一起改，
**无法单独衡量训练随机性**。正确做法：用 `--fold-seeds` 固定切分，
再用 `--set seed=` 单独扫训练随机性。不同 `--set` 的结果会自动写到不同文件名。

### 地形特征

```bash
.venv/bin/python scripts/data/era5_terrain.py
# -> env_terrain.parquet（7 维：山脉阻挡、陆地占比、地形梯度等）

# 强度 + 地形
.venv/bin/python scripts/intensity/train_intensity.py --terrain data/processed/env_terrain.parquet --cv 5
```

### 分层评估

```bash
# 按海域 / 强度分档看误差（持续法基线难度）
.venv/bin/python scripts/evaluation/stratified_report.py
```

---

## 目录结构

```
typhoon-transformer/
├── configs/default.yaml          全部超参数
├── data -> /mnt/datasets/typhoon 数据（软链接，不入库）
├── data.sample/                  数据结构模板（入库）
├── docs/
│   ├── TECHNICAL_REPORT.md       完整技术报告（实验记录 + 方法论）
│   └── DATASETS.md               数据结构、格式、挂载指南
├── scripts/
│   ├── data/                     预处理：样本、环境场、地形、卫星 patch
│   ├── path/                     路径：训练、对比、环境场消融、基线
│   ├── intensity/                强度：训练、分段对比、三变体重跑
│   ├── ri/                       RI 事件分类
│   ├── evaluation/               分层评估
│   └── utils/                    GPU 自检、数据分离
├── src/
│   ├── config.py                 配置加载 + 路径解析
│   ├── data/                     数据加载、特征、数据集
│   ├── models/                   Transformer / GRU / CNN 融合 / RI
│   ├── training/                 训练循环、球面距离损失
│   └── evaluation/               指标、绘图、分层统计
├── tests/test_core.py            14 项自检
└── outputs/                      实验结果（见 outputs/INDEX.md，权重不入库）
```

---

## 关键设计决策

| 决策 | 理由 |
|---|---|
| **增量预测**而非绝对坐标 | 避免数值范围问题，符合"位移"的物理直觉 |
| **Haversine 球面距离损失**（伪 Huber） | 经纬度网格在高纬度被压缩，欧氏距离会失真 |
| **按台风切分**，而非按年份 | 同一台风的相邻窗口必须在同一集合，否则信息泄漏 |
| **K 折 + 多种折划分** | 单一划分的结论不可靠（实测可从 −10 km 变到 +3 km） |
| **报告 24h 与 4 时次平均两个口径** | 两者灵敏度不同，混用会造成误判 |
| **消融时每组配置独立早停** | 新增特征需要更多轮次，固定 epoch 会让特征组吃亏 |

---

## 配置

```bash
# 改超参不改文件
.venv/bin/python scripts/path/train.py --set model.d_model=192 train.epochs=100

# 常用覆盖
--set device=cpu                     # 强制 CPU
--set data.in_len=20 data.out_len=8  # 120h 输入 -> 48h 输出
--set model.name=gru                 # 换模型
```

主要配置项见 [configs/default.yaml](configs/default.yaml)。

---

## 已知限制

| 限制 | 说明 |
|---|---|
| **与业务差距 3.9 倍** | 24h 路径 208 km vs ECMWF AIFS 54 km。差距来自数据规模、分辨率与端到端训练，非本工程特征能弥补 |
| 环境场仅覆盖 5 年台风季 | 2017-2021，每年 6-11 月（冬季无数据），共 128 台风 |
| ERA5 分辨率 0.25° | 约 28 km，台风内核结构被严重平滑 |
| 卫星云图已判定无效 | 224 台风 / 17193 帧已全部用上，四种改进方向均无增益 |
| 未做端到端训练 | 环境场是预先提取的 9 维特征，存在归纳偏置 |
| 硬件 | RTX 3050 Ti (4 GB)，模型规模与 batch 受限 |

---

## 下一步方向

按投入产出比排序：

1. **端到端训练** —— 直接从 ERA5 网格学习（而非 9 维手工特征），保留空间结构
2. **更高分辨率环境场** —— 0.25° → 0.1°，可能带来实质改善（数据量约 6 倍）
3. **延长输入窗口** —— 72h → 120h（转向信号往往更早出现）
4. **多任务联合训练** —— 路径与强度共享底层表征
5. **模型集成** —— 已有 GRU / Transformer / 线性，可做加权或 stacking

**已排除的方向**（不必再投入）：

| 方向 | 排除依据 |
|---|---|
| 卫星云图（红外） | 4 种配置 × 15 job，全部 0/15 全负 |
| 暖心结构特征反演 | 15 job，p=0.42，无变化 |
| 地形特征 | 15 job，p=0.55，无变化 |
| 纬度自适应分区 | p=0.53，无差异 |

若仍要在云图上尝试，**唯一方向性不同的思路是改变任务表达而非模型**：
不预测绝对强度，改为预测相对持续法的增量并配合增强/减弱分类辅助损失。
但基于四轮零效果，预期收益有限。

---

## 环境要求

Python 3.10+、PyTorch、numpy、pandas、pyarrow、PyYAML、matplotlib、netCDF4、zarr、numcodecs。

GPU 非必需（CPU 可跑，速度约慢 20 倍）。
