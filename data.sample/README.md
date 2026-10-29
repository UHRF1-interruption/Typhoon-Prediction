# 数据结构模板（data.sample）

仓库**不含实际数据**（约 14 GB）。本目录只放**目录骨架 + 说明**，
帮助你了解需要准备哪些文件，以及每个文件的格式。

## 如何还原成真正的 data/

```bash
# 方式一：复制模板，再按需放入数据
cp -r data.sample data
# 然后按本文件说明填充 raw/ 下的内容

# 方式二：数据放在外部盘，用软链接挂载（推荐）
mkdir -p /mnt/datasets/typhoon
cp -r data.sample/* /mnt/datasets/typhoon/
rm -rf data
ln -s /mnt/datasets/typhoon data

# 方式三：完全不建 data/，用环境变量
export TYPHOON_DATA_ROOT=/mnt/datasets/typhoon
```

配置优先级：`TYPHOON_DATA_ROOT` > `config.yaml` 的 `data_root` > `<项目>/data`。
详见 [docs/DATASETS.md](../docs/DATASETS.md)。

---

## 目录总览

```
data/
├── raw/                        原始数据（需要你提供）
│   ├── era5/                   ★ ARCO-ERA5 分块（14 GB，最大的一部分）
│   ├── bsts/                   风云卫星数据包清单 + 站点风观测
│   ├── cma_attachments/        CMA 最佳路径 + 业务预报误差表
│   ├── metadata/               辅助说明文件
│   └── tracks_WP_clean.parquet ★ 清洗后的轨迹（或由 IBTrACS 生成）
└── processed/                  处理后数据（可由脚本重新生成）
```

---

## raw/era5/ —— 环境场（可选，用于路径预测实验）

**来源**：Google Cloud 公开数据集 ARCO-ERA5（免注册）。
格式说明见 [FORMAT.md](raw/era5/FORMAT.md)。

**目录要求**：每个变量一个子目录，所有时次**平铺**在同一层（不要按月分子目录）。

| 子目录 | 变量 | 文件名格式 | 单文件大小 | 说明 |
|---|---|---|---|---|
| `geopotential/` | 位势高度 | `<idx>.0.0.0` | ~29 MB | 13 个气压层 |
| `u_component_of_wind/` | 纬向风 | `<idx>.0.0.0` | ~42 MB | 13 个气压层 |
| `v_component_of_wind/` | 经向风 | `<idx>.0.0.0` | ~43 MB | 13 个气压层 |
| `sea_surface_temperature/` | 海温 | `<idx>.0.0` | ~1.4 MB | 单层 |
| `mean_sea_level_pressure/` | 海平面气压 | `<idx>.0.0` | ~2.2 MB | 单层 |
| `level/` | 层坐标 | `0` | 4 KB | 只需一个 |
| `geopotential_at_surface/` | 地形高度 | `0.0` | 2.8 MB | **静态场**，全数据集一份 |
| `land_sea_mask/` | 海陆掩膜 | `0.0` | 0.8 MB | **静态场**，全数据集一份 |

后两个是静态场（无时间维），用于地形特征提取
（`scripts/data/era5_terrain.py`）。每个只需下载一个文件，与请求多少时次无关。

**文件名格式容易错**：

- 多层变量是 **4 段**：`91432.0.0.0`
- 单层变量是 **3 段**：`91432.0.0`
- `<idx>` 是时间索引：`(目标时间 - 1959-01-01 00:00) 小时数 ÷ 6`
  例如 `91432` = 2021-08-01 00:00

例：`raw/era5/geopotential/91432.0.0.0`

---

## raw/tracks_WP_clean.parquet —— 轨迹数据（必需）

**来源**：IBTrACS（NOAA 国际最佳路径）清洗后，或由 `scripts/data/prepare_data.py` 生成。

**格式**：pandas parquet，最少需要这些列：

| 列名 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `sid` | str | 台风编号 | `2021214N21110` |
| `time` | datetime64 | 观测时间（UTC） | `2021-08-15 06:00:00+00:00` |
| `lat` | float | 纬度（-90~90） | `21.3` |
| `lon` | float | 经度（0~360） | `128.7` |
| `wind` | float | 最大风速（knot/ft 混合，见下） | `75.0` |
| `pres` | float | 中心气压（hPa） | `965.0` |

**规模参考**：1420 个台风、72206 条记录、约 776 KB。

**生成方式**：

```bash
# 从 IBTrACS CSV 生成（会下载约 115 MB）
.venv/bin/python scripts/data/prepare_data.py --set data.source=ibtracs
```

---

## raw/bsts/ —— 卫星与站点数据

| 文件 | 必需 | 说明 |
|---|---|---|
| `zip_list.txt` | 强度模型必需 | 卫星 NetCDF 压缩包路径，每行一个 zip 的绝对路径 |
| `1949-2018_Wind.csv` | 可选 | 历史站点风速观测 |
| `1949-2018_Gust.csv` | 可选 | 历史站点阵风观测 |
| `1949-2018_W8Date.csv` | 可选 | 站点台风影响起止日期 |

`zip_list.txt` 样例（3 行）：

```
/home/haru/.dsh/attachments/v1/files/0e/0e6e.../202428.2426.zip
/home/haru/.dsh/attachments/v1/files/18/1858.../202417.2400.zip
```

**卫星数据来源**：风云卫星台风数据集（FYSATTY，上海台风所）。
本项目收集了 2021–2024 年共 103 个压缩包（43+ 台风）。

---

## raw/cma_attachments/ —— CMA 业务数据（可选）

| 文件 | 说明 |
|---|---|
| `CHSATdata.zip` | 卫星定位的台风中心（2004–2013） |
| `2024_tc_tra_err.csv` | 2024 年路径预报误差（21 个方法） |
| `2024_tc_int_err.csv` | 2024 年强度预报误差 |
| `2025_tc_tra_err.csv` / `2025_tc_int_err.csv` | 2025 年同样内容 |

用于 `scripts/path/baseline_report.py` 生成业务基线对比表。

---

## processed/ —— 处理后数据（自动生成）

**这些文件不需要手动提供**，运行对应脚本即可生成：

| 文件 | 生成命令 | 大小 | 说明 |
|---|---|---|---|
| `samples.npz` | `scripts/data/prepare_data.py` | 1.1 MB | 路径预测滑窗样本（20094 个） |
| `samples_spec.json` | 同上 | 0.5 KB | 标准化参数（推理必须复用） |
| `sat_scalars.parquet` | `scripts/data/build_sat_patches.py` | 352 KB | 卫星强度标量表 |
| `sat_patches.npz` | 同上 | 26 MB | 云图 patch 缓存（5925 帧） |
| `env_era5.parquet` | `scripts/data/era5_env.py` | 24 KB | 环境场特征表 |
| `used_config.yaml` | 预处理时自动保存 | 1 KB | 当时的配置快照 |

---

## 最小可运行配置

只想快速验证代码能跑通？只需要两样东西：

```bash
# 1. 轨迹数据（必需）
data/raw/tracks_WP_clean.parquet

# 2. 生成样本
.venv/bin/python scripts/data/prepare_data.py --set data.source=ibtracs

# 3. 训练
.venv/bin/python scripts/path/train.py --set device=cpu train.epochs=2
```

想跑**路径 + 环境场**实验，额外需要 `raw/era5/`。

想跑**强度预测**，额外需要 `raw/bsts/zip_list.txt` 指向的卫星压缩包。

---

## 检查是否就绪

```bash
# 检查 ERA5 数据
.venv/bin/python scripts/data/era5_env.py --check

# 检查轨迹数据
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/raw/tracks_WP_clean.parquet')
print('台风数:', df.sid.nunique(), '| 记录数:', len(df))
print('时间范围:', df.time.min(), '~', df.time.max())
"
```
