# 数据集说明

原始数据约 **374 GB**，不随仓库分发。本文说明目录要求、各文件格式来源，以及如何挂载。

---

## 一、挂载方式

数据根目录的解析优先级：

| 优先级 | 方式 | 适用场景 |
|---|---|---|
| 1 | 环境变量 `TYPHOON_DATA_ROOT` | 临时切换、多数据集共存 |
| 2 | 配置 `configs/default.yaml` 的 `data_root` | 固定路径 |
| 3 | `<项目>/data`（默认） | 软链接或实际目录 |

`data.raw_dir`、`data.processed_dir` 等都**相对数据根目录**解析。

### 方式一：软链接（命令完全不用改）

```bash
# 预览要做什么
.venv/bin/python scripts/utils/separate_data.py --target /mnt/datasets/typhoon

# 执行（移动数据 + 建软链接）
.venv/bin/python scripts/utils/separate_data.py --target /mnt/datasets/typhoon --apply
```

之后所有命令照常。跨文件系统移动可能失败，可改用 `--copy` 再手动删原目录。

### 方式二：环境变量

```bash
export TYPHOON_DATA_ROOT=/mnt/datasets/typhoon
.venv/bin/python scripts/path/train.py
```

适合多数据集共存：

```
/mnt/datasets/
├── typhoon/            主数据集（5 年台风季 ERA5）
├── typhoon_mini/       小样本，快速调试
└── typhoon_2022/       扩展年份
```

### 方式三：配置文件

```yaml
# configs/default.yaml
data_root: /mnt/datasets/typhoon
```

---

## 二、目录结构

```
<数据根目录>/
├── raw/
│   ├── era5/                   ARCO-ERA5 分块（约 374 GB）
│   │   ├── geopotential/       <idx>.0.0.0
│   │   ├── u_component_of_wind/
│   │   ├── v_component_of_wind/
│   │   ├── sea_surface_temperature/   <idx>.0.0
│   │   ├── mean_sea_level_pressure/
│   │   ├── geopotential_at_surface/  0.0（静态场，地形）
│   │   ├── land_sea_mask/       0.0（静态场）
│   │   └── level/               0
│   ├── bsts/                   卫星包清单 + 站点风观测
│   ├── cma_attachments/        CMA 最佳路径 + 业务预报误差表
│   ├── metadata/
│   └── tracks_WP_clean.parquet 清洗后轨迹
└── processed/                  由脚本生成
    ├── samples.npz
    ├── samples_spec.json
    ├── sat_scalars.parquet
    ├── sat_patches.npz
    ├── env_era5.parquet
    └── env_terrain.parquet
```

---

## 三、各数据说明

### tracks_WP_clean.parquet（必需）

**来源**：IBTrACS v4 西北太平洋清洗后。

| 列 | 类型 | 说明 |
|---|---|---|
| `sid` | str | 台风编号（如 `2021214N21110`） |
| `time` | datetime64 | UTC 观测时间 |
| `lat` / `lon` | float | 纬度 / 经度（0-360） |
| `wind` | float | 最大风速（knot） |
| `pres` | float | 中心气压（hPa） |

**规模**：1420 台风 / 72206 记录 / 约 776 KB。

### raw/era5/（路径实验必需）

**来源**：ARCO-ERA5 公开数据集（Google Cloud，免注册）。

**命名规则**（最容易出错的地方）：

| 变量类型 | 文件名 | 示例 |
|---|---|---|
| 多层（geopotential / u / v） | `<idx>.0.0.0` | `91432.0.0.0` |
| 单层（sst / msl） | `<idx>.0.0` | `91432.0.0` |
| 静态场（地形 / 海陆） | **`0.0`** | 全数据集仅一份 |

`<idx>` 是时间索引：`(目标时间 - 1959-01-01 00:00) 小时数 ÷ 6`。

**数据布局**：

| 项 | 值 |
|---|---|
| 压缩 | blosc（lz4 + shuffle），float32 小端 |
| 网格 | 全球 0.25°，721 × 1440（90°N → 90°S） |
| 多层形状 | 解压后 `(13, 721, 1440)` |
| 单层形状 | 解压后 `(721, 1440)` |

气压层顺序：

```
索引:  0    1    2    3    4    5    6    7    8    9   10   11    12
层:   50  100  150  200  250  300  400  500  600  700  850  925  1000  (hPa)
                            ▲         ▲         ▲
                         200(3)    500(7)    850(10)
```

行列映射与单位换算：

```python
row = int(round((90.0 - lat) / 0.25))    # 第 0 行是 90°N
col = int(round((lon % 360.0) / 0.25))   # 第 0 列是 0°E

geopotential      ÷ 9.80665 → 位势米
sst               − 273.15  → °C
mean_sea_level    ÷ 100     → hPa
```

**当前覆盖**：2017-2021 年台风季（6-11 月），共 3421 时次 / 17105 文件。

> 注意：数据集名 `1959-2022-6h` 有误导，**实际最后时次是 2021-12-31**。
> 需要 2022 年及以后的数据要用 `full_37-1h-0p25deg-chunk-1.zarr-v3`（1940-2026，37 气压层）。

**完整性检查**：

```bash
.venv/bin/python scripts/data/era5_env.py --check
```

会报告各变量时次数、时间范围、一致性，并抽样解压验证物理量合理性。

出现过的问题（供排查参考）：

| 现象 | 原因 |
|---|---|
| 文件大小正常但解压失败 | 下载中断导致内容损坏 |
| 不同变量时次数不一致 | 某变量的下载被中断 |
| u/v 风场数值达 27864 | 文件内容实为 geopotential（下载串包） |
| 文件名带 URL 前缀 | 部分下载工具会把路径拼进文件名 |

### raw/bsts/（强度实验必需）

| 文件 | 说明 |
|---|---|
| `zip_list.txt` | 卫星 NetCDF 压缩包路径，每行一个绝对路径 |
| `1949-2018_Wind.csv` 等 | 历史站点风速/阵风观测（可选） |

**来源**：风云卫星台风数据集（FYSATTY）。当前收集 2021-2024 年共 103 个压缩包。

### raw/cma_attachments/（业务基线）

| 文件 | 说明 |
|---|---|
| `CHSATdata.zip` | 卫星定位台风中心（2004-2013） |
| `2024_tc_tra_err.csv` / `2025_tc_tra_err.csv` | 路径预报误差（21 个方法） |
| `2024_tc_int_err.csv` / `2025_tc_int_err.csv` | 强度预报误差 |

用于 `scripts/path/baseline_report.py`。

### processed/（脚本生成，无需手动准备）

| 文件 | 生成命令 |
|---|---|
| `samples.npz` | `scripts/data/prepare_data.py` |
| `sat_scalars.parquet` | `scripts/data/build_sat_patches.py` |
| `sat_patches.npz` | 同上 |
| `env_era5.parquet` | `scripts/data/era5_env.py` |
| `env_terrain.parquet` | `scripts/data/era5_terrain.py` |

---

## 四、最小可运行配置

```bash
# 只需轨迹数据
data/raw/tracks_WP_clean.parquet

.venv/bin/python scripts/data/prepare_data.py
.venv/bin/python scripts/path/train.py --set device=cpu train.epochs=2
```

要跑环境场实验，额外需要 `raw/era5/`；
要跑强度实验，额外需要 `raw/bsts/zip_list.txt` 指向的卫星压缩包。

---

## 五、GitHub 上传注意事项

### 软链接的坑

如果 `data` 是软链接，`.gitignore` 里写 `data/` **不管用**——`data/` 只匹配目录，
软链接会被当成文件提交，内容只是路径字符串。别人克隆后得到一个指向不存在路径的断链。

```gitignore
# 必须写 "data"（不带斜杠），同时匹配目录和软链接
data
```

### 本仓库的 .gitignore

```gitignore
# 数据（软链接和目录都要忽略）
data

# 但保留数据模板
!data.sample/
!data.sample/**

outputs/
*.pt
*.ckpt
.venv/
__pycache__/
```

### 上传前检查

```bash
# 确认 data 被忽略
git check-ignore -v data

# 确认没有软链接被跟踪（应无输出）
git ls-files -s | awk '$1=="120000" {print "软链接:", $4}'

# 确认仓库体积（应远小于 50 MB）
git ls-files | xargs -r cat | wc -c
```

### 别人克隆后

仓库不含数据，需要自行准备后用上文三种方式之一挂载。
`data.sample/` 提供了目录骨架。
