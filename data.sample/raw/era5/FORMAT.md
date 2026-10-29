# ARCO-ERA5 分块文件格式

## 文件命名

```
<time_idx>.<level_idx>.<lat_idx>.<lon_idx>   ← 多层变量（geopotential / u / v）
<time_idx>.<lat_idx>.<lon_idx>               ← 单层变量（sst / msl）
```

- `time_idx`：第几个 6 小时时次，起点 1959-01-01 00:00 UTC
- 其余索引：**恒为 0**（分块已含全部层与全球网格）

## 时间索引换算

```python
from datetime import datetime, timedelta

TIME_ORIGIN = datetime(1959, 1, 1)

def time_to_index(dt):
    return int((dt - TIME_ORIGIN).total_seconds() // (6 * 3600))

def index_to_time(idx):
    return TIME_ORIGIN + timedelta(hours=idx * 6)

# 示例
time_to_index(datetime(2021, 8, 1))   # -> 91432
index_to_time(91432)                  # -> 2021-08-01 00:00:00
```

## 数据布局

| 项 | 值 |
|---|---|
| 压缩算法 | blosc（lz4，shuffle） |
| 数据类型 | float32，小端 |
| 网格 | 全球 0.25°，纬度 721 行（90°N→90°S），经度 1440 列（0°E→360°E） |
| 多变量形状 | 解压后 `(13, 721, 1440)` |
| 单变量形状 | 解压后 `(721, 1440)` |

### 13 个气压层顺序

```
索引:  0    1    2    3    4    5    6    7    8    9   10   11    12
层:   50  100  150  200  250  300  400  500  600  700  850  925  1000  (hPa)
                              ▲         ▲         ▲
                           200(3)    500(7)    850(10)
```

### 经纬度 → 行列

```python
row = int(round((90.0 - lat) / 0.25))    # 第 0 行是 90°N
col = int(round((lon % 360.0) / 0.25))   # 第 0 列是 0°E
```

### 单位换算

| 变量 | 原始单位 | 换算 |
|---|---|---|
| geopotential | m²/s² | ÷ 9.80665 → 位势米 |
| u/v_component_of_wind | m/s | 无 |
| sea_surface_temperature | K | − 273.15 → °C |
| mean_sea_level_pressure | Pa | ÷ 100 → hPa |

## 读取示例

```python
import numpy as np
from numcodecs import Blosc

dec = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE, blocksize=0)

with open('geopotential/91432.0.0.0', 'rb') as f:
    data = dec.decode(f.read())

z = np.frombuffer(data, '<f4').reshape(13, 721, 1440)
h500 = z[7] / 9.80665          # 500hPa 位势高度（米）

lat, lon = 15.0, 130.0
row, col = int((90-lat)/0.25), int((lon%360)/0.25)
print(f"{lat}N {lon}E 的 500hPa 高度: {h500[row, col]:.0f} m")
```
