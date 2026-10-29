"""合成台风轨迹生成器。

用途:
1. 无需联网的冒烟测试（验证代码路径、调通训练）；
2. 单元测试与 CI。

注意: 合成数据的统计特性与真实台风不同，模型在合成数据上得到的误差
数字不能当作科学结论，只能证明流水线可运行。

生成模型是一个简化版的路径-强度随机过程:
- 台风沿西北方向移动，接近高纬逐渐转向东北（影响路径的常见形态）
- 强度随海温/时间先增强后衰减
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import dest_point, haversine_km


def generate_synthetic_storms(
    n_storms: int = 260,
    n_years: int = 14,
    points_per_storm: int = 48,
    interval_hours: float = 6.0,
    start_year: int = 1995,
    seed: int = 0,
) -> pd.DataFrame:
    """返回与 IBTrACS 读取结果列名一致的长表。"""
    rng = np.random.default_rng(seed)
    records = []
    year_list = rng.integers(start_year, start_year + n_years, size=n_storms)

    for i in range(n_storms):
        sid = f"SYN{i:05d}"
        year = int(year_list[i])
        # 起点：西北太平洋低纬海域
        lat = rng.uniform(8.0, 20.0)
        lon = rng.uniform(125.0, 150.0)

        # 初始运动：西北向，带随机扰动
        bearing = rng.normal(305.0, 25.0) % 360.0
        speed_kmh = rng.uniform(12.0, 28.0)
        # 转向纬度：接近它以后逐渐转为东北向
        recurve_lat = rng.uniform(24.0, 34.0)

        # 强度演化：先增强后减弱
        peak_at = rng.integers(8, max(9, points_per_storm - 12))
        wind_peak = rng.uniform(35.0, 75.0)     # knots
        t0 = pd.Timestamp(f"{year}-{rng.integers(6, 11):02d}-{rng.integers(1, 28):02d}",
                          tz="UTC")
        t0 = t0 + pd.Timedelta(hours=int(rng.integers(0, 3)) * 6)

        for t in range(points_per_storm):
            if lat > recurve_lat:
                # 转向东北
                delta_b = ((60.0 - bearing + 180) % 360) - 180
                bearing = (bearing + 0.15 * delta_b + rng.normal(0, 3)) % 360.0
            else:
                bearing = (bearing + rng.normal(0, 6)) % 360.0

            move_km = speed_kmh * interval_hours
            new_lat, new_lon = dest_point(lat, lon, bearing, move_km)
            if haversine_km(lat, lon, new_lat, new_lon) > 700:   # 异常跳变兜底
                break
            lat, lon = float(new_lat), float(new_lon % 360.0)
            if lat > 50 or lat < -2 or not (100 <= lon <= 200):
                break

            # 强度曲线：抛物线 + 小型周期扰动
            x = t / max(1, points_per_storm - 1)
            shape = 1.0 - ((x - peak_at / points_per_storm) / 0.45) ** 2
            wind = max(10.0, wind_peak * max(0.25, shape) + rng.normal(0, 4))
            pres = 1012.0 - 1.35 * (wind - 20.0) + rng.normal(0, 2.5)

            records.append({
                "sid": sid,
                "time": t0 + pd.Timedelta(hours=interval_hours * t),
                "lat": lat,
                "lon": lon,
                "wind": wind,
                "pres": max(900.0, pres),
                "basin": "WP",
                "season": year,
            })

    df = pd.DataFrame(records).sort_values(["sid", "time"]).reset_index(drop=True)
    print(f"[synthetic] 生成 {df['sid'].nunique()} 个风暴, {len(df)} 条记录")
    return df
