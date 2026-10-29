# outputs 目录索引

> 本目录存放**实验结果**，体积大的模型权重已全部清除（详见"关于权重文件"）。
> 当前总量约 3 MB / 254 个文件。
> 每一条结论在技术报告中的位置见"对应报告"列。

---

## 一、路径预测

| 目录 | 内容 | 24h MAE | 4 时效平均 | 对应报告 | 生成命令 |
|---|---|---|---|---|---|
| `compare_transformer/` | Transformer | **208.4 km** | 119.0 km | §3 路径基线 | `scripts/path/train.py` |
| `compare_gru/` | GRU 对照 | 206.5 km | 118.4 km | §3 | `scripts/path/compare.py` |
| `compare_linear/` | 线性对照 | 259.6 km | 143.1 km | §3 | 同上 |
| `compare_persistence/` | 持续法对照 | 566.6 km | 353.0 km | §3 | 同上 |
| `comparison/` | **四模型汇总**（`metrics.md` + 图） | — | — | §3.2 | `scripts/path/compare.py --compare` |
| `baselines/` | **基线表 + 3 张图**（入库） | — | — | §3.2 | `scripts/path/baseline_report.py` |

> **口径提醒**：24h 单时效与 4 时效平均差 90 km 左右，引用时必须声明是哪个
> （报告 §3 用 24h＝208.4 km）。此表中 GRU(206.5) 与 Transformer(208.4) 几乎持平，
> 该次运行 GRU 略优——与报告中"Transformer 208 km 为最佳"的表述基于不同划分，
> 以 `baselines/baseline_table.md` 的多折结论为准。

### 环境场消融（核心正面结论）

三套环境场特征版本，各 10 个 seed、30 epoch。目录名对应特征版本而非 run 序号：

| 目录 | 特征文件 | baseline(仅轨迹) | +环境场 | 仅环境场 | 结论 |
|---|---|---|---|---|---|
| `parallel/` | `env_era5.parquet`（v1，9 维） | — | — | — | v1 原始消融 |
| `v1_2020_ablation/` | `env_era5.parquet`（v1） | 206.5 km | 206.6 km | 226.8 km | 30 epoch 下**无效** |
| `v2_ablation/` | `env_era5_v2.parquet` | — | — | — | v2 更差 |
| `v3_ablation/` | `env_era5_v3.parquet` | 211.8 km | 207.7 km | 244.1 km | v3 更差 |

**注意**：`v1_2020_ablation` 显示 +0.1 km（无效），是因为只跑了 **30 epoch**。
报告 §3.3.2 证明该结论对轮次极敏感——**60 epoch 下为 −10.1 km（p=0.0005）**。
这组数据保留下来正是为了佐证"训练不足会得出相反结论"这一方法学陷阱。

| 其他目录 | 内容 | 关键数字 | 对应报告 |
|---|---|---|---|
| `latitude_adaptive/` | 纬度自适应分区，62 个 `pred_*.npz` + `policy_compare.csv` | track 329.7 / env 319.6 / **adapt 320.2 km** | §3.4（p=0.53，无收益） |
| `stratified_ablation/` | 分海域/强度层评估，23 个 `pred_*.npz` | — | §3.5 |
| `segment_compare/` | 分段策略 A/B 对比 | 见 `summary.json` | §3.6 |
| `segment_approaches/` | 分段建模 3 种方式 | 见 `summary.json` | §3.6 |
| `default/`、`vtest/` | **早期调试运行**（24h 178 / 236 km） | 已被 208.4 km 取代 | — |

---

## 二、强度预测

### 卫星云图（已定论：无效）

技术报告 §4.1.2 的四次尝试，**每一行都能对应到一个结果文件**，文件内 `_meta`
字段标注了该行编号：

| 报告行 | 结果文件 | 配置 | ΔMAE | 胜率 |
|---|---|---|---|---|
| #1 | `intensity_rerun_base/cv5_summary.json` | 90 台风 / 32px / `/255` / 15 维 | +0.177 | 1/15 |
| #2 | `sat_intensity/cv5_32px-255norm-15feat_summary.json` | 224 台风 / 32px / `/255` / 15 维 | +0.110 | 0/15 |
| #3 | `sat_intensity/cv5_satellite-model-dropout-0-25_*.json` | 224 台风 / 32px / dropout 0.25 | +0.116 | 0/15 |
| **#4** | **`sat_intensity/cv5_summary.json`** | **224 台风 / 64px / z-score / 23 维** | **+0.109** | **0/15** |

查任意文件的配置与对应报告位置：

```bash
.venv/bin/python -c "import json;print(json.load(open('outputs/sat_intensity_history/cv5_summary.json'))['_meta'])"
```

> **命名规则**：脚本输出 `cv{N}_{配置指纹}_{hash}_summary.json`，指纹只含影响结果的
> `satellite.*` 参数（不含 device/run_name）；无覆盖参数时退回 `cv{N}_summary.json`。
> 因此 `cv5_summary.json` 永远是**默认配置**的结果，重新以默认配置运行会覆盖它。

### 目录名约定（2026-10-30 起）

| 目录 | 性质 | 说明 |
|---|---|---|
| `sat_intensity_history/` | **归档，只读** | 上表 #1~#4 的证据都在这里。混装 32px 与 64px 各轮，不要再往里写新结果 |
| `sat_intensity_64px/` | **当前默认配置的输出** | 由 `satellite.run_name` 决定，初始为空，跑实验时才创建 |

改动 `patch_size` 或特征维数后，**请一并修改 `configs/default.yaml` 的
`satellite.run_name`**（如 `sat_intensity_32px`），否则新一轮结果会覆盖
`sat_intensity_64px/` 里的同名文件。

**历史提醒**：#2 的数据原本只存在于 `.scratch/`（未入库、随时可能被清理），
2026-10-30 整理时补存为正式文件并补了元数据。`sat_intensity_history/summary.json`
是更早的单次划分结果，仅供追溯。

### 地形 / 暖心特征消融（结论：均无影响）

存在两批重复实验，**新批次以 `intensity_rerun_*` 为准**：

| 旧批次（2026-10-28） | 新批次（2026-10-29） | 平均 MAE |
|---|---|---|
| `cv_base/` | `intensity_rerun_base/` | 1.786 / 1.803 |
| `cv_no_terrain/` | —（与 `cv_base` 逐字节相同） | 1.786 |
| `cv_with_terrain/` | `intensity_rerun_terrain/` | 1.869 / 1.835 |
| `cv_warmcore/` | `intensity_rerun_warmcore/` | 1.816 / 1.830 |
| `no_terrain/`、`with_terrain/` | 单次切分版本 | — |

汇总见 `intensity_rerun_summary.json`。两批结论一致：**地形、暖心均无增益**。

---

## 三、RI 快速增强

| 文件 | 内容 | 对应报告 |
|---|---|---|
| `ri_events/cv10_thr10_summary.json` | 10 折，阈值 10 m/s（正样本 11.7%） | §4.2 |
| `ri_events/cv10_thr15_summary.json` | 10 折，阈值 15 m/s | §4.2 |

含 `paired_tests` 字段，可直接查看配对检验结果。

---

## 四、报告配图

`figures/` 由脚本生成，**只用已保存的结果文件重画，不重新训练**：

```bash
MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python scripts/evaluation/plot_intensity_figures.py
```

| 文件 | 内容 | 报告位置 |
|---|---|---|
| `intensity_mae_vs_horizon.png` | 强度误差随时效（4 模型 + 误差棒） | 图 4-1（§4.1.1） |
| `cloud_ablation_rounds.png` | 云图四轮消融 ΔMAE + 胜率 | 图 4-2（§4.1.2） |
| `ri_auc_compare.png` | RI 的 AUC（2 阈值 × 4 模型） | 图 4-4（§4.2） |
| `feature_ablation_delta.png` | 地形/暖心特征消融 ΔMAE | 图 4-5（§4.5） |

图注编号与报告正文一致；数字全部从结果文件读取，脚本内无硬编码指标值，
因此图表与表格不会失配。§3 的 4 张图（图 3-1~3-4）仍由原路径脚本生成，
位于 `comparison/` 与 `baselines/`。

---

## 五、数据构建日志

`build_logs/` 记录了卫星 patch / 标量表的历次构建，对应技术报告 §2.4 的规模演进：

| 日志 | 台风 | 帧数 | patch | 说明 |
|---|---|---|---|---|
| `build_patches.log` | 43 | 2,907 | 32×32 | 2023-2024 首次构建 |
| `build_patches2.log` | 68 | 4,253 | 32×32 | 追加 2022 |
| `build_patches3.log` | 90 | 5,925 | 32×32 | 追加 2021（旧数据集最终版） |
| `build_scalars3.log` | 90 | — | — | 同批标量表 |

最终的 224 台风 / 17,193 帧 / 64×64 版本未保存日志文件，参数见技术报告 §2.4。

---

## 六、重复与冗余（已识别，未删除）

保留是因为它们可能被文档引用；如需清理可安全删除：

| 文件 | 与谁重复 |
|---|---|
| `cv_no_terrain/cv5_summary.json` | 与 `cv_base/cv5_summary.json` **逐字节相同** |
| `ablation_runs/v2_10seed.json` | 与 `env_ablation/parallel_summary.json` 逐字节相同 |
| `env_ablation/parallel_summary.json` | 内容与 `v3_ablation/parallel_summary.json` 相同（元数据被剥离） |
| `latitude_adaptive/pred_track_j0.npz` | 与 `pred_track_j0_0.npz` 相同 |
| `latitude_adaptive/pred_env_j0.npz` | 与 `pred_env_j0_0.npz` 相同 |

> `ablation_runs/v2_10seed.json` 的**命名有误**：其内容（baseline 211.8 km）与
> `v3_ablation` 一致，并非 v2。以 `v3_ablation/parallel_summary.json` 为准。

---

## 七、关于权重文件

本目录原有 **46 个 `best.pt`（共 231 MB，占全部体积 99%）已于 2026-10-30 清除**。

依据：
1. `.gitignore` 中已声明 `*.pt` 为训练产物、不入库；
2. 全仓库检索确认**没有任何脚本读取这些权重**——所有 `load_state_dict()`
   都使用训练过程中的内存副本 `best_state`，评估与出图只用 `logs/summary.json`
   和 `pred_*.npz`；
3. 全部可由原命令重跑生成。

因此如需复现某项结果，重跑对应训练命令即可；仅"重新评估已有模型"这一操作
需要重新训练（约原训练时间）。

---

## 八、目录命名与脚本默认值的对应关系

以下 `--out` 默认值写死在脚本里，**改目录名会破坏默认命令**，故保持原名：

| 脚本 | 默认 `--out` |
|---|---|
| `scripts/path/run_ablation_gpu.py` | `outputs/parallel` |
| `scripts/path/stratified_ablation.py` | `outputs/stratified_ablation` |
| `scripts/path/latitude_adaptive.py` | `outputs/latitude_adaptive` |

`v1_2020_ablation` / `v2_ablation` / `v3_ablation` 是上述脚本用不同
`--out` 参数跑出的结果，命名为区分特征版本，不遵循默认值。
