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
| `comparison/` | **四模型汇总**（`metrics.md` + 图） | — | — | §3.2 | `scripts/path/compare.py` |
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
| `latitude_adaptive/` | 纬度自适应分区，`policy_compare.csv`（`pred_*.npz` 不入库，见下方说明） | track 329.7 / env 319.6 / **adapt 320.2 km** | §3.4（p=0.53，无收益） |
| `stratified_ablation/` | 分海域/强度层评估（`pred_*.npz` 不入库） | — | §3.5 |
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

### 地形 / 暖心特征消融（224 台风为当前结论）

**当前结论以 `intensity_rerun_224_*` 为准**（2026-10-30，224 台风，15 job）：

| 变体 | gru | CNN+云图 | 特征+云图 | 结论 |
|---|---|---|---|---|
| `intensity_rerun_224_base/` | 1.533±0.069 | 1.642±0.087 | 1.711±0.076 | — |
| `intensity_rerun_224_warmcore/` | +0.031（p=0.0007） | +0.067（p=0.0004） | +0.025（p=0.026） | **暖心显著有害** |
| `intensity_rerun_224_terrain/` | −0.002（p=0.83） | −0.008（p=0.60） | −0.019（p=0.0496） | 地形无影响 |

汇总见 `intensity_rerun_224_summary.json`，逐变体日志见 `intensity_rerun_224_*.log`。

> **base 可复现性已核对**：`intensity_rerun_224_base` 与主结果图所用的
> `sat_intensity_history/cv5_summary.json` 四位小数完全一致（Δ=+0.0000）。

**90 台风时代的旧结果**（结论不同，仅作历史对照）：

| 目录 | 说明 | 当时结论 |
|---|---|---|
| `intensity_rerun_base/` | 90 台风基线（也是报告 §4.1.2 第 1 行的证据） | — |
| `intensity_rerun_warmcore/` | 90 台风 +暖心 | +0.010，p=0.42「无影响」 |
| `intensity_rerun_terrain/` | 90 台风 +地形 | +0.009，p=0.55「无影响」 |
| `cv_base/`、`cv_no_terrain/`、`cv_with_terrain/`、`cv_warmcore/` | 更早批次（10-28），已被取代 | — |
| `no_terrain/`、`with_terrain/` | 单次切分版本 | — |

> ⚠️ **暖心结论在数据扩到 224 台风后反转**（90 台风 p=0.42「无影响」→
> 224 台风 p=0.0007「显著更差」）。引用时务必看清用的是哪个目录。

---

## 三、RI 快速增强

| 文件 | 内容 | 数据规模 | 对应报告 |
|---|---|---|---|
| **`ri_events_64px/cv10_thr15_summary.json`** | **当前结果**：10 折，阈值 15 m/s | **224 台风**（正样本 4.8%） | §4.2.1 |
| **`ri_events_64px/cv10_thr10_summary.json`** | **当前结果**：10 折，阈值 10 m/s | **224 台风**（正样本 12.3%） | §4.2.2 |
| `ri_events_history/*.json` | 90 台风时代的旧结果，仅供对照 | 90 台风 | — |

含 `paired_tests` 字段（AUC / PR-AUC / CSI 各一组配对 t + Wilcoxon）。

两个阈值现在都是 224 台风结果，可横向比较。

> ⚠️ 切换阈值要用 `--set satellite.ri.threshold_ms=10`，脚本**没有** `--threshold` 参数
> （曾给错命令导致执行失败）。输出目录由 `satellite.ri.run_name` 决定（当前 `ri_events_64px`）。

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

## 五、日志与目录结构

本目录的结构约定（2026-11-01 整理后）：

| 位置 | 内容 |
|---|---|
| `logs/` | 13 个**散落的训练日志**（微波 `mw_*.log`、RI 概率 `ri_prob*.log`、强度重跑） |
| `build_logs/` | 数据构建过程日志 |
| `physics_experiments/` | 物理约束实验的全部产物（见 §九） |
| 顶层 `*_summary.json` | **被脚本读取**的汇总（`acc_report.py` 依赖前两个，勿移动） |

> `intensity_rerun_224_corr_summary.json` 与 `intensity_rerun_224_summary.json`
> 是 `scripts/evaluation/acc_report.py` 的输入（默认路径），**不能挪走**。

### 5.2 卫星 patch / 标量表构建记录

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
   都使用训练过程中的内存副本 `best_state`；
3. 全部可由原命令重跑生成。

**仓库里实际保留了哪些**：

| 类型 | 是否入库 | 说明 |
|---|---|---|
| `*.pt` / `*.ckpt` 权重 | ❌ | 可重跑 |
| `*.npz` 逐样本预测 | ❌ | 约 4 MB/个，可重跑 |
| `summary.json` / `*.csv` | ✅ | 指标与汇总，体积小 |
| `*.png` 图表 | ✅ | 报告引用的图 |
| `*.log` | ✅ | 完整训练记录 |

因此如需复现某项结果，重跑对应训练命令即可；仅"重新评估已有模型"这一操作
需要重新训练（约原训练时间）。**依赖逐样本预测的细分分析**（如 §4.9 的
分层验证）同样需要重跑训练来生成 `.npz`，具体命令见第十节。

> **2026-11-01 起有一个例外**：`scripts/evaluation/perturb_ensemble.py`
> 需要 `--ckpt` 读入权重（这是第一个真正读 `.pt` 的脚本）。因此保留
> **2 个**权重供其复用：`physics_experiments/phys_base/checkpoints/best.pt`
> 与 `physics_experiments/phys_r0.1/checkpoints/best.pt`。
> 它们同样不入库（`.gitignore` 的 `*.pt` 规则未变）。其余权重仍按上述政策清理。

---

## 八、目录命名与脚本默认值的对应关系

> **环境场文件位置**：完整版在仓库根的 `data_ext/env_era5.parquet`
> （128 台风 / 5866 行）。`data/processed/env_era5.parquet` 在本机是只读挂载上的
> **5 台风样例（0.4% 覆盖）**，用它跑环境场消融会得到近乎退化的结果。
> 文档中的 `--env` 参数已统一改为 `data_ext/env_era5.parquet`。

以下 `--out` 默认值写死在脚本里，**改目录名会破坏默认命令**，故保持原名：

| 脚本 | 默认 `--out` |
|---|---|
| `scripts/path/run_ablation_gpu.py` | `outputs/parallel` |
| `scripts/path/stratified_ablation.py` | `outputs/stratified_ablation` |
| `scripts/path/latitude_adaptive.py` | `outputs/latitude_adaptive` |

`v1_2020_ablation` / `v2_ablation` / `v3_ablation` 是上述脚本用不同
`--out` 参数跑出的结果，命名为区分特征版本，不遵循默认值。

---

## 九、物理约束与扰动集合（2026-11-01 新增）

全部产物在 **`physics_experiments/`**，该目录的
[README.md](physics_experiments/README.md) 含完整结果表与有效性判定。

| 分组 | 目录 | 结论 |
|---|---|---|
| 基线 / 硬约束 / 权重扫描 | `phys_base` `phys_hard` `phys_r*` | 硬约束**有害**（+5.2 km，p=0.042，3 seed 一致） |
| 多 seed 对照 | `phys_seed_base_*` `phys_seed_w0.1_*` | 3/3 一致为正，d=+2.73 |
| 无效实验（权重 bug） | `phys_full` `phys_w0.01/0.03/0.3` | 物理项仅占 0.04~1.07%，等于没加 |

**权重保留**：仅 `phys_base` 与 `phys_r0.1` 的 `best.pt`（供 `perturb_ensemble.py` 复用），
其余 13 个已删（69.3 MB）。理由见 §七与上述 README。

**散落的 15 个 `phys_*.log`** 已一并归入 `physics_experiments/`。

---

## 十、分时效相关与分层验证（2026-10-31 新增）

| 文件 | 内容 | 用途 |
|---|---|---|
| `figures/fig_acc_by_lead.png` | 路径：环境场 vs 纯轨迹的距平相关随时效 | 技术报告 §4.8 |
| `figures/fig_sat_acc_by_lead.png` | 强度：GRU vs 云图的距平相关随时效 | §4.7.8 |
| `figures/fig_stratify_change.png` | 按强度变化幅度分层 + 云图 ΔMAE | §4.9 |
| `intensity_rerun_224_corr_summary.json` | **首次带 `corr_by_step`** 的强度消融结果 | §4.7.8 |
| `intensity_rerun_224_pred_base/cv5_summary.npz` | **逐样本预测**（含 `last` 与 `job_offsets`） | 分层验证输入 |

**重新生成**：

```bash
python scripts/evaluation/acc_report.py            # 距平相关表
python scripts/evaluation/plot_acc_figures.py      # 三张图
python scripts/evaluation/stratify_by_change.py \
    --npz outputs/intensity_rerun_224_pred_base/cv5_summary.npz
```

**注意**：`intensity_rerun_224_corr_*` 与 `intensity_rerun_224_pred_*` 的 MAE
与旧目录 `intensity_rerun_224_*` **完全一致**（同一配置重跑，仅多了新字段），
可以互相替代；旧目录保留用于对照。

**`.npz` 不入库**：`.gitignore` 里的 `outputs/**/*.npz` 会排除所有逐样本预测数组
（本次 `intensity_rerun_224_pred_base/cv5_summary.npz` 也**不会**被提交）。
这样做的理由：单个文件约 4 MB，且完全可由训练重跑生成。

**复现分层验证需要重跑**：

```bash
python scripts/intensity/rerun_ablations.py --folds 5 --fold-seeds 42 0 7 \
    --variants base --out intensity_rerun_224_pred
# -> 自动生成 cv5_summary.npz（含 pred/true/last/job_offsets）
python scripts/evaluation/stratify_by_change.py \
    --npz outputs/intensity_rerun_224_pred_base/cv5_summary.npz
```

`summary.json` 与日志**会**入库，所以 §4.9 的分层数字可以从仓库里的
`figures/fig_stratify_change.png` 复核；只有逐样本的细分计算需要重跑。

---

## 十一、无标签物理律（2026-11-01 新增）

| 目录 | 配置 | 结论 |
|---|---|---|
| `lawA_w0.0` `lawA_w0.1` `lawA_w0.5` | 全量 20094，seed 42，smooth+curvature | 119.04 / **118.32** / 120.76（w=0.5 过强） |
| `lawA_s{0,1,2}_w{0.0,0.1}` | 全量，3 个种子 | 4/4 seed 一致 **−1.26 km**（p=0.13）；种子方差 1.48→0.32 |
| `lawB_w0.0` `lawB_w0.3` | 环境场子集 1644 + storm 划分，三项全开 | w=0.3 更差 **+5.67 km**（6h +8.13，p<0.001） |

每个目录内含 `checkpoints/best.pt`（不入库）、`logs/summary.json`、`logs/history.json`、
`used_config.yaml`；同名 `.log` 是完整训练日志。结论与机制见技术报告 §4.13。

**要点**：smooth/curvature 只约束"形状"，而两组的主导误差都是沿轨欠位移
（B 组 24h 预测 345 km vs 实况 572 km）；steering 先验自身 6h 误差 140.6 km
（模型 65.3），只在 24h 略好 11 km，因此不适合做锚点。

**汇总表与配图**（从 checkpoint 重新推理，不依赖手工数字）：

```bash
python scripts/evaluation/law_summary.py            # -> physics_laws/law_summary.json
python scripts/evaluation/plot_physics_law_figures.py
# -> figures/fig_lawA_delta_by_seed.png
#    figures/fig_lawA_step_profile.png
#    figures/fig_lawB_diagnosis.png
```

`law_summary.py` 会把重新推理的 MAE 与各 run 的 `logs/summary.json` 逐项对账，
不一致直接抛错（防止报告引用了错误的 checkpoint）。配对 bootstrap 以**台风**为单位重采样。

**B 组复现需注意**：必须 `--env data_ext/env_era5.parquet --split storm`。
环境场只覆盖 2017-2021 的 128 个台风，按年份切会得到空验证集（`val_mae=nan`）。

