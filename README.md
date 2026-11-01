# AI Tongue Digital Phenotyping & Multimodal Biomarker Association Platform

# AI 舌象数字表型与多模态客观健康指标关联研究平台

**Language / 语言：** [English](#english) · [中文](#中文)

Maintainer / 维护者：Yunzi Chen · Qixiangli Health Technology

---

<a id="english"></a>

# English

## Overview

A research platform that uses computer vision, multimodal machine learning and longitudinal data analysis to study the **statistical association, predictive value and temporal dynamics** between digital tongue features and objective health indicators. It also provides an interface for **research-grade** TCM (Traditional Chinese Medicine) pattern-differentiation support.

> ⚠️ **Research use only.** This project is not a medical device. Outputs are not diagnostic or treatment advice. The syndrome rule weights are illustrative priors and must be reviewed by clinical experts. Any clinical interpretation requires prospective validation at independent sites. The bundled demo uses **synthetic data**; its associations and AUCs are **not real-world evidence**.

## Four core technologies → code modules

| Core technology                     | Module                       | Capabilities                                                                                                                                                                                                         |
| ----------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Digital Tongue Phenotyping       | `tcm_tongue/phenotyping.py`  | GrabCut tongue segmentation; 24 features: Lab colour, coating (clustering), GLCM texture, shape, tooth marks (convexity defects), cracks (black-hat), gloss; `qc_*` quality fields                                   |
| 2. Multimodal Biomarker Association | `tcm_tongue/association.py`  | Covariate-adjusted regression with subject-clustered robust SE and BH-FDR; linear mixed models separating within-/between-subject effects; CCA with permutation test                                                 |
| 3. Personalised Tongue Monitoring   | `tcm_tongue/personalised.py` | Personal baseline (median/MAD), z-scores, EWMA, two-sided CUSUM, multi-feature joint alarm                                                                                                                           |
| 4. Evidence-based AI Modeling       | `tcm_tongue/evidence.py`     | Site-wise development / external validation split; subject-grouped CV; AUC with cluster-bootstrap CI; calibration slope/intercept, Brier score; ΔAUC of tongue features over baseline covariates; leave-one-site-out |
| TCM pattern support                 | `tcm_tongue/syndrome.py`     | Interpretable rule scoring (qi deficiency, yin deficiency, damp-heat, phlegm-dampness, blood stasis, yang deficiency) plus a supervised classifier                                                                   |

## Quick start

```bash
git clone <your-repo-url> && cd tongue-phenotype-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .

pytest -q                                   # unit tests
python -m tcm_tongue demo --out outputs     # end-to-end synthetic demo (~1–2 min)
streamlit run app/streamlit_app.py          # interactive upload demo (run the demo first)
```

Demo outputs (`outputs/`): `summary.json`, `association_results.csv`, `longitudinal_mixed.csv`, `monitoring_timeseries.csv`, `validation.json`, `syndrome_rule_based.csv`, `figures/*.png`.

## Using real data

1. **Extract features:** `python -m tcm_tongue extract --images path/to/images --out features.csv`
2. **Prepare a metadata CSV** (one row per image):

| Column                        | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| `image_file`                  | Image file name, used to merge with `features.csv`  |
| `subject_id`, `visit`, `site` | Subject, numeric visit index, centre / device batch |
| covariates                    | e.g. `age`, `female`, `bmi`                         |
| objective biomarkers          | e.g. `crp`, `hba1c`, `hb`, `tg`, `alt`, `sbp`       |
| binary targets                | e.g. `elevated_crp`, `high_hba1c`, `anemia` (0/1)   |
| `syndrome_label` (optional)   | Expert TCM pattern label                            |

3. **Edit** `configs/default.yaml` (column names, development/external sites, targets), then run:
   `python -m tcm_tongue analyze --features features.csv --meta meta.csv --config configs/default.yaml --out outputs`

## Study-design guidance

- **Standardised capture:** fixed light, camera and distance; include a grey/colour card. Without cross-device colour calibration, `color_*` values are not directly comparable.
- **Independent validation:** the external set (different site / device / period) must never be used for fitting or tuning. Do not draw leave-one-site-out conclusions with too few sites.
- **Leakage control:** all visits of one subject stay in the same fold (grouped by `subject_id`).
- **Multiple testing:** judge associations by `q_fdr`, not raw p-values.
- **Personalised monitoring:** calibrate `cusum_h` / `min_alarm_features` on a prospective cohort with known events. Defaults give roughly 87–92% detection and ~22% false alerts on synthetic data and must be re-tuned for real settings.
- **Known limitations:** tooth marks, cracks and gloss are heuristic image proxies; GrabCut can fail on complex backgrounds (see `qc_ok`); rule weights are examples, not evidence. For production, replace segmentation with a U-Net / SAM trained on annotated data (keep the `segment_tongue(img) -> mask` interface).

## Repository layout

```
tcm_tongue/   core package (phenotyping / association / personalised / evidence / syndrome / synthetic / pipeline / cli)
configs/      analysis configuration
app/          Streamlit demo
tests/        unit tests
.github/      CI (pytest + demo smoke test)
```

## License

MIT © 2025 Yunzi Chen, Qixiangli Health Technology

---

<a id="中文"></a>

# 中文

## 项目简介

通过计算机视觉、多模态机器学习与纵向数据分析，研究**舌象数字特征与客观健康指标**之间的统计关联、预测价值和动态变化规律，并提供中医辨证的**辅助研究**接口。

> ⚠️ **研究用途声明**：本项目不是医疗器械，输出不构成诊断或治疗建议。辨证规则权重为示例先验，须经临床专家审订；任何临床解读均需以独立中心的前瞻验证为前提。内置演示数据为**合成数据**，其中的关联与 AUC 等数字**不代表任何真实世界证据**。

## 四大核心技术 ↔ 代码模块

| 核心技术             | 模块                           | 主要能力                                                                                  |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------------------- |
| 1. 舌象数字表型提取      | `tcm_tongue/phenotyping.py`  | GrabCut 舌体分割；24 个特征：Lab 颜色、舌苔（聚类）、GLCM 纹理、形态、齿痕（凸缺陷）、裂纹（黑帽）、光泽；`qc_*` 质控字段            |
| 2. 多模态指标关联建模     | `tcm_tongue/association.py`  | 协变量校正回归 + 受试者聚类稳健标准误 + BH-FDR；区分个体内/个体间效应的线性混合模型；CCA 置换检验                             |
| 3. 个体化舌象动态监测     | `tcm_tongue/personalised.py` | 个体基线（中位数/MAD）、z 分数、EWMA、双侧 CUSUM、多特征联合报警                                              |
| 4. 基于独立验证的 AI 建模 | `tcm_tongue/evidence.py`     | 按中心划分开发集/外部验证集；按受试者分组交叉验证；AUC 及聚类 bootstrap 置信区间；校准斜率/截距、Brier；舌象相对基础协变量的 ΔAUC；留一中心验证 |
| 中医辨证辅助           | `tcm_tongue/syndrome.py`     | 可解释规则打分（气虚、阴虚、湿热、痰湿、血瘀、阳虚）+ 有监督分类器                                                    |

## 快速开始

```bash
git clone <your-repo-url> && cd tongue-phenotype-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .

pytest -q                                   # 单元测试
python -m tcm_tongue demo --out outputs     # 合成数据端到端演示（约 1–2 分钟）
streamlit run app/streamlit_app.py          # 上传舌象的交互演示（需先跑 demo）
```

演示产出（`outputs/`）：`summary.json`、`association_results.csv`、`longitudinal_mixed.csv`、`monitoring_timeseries.csv`、`validation.json`、`syndrome_rule_based.csv`、`figures/*.png`。

## 用于真实数据

1. **提取特征：** `python -m tcm_tongue extract --images path/to/images --out features.csv`
2. **准备元数据 CSV**（每行一张图像）：

| 列                           | 说明                                          |
| --------------------------- | ------------------------------------------- |
| `image_file`                | 图像文件名，用于与 `features.csv` 合并                 |
| `subject_id`、`visit`、`site` | 受试者、数值型随访序号、中心/设备批次                         |
| 协变量                         | 如 `age`、`female`、`bmi`                      |
| 客观指标                        | 如 `crp`、`hba1c`、`hb`、`tg`、`alt`、`sbp`       |
| 二分类终点                       | 如 `elevated_crp`、`high_hba1c`、`anemia`（0/1） |
| `syndrome_label`（可选）        | 专家辨证标签                                      |

3. **修改** `configs/default.yaml`（列名、开发/外部中心、终点等），然后运行：
   `python -m tcm_tongue analyze --features features.csv --meta meta.csv --config configs/default.yaml --out outputs`

## 研究设计建议

- **拍摄标准化：** 固定光源、相机与距离，放置灰卡或色卡；跨设备未做色彩标定时，`color_*` 不可直接比较。
- **独立验证：** 外部集（不同中心/设备/时间段）不得参与任何拟合与调参；中心数过少时不要下留一中心验证的结论。
- **防止信息泄漏：** 同一受试者的所有访视必须在同一折（已按 `subject_id` 分组）。
- **多重比较：** 关联结果以 `q_fdr` 为准，而非原始 p 值。
- **个体监测：** 需在有真值事件的前瞻队列中标定 `cusum_h` / `min_alarm_features`；默认值在合成数据上约 87–92% 检出、约 22% 误报，真实场景须重新权衡。
- **已知局限：** 齿痕、裂纹、光泽为图像启发式代理指标；GrabCut 在复杂背景下可能失败（见 `qc_ok`）；规则权重仅为示例，并非循证结论。正式研究建议将分割替换为在标注数据上训练的 U-Net / SAM（保持 `segment_tongue(img) -> mask` 接口即可）。

## 目录结构

```
tcm_tongue/   核心包（phenotyping / association / personalised / evidence / syndrome / synthetic / pipeline / cli）
configs/      分析配置
app/          Streamlit 演示
tests/        单元测试
.github/      CI（pytest + 演示冒烟测试）
```

## 许可证

MIT © 2025 Yunzi Chen, Qixiangli Health Technology
