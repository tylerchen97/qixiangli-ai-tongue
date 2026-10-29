# AI舌诊 · AI Tongue Diagnosis(Qixiangli Tongue AI 舌诊智能分析引擎)

**[English](README.md)** | **简体中文**


面向中医舌诊场景的舌象结构化分析系统。基于计算机视觉与深度学习,对舌象图像进行**舌色、舌形、苔色、苔质**等核心要素的自动识别,结合**中医九种体质**理论输出量化、标准化的分析结果,支持本地部署与 API 集成。

本项目是 **Qixiangli Health Tech** 体系中舌诊智能分析能力的核心技术模块,适用于科研研究、教学实践与产业化应用。

> ⚠️ **免责声明**:本项目输出为 AI 辅助的舌象特征分析,仅供科研、教学与健康参考,**不构成医疗诊断或治疗建议**。用于临床或商业场景前,请完成数据合规与临床验证,并依所在地法规(如中国 NMPA)评估医疗器械注册要求。

---

## 核心能力

| 能力 | 说明 |
|---|---|
| 图像质检 | 检测模糊、过暗、过曝、分辨率不足,识别疑似拼图 |
| 预处理 | 白平衡色彩归一、舌体分割与标准化裁剪 |
| 多任务识别 | 舌色、舌形(胖瘦)、苔色、苔厚薄、苔润燥、苔腻腐,以及齿痕、裂纹、点刺 |
| 体质匹配 | 输出舌象与九种体质(平和、气虚、阳虚、阴虚、痰湿、湿热、血瘀、气郁、特禀)的匹配度 |
| 标准化输出 | 统一 JSON 结构,含置信度、质检结果与版本信息 |
| 灵活部署 | 本地 / Docker 部署,REST API 接入,分割与识别模型均可替换升级 |

## 技术流程

```
图像采集 → 质检 → 白平衡 → 舌体分割 → 多任务识别 → 九体质匹配 → 标准化 JSON
```

## 项目状态

项目处于早期阶段,请在使用前了解各模块的成熟度:

| 模块 | 状态 |
|---|---|
| 质检、白平衡、入库分诊 | ✅ 可用 |
| 舌体分割 | ⚠️ 内置 GrabCut 仅作基线,在复杂背景下效果有限;建议接入 SAM 或自训分割模型 |
| 多任务识别模型与训练脚本 | ✅ 代码就绪;❌ 暂无预训练权重,需使用标注数据训练 |
| 九体质映射 | ⚠️ 规则权重为经验先验,**尚未经临床数据校准**,不构成体质判定 |
| REST API / Docker | ✅ 可用;未加载权重时运行于 mock 模式(随机输出,仅供联调) |

> 舌象只是体质辨识的辅助依据,体质判定应以《中医体质分类与判定》(ZYYXH/T 157-2009)量表及医师问诊为主。

## 快速开始

```bash
git clone https://github.com/<your-org>/qixiangli-ai-tongue-diagnosis.git
cd qixiangli-ai-tongue-diagnosis
pip install -r requirements.txt
export PYTHONPATH=src

pytest                                        # 运行测试
uvicorn tongueai.api:app --port 8000          # 启动服务(无权重时为 mock 模式)
curl -F "file=@tongue.jpg" http://localhost:8000/v1/tongue/analyze
```

### Docker

```bash
docker build -t qixiangli-tongue .
docker run -p 8000:8000 -v $PWD/runs/exp1:/models qixiangli-tongue
```

### 环境变量

| 变量 | 说明 |
|---|---|
| `TONGUEAI_CKPT` | 模型权重路径;缺省为 mock 模式 |
| `TONGUEAI_API_KEY` | 接口密钥,通过请求头 `X-API-Key` 校验 |
| `TONGUEAI_MAX_MB` | 上传图片大小上限,默认 10 |
| `TONGUEAI_SEGMENTER` | 自定义分割器,格式 `模块:函数`,签名 `fn(bgr) -> uint8 掩码` |

## 输出示例

```json
{
  "schema": "tongue-schema/1.0",
  "mock": false,
  "quality": {"pass": true, "issues": []},
  "features": {
    "tongue_color": {"name": "舌色", "label": "淡红", "confidence": 0.91},
    "teeth_marks":  {"name": "齿痕", "label": true,   "confidence": 0.84}
  },
  "constitution": {
    "scores": [{"constitution": "气虚质", "match": 0.62, "reliability": "tongue_supported"}],
    "leading": ["气虚质"]
  },
  "disclaimer": "..."
}
```

## 训练自己的模型

```bash
# 1. 批量预处理:原图 → 舌体裁剪图
python -m tongueai.preprocess --src raw_images/ --dst processed/

# 2. 入库分诊:生成质检报告与待标注清单
python scripts/make_manifest.py --src raw_images/ --out manifest.csv

# 3. 按 docs/data_spec.md 完成标注,得到 labels.csv

# 4. 训练(按 patient_id 划分训练/验证/测试集,避免数据泄漏)
python -m tongueai.train --csv labels.csv --root processed/ --out runs/exp1
```

数据采集与标注规范见 [`docs/data_spec.md`](docs/data_spec.md)。

## 项目结构

```
src/tongueai/   preprocess · labels · dataset · model · train · infer · constitution · api
scripts/        入库分诊与清单生成
docs/           数据采集与标注规范
tests/          体质规则与接口测试
```

## 数据与隐私

- 舌象照片含人脸与个人健康信息,**请勿提交到仓库**(`.gitignore` 已默认排除数据与模型文件)
- 采集前须取得受试者**知情同意**;训练数据建议由至少两名有资质的中医师独立标注
- 模型权重请通过 Hugging Face 或 GitHub Releases 分发

## 路线图

- [ ] 采集带知情同意的临床数据,建立双人标注流程
- [ ] 接入 SAM / 自训 U-Net 分割模型,替换 GrabCut 基线
- [ ] 舌面 / 舌下图像自动分流
- [ ] 基于"体质量表 + 舌象"配对数据校准九体质映射
- [ ] 多模态大模型辨证描述生成,结合中医知识库检索增强
- [ ] ONNX / TensorRT 加速,移动端与多终端接入,模型版本管理

## 许可证

待定(发布前请补充 LICENSE 文件,如 Apache-2.0)。

## 联系

Qixiangli Health Tech · 欢迎通过 Issues 反馈问题与合作意向。
