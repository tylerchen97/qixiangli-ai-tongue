# Qixiangli Tongue AI · 舌诊智能分析引擎

面向中医舌诊场景的舌象结构化识别与服务系统:**采集质检 → 色彩归一 → 舌体分割 → 多任务识别 → 九种体质匹配 → 标准化 JSON 输出**,支持本地部署与 API 集成。

> ⚠️ 本项目输出仅供科研、教学与健康参考,**不构成医疗诊断**。用于临床/商业前请完成数据合规、临床验证,并按所在地法规(如中国 NMPA)评估医疗器械注册要求。

## 当前状态(请务必阅读)
| 模块 | 状态 |
|---|---|
| 质检、白平衡、入库分诊 | ✅ 可用 |
| GrabCut 舌体分割 | ⚠️ 仅基线:在真实自拍上会带入嘴唇/皮肤,须替换为 SAM 或自训模型 |
| 多任务识别模型 + 训练脚本 | ✅ 代码就绪;❌ **无预训练权重**,需自备标注数据训练 |
| 九体质规则引擎 | ⚠️ 权重为经验先验,**未经临床校准** |
| REST API / Docker | ✅ 可用;无权重时为 mock 模式(随机输出,仅供联调) |

## 识别要素
舌色、舌形(胖瘦)、苔色、苔厚薄、苔润燥、苔腻腐、齿痕、裂纹、点刺(见 `labels.py`)。

## 快速开始
```bash
pip install -r requirements.txt
export PYTHONPATH=src
pytest
uvicorn tongueai.api:app --port 8000        # 无权重 => mock 模式
curl -F "file=@tongue.jpg" localhost:8000/v1/tongue/analyze
```

## 训练与部署
```bash
python -m tongueai.preprocess --src raw_images/ --dst processed/
# 按 docs/data_spec.md 准备 labels.csv
python -m tongueai.train --csv labels.csv --root processed/ --out runs/exp1
TONGUEAI_CKPT=runs/exp1/best.pt uvicorn tongueai.api:app --host 0.0.0.0 --port 8000
```
Docker:`docker build -t qixiangli-tongue . && docker run -p 8000:8000 -v $PWD/runs/exp1:/models qixiangli-tongue`

可选环境变量:`TONGUEAI_API_KEY`(请求头 `X-API-Key`)、`TONGUEAI_MAX_MB`。

## 真实数据入库(分诊)
```bash
python scripts/make_manifest.py --src raw_images/ --out manifest.csv --sheet review.jpg
```
生成质检/分诊指标和待标注列(view、patient_id、各舌象标签),交中医师填写。
自定义分割器(SAM / 自训 U-Net):`TONGUEAI_SEGMENTER=mypkg.seg:segment`,函数签名 `fn(bgr)->uint8 掩码`。
内置 GrabCut 在真实自拍(带嘴唇、下巴、衣物)上不可靠,**正式使用前应替换**。

## 输出示例(节选)
```json
{"schema":"tongue-schema/1.0","mock":false,
 "quality":{"pass":true,"issues":[]},
 "features":{"tongue_color":{"name":"舌色","label":"淡红","confidence":0.91},
             "teeth_marks":{"name":"齿痕","label":true,"confidence":0.84}},
 "constitution":{"scores":[{"constitution":"气虚质","match":0.62,"reliability":"tongue_supported"}],
                 "leading":["气虚质"]},
 "disclaimer":"..."}
```

## 路线图
1. 与中医院合作采集带知情同意的数据,双人标注
2. 训练舌体分割模型(U-Net/SAM → ONNX),替换 GrabCut
3. 用"体质量表 + 舌象"配对数据校准体质映射
4. 多模态大模型(Qwen2-VL 等 LoRA)生成辨证描述 + 中医知识库 RAG
5. ONNX/TensorRT 加速、移动端接入、模型版本管理与审计日志
