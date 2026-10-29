# AI Tongue Diagnosis (AI舌诊) · Qixiangli Tongue AI

**English** | [简体中文](README.zh-CN.md)

An AI system for structured analysis of tongue images in Traditional Chinese Medicine (TCM). Using computer vision and deep learning, it automatically identifies key tongue-diagnosis features — **tongue colour, tongue shape, coating colour and coating texture** — and, based on the TCM theory of **nine body constitutions**, outputs quantified, standardised results. It can be deployed locally or integrated through a REST API.

This project is the core tongue-analysis module of the **Qixiangli Health Tech** platform, intended for research, teaching and industrial applications.

> ⚠️ **Disclaimer:** Outputs are AI-assisted tongue-feature analyses for research, education and general wellness reference only. They are **not a medical diagnosis or treatment advice**. Before any clinical or commercial use, complete data-compliance work and clinical validation, and assess medical-device regulatory requirements in your jurisdiction (e.g. UK MHRA, China NMPA).

---

## Key Capabilities

| Capability | Description |
|---|---|
| Image quality control | Detects blur, under/over-exposure and low resolution; flags suspected collages |
| Pre-processing | White-balance colour normalisation, tongue segmentation, standardised cropping |
| Multi-task recognition | Tongue colour, tongue size, coating colour / thickness / moisture / greasiness, teeth marks, cracks, prickles |
| Constitution matching | Match scores against the nine TCM constitutions (Balanced, Qi-deficiency, Yang-deficiency, Yin-deficiency, Phlegm-dampness, Damp-heat, Blood-stasis, Qi-stagnation, Special-diathesis) |
| Standardised output | Unified JSON schema with confidence scores, quality results and version info |
| Flexible deployment | Local / Docker, REST API; segmentation and recognition models are replaceable |

## Pipeline

```
Image capture → QC → White balance → Tongue segmentation → Multi-task recognition → Constitution matching → Standardised JSON
```

## Project Status

This is an early-stage project. Please note the maturity of each module:

| Module | Status |
|---|---|
| Quality control, white balance, intake triage | ✅ Working |
| Tongue segmentation | ⚠️ Built-in GrabCut is a baseline only and is unreliable on cluttered backgrounds; integrating SAM or a self-trained model is recommended |
| Recognition model and training scripts | ✅ Code ready; ❌ no pretrained weights yet — requires annotated data |
| Nine-constitution mapping | ⚠️ Rule weights are expert-style priors and **not yet calibrated on clinical data**; the output is not a constitution diagnosis |
| REST API / Docker | ✅ Working; runs in mock mode (random output, for integration testing only) when no weights are loaded |

> Tongue appearance is only supporting evidence for constitution assessment. Constitution determination should rely primarily on the standard questionnaire (ZYYXH/T 157-2009, *Classification and Determination of Constitution in TCM*) and clinician consultation.

## Quick Start

```bash
git clone https://github.com/<your-org>/qixiangli-ai-tongue-diagnosis.git
cd qixiangli-ai-tongue-diagnosis
pip install -r requirements.txt
export PYTHONPATH=src

pytest                                        # run tests
uvicorn tongueai.api:app --port 8000          # start service (mock mode without weights)
curl -F "file=@tongue.jpg" http://localhost:8000/v1/tongue/analyze
```

### Docker

```bash
docker build -t qixiangli-tongue .
docker run -p 8000:8000 -v $PWD/runs/exp1:/models qixiangli-tongue
```

### Environment Variables

| Variable | Description |
|---|---|
| `TONGUEAI_CKPT` | Path to model weights; mock mode if unset |
| `TONGUEAI_API_KEY` | API key checked via the `X-API-Key` header |
| `TONGUEAI_MAX_MB` | Maximum upload size (default 10) |
| `TONGUEAI_SEGMENTER` | Custom segmenter as `module:function`, signature `fn(bgr) -> uint8 mask` |

## Example Output

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

## Training Your Own Model

```bash
# 1. Batch pre-processing: raw images -> tongue crops
python -m tongueai.preprocess --src raw_images/ --dst processed/

# 2. Intake triage: quality report and annotation sheet
python scripts/make_manifest.py --src raw_images/ --out manifest.csv

# 3. Annotate following docs/data_spec.md to obtain labels.csv

# 4. Train (patient-level train/val/test split to prevent data leakage)
python -m tongueai.train --csv labels.csv --root processed/ --out runs/exp1
```

See [`docs/data_spec.md`](docs/data_spec.md) for data collection and annotation guidelines.

## Repository Structure

```
src/tongueai/   preprocess · labels · dataset · model · train · infer · constitution · api
scripts/        intake triage and manifest generation
docs/           data collection and annotation specification
tests/          constitution-rule and API tests
```

## Data & Privacy

- Tongue photos may contain faces and personal health information. **Never commit them to this repository** (`.gitignore` excludes data and model files by default).
- Obtain **informed consent** before collection; annotation by at least two qualified TCM practitioners is recommended.
- Distribute model weights via Hugging Face or GitHub Releases.

## Roadmap

- [ ] Collect consented clinical data and establish double-annotation workflow
- [ ] Integrate SAM / self-trained U-Net segmentation to replace the GrabCut baseline
- [ ] Automatic routing of dorsal (tongue-surface) vs sublingual images
- [ ] Calibrate constitution mapping with paired questionnaire + tongue data
- [ ] Multimodal LLM for pattern-differentiation descriptions with TCM knowledge retrieval
- [ ] ONNX / TensorRT acceleration, mobile and multi-device access, model version management

## License

To be determined (add a LICENSE file, e.g. Apache-2.0, before release).

## Contact

Qixiangli Health Tech · Issues and collaboration enquiries are welcome.
