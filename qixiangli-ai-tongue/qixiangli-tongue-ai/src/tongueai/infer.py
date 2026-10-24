from __future__ import annotations
import hashlib
import numpy as np
from . import __version__
from .constitution import score_constitutions
from .labels import SINGLE_HEADS, MULTI_KEYS, MULTI_ZH, HEAD_ZH, SCHEMA_VERSION
from .preprocess import prepare, load_segmenter

DISCLAIMER = "本结果为 AI 辅助的舌象特征分析,仅供科研/教学/健康参考,不构成医疗诊断或治疗建议。"
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class Predictor:
    """ckpt=None 进入 mock 模式(随机输出,仅用于对接联调,结果带 mock=true 标记)。"""
    def __init__(self, ckpt: str | None = None, device: str | None = None):
        self.mock = ckpt is None
        self.model_version = "mock" if self.mock else str(ckpt)
        if not self.mock:
            import torch
            from .model import TongueNet
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            blob = torch.load(ckpt, map_location="cpu")
            self.size = blob.get("size", 224)
            self.model = TongueNet(blob["backbone"], pretrained=False)
            self.model.load_state_dict(blob["state"]); self.model.to(self.device).eval()

    def _feats(self, rgb: np.ndarray) -> dict:
        if self.mock:
            rng = np.random.default_rng(int(hashlib.md5(rgb.tobytes()).hexdigest()[:8], 16))
            return {"single": {h: dict(zip(c, rng.dirichlet(np.ones(len(c))).tolist())) for h, c in SINGLE_HEADS.items()},
                    "multi": {k: float(rng.random()) for k in MULTI_KEYS}}
        torch = self.torch
        x = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        with torch.no_grad():
            out = self.model(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
        single = {h: dict(zip(c, torch.softmax(out[h], 1)[0].cpu().tolist())) for h, c in SINGLE_HEADS.items()}
        multi = dict(zip(MULTI_KEYS, torch.sigmoid(out["multi"])[0].cpu().tolist()))
        return {"single": single, "multi": multi}

    def analyze(self, bgr: np.ndarray) -> dict:
        rgb, quality = prepare(bgr, getattr(self, "size", 224), load_segmenter())   # 可能抛 TongueNotFound
        f = self._feats(rgb)
        features = {}
        for h, probs in f["single"].items():
            lab = max(probs, key=probs.get)
            features[h] = {"name": HEAD_ZH[h], "label": lab, "confidence": round(probs[lab], 4),
                           "probs": {k: round(v, 4) for k, v in probs.items()}}
        for k, p in f["multi"].items():
            features[k] = {"name": MULTI_ZH[k], "label": bool(p >= 0.5), "confidence": round(max(p, 1 - p), 4),
                           "probs": {"true": round(p, 4)}}
        return {"schema": SCHEMA_VERSION, "engine_version": __version__, "model": self.model_version,
                "mock": self.mock, "quality": quality, "features": features,
                "constitution": score_constitutions(f), "disclaimer": DISCLAIMER}
