"""uvicorn tongueai.api:app --host 0.0.0.0 --port 8000
环境变量: TONGUEAI_CKPT=模型路径(缺省=mock) TONGUEAI_API_KEY=接口密钥(缺省=不鉴权) TONGUEAI_MAX_MB=10"""
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from . import __version__
from .infer import Predictor
from .preprocess import TongueNotFound

app = FastAPI(title="Qixiangli Tongue AI", version=__version__)
_pred = None


def predictor() -> Predictor:
    global _pred
    if _pred is None:
        _pred = Predictor(os.getenv("TONGUEAI_CKPT"))
    return _pred


@app.get("/health")
def health():
    p = predictor()
    return {"status": "ok", "version": __version__, "model": p.model_version, "mock": p.mock}


@app.post("/v1/tongue/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    key = os.getenv("TONGUEAI_API_KEY")
    if key and x_api_key != key:
        raise HTTPException(401, "invalid api key")
    data = await file.read()
    if len(data) > float(os.getenv("TONGUEAI_MAX_MB", "10")) * 1024 * 1024:
        raise HTTPException(413, "file too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a valid image")
    try:
        return predictor().analyze(img)
    except TongueNotFound as e:
        raise HTTPException(422, f"tongue not detected: {e}")
