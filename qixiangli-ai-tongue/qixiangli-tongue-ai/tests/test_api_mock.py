import cv2, numpy as np
import pytest
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from tongueai.api import app


def fake_tongue():
    img = np.full((480, 480, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 260), (110, 150), 0, 0, 360, (90, 90, 200), -1)
    img = cv2.add(img, np.random.default_rng(0).integers(0, 25, img.shape, dtype=np.uint8))
    return cv2.imencode(".png", img)[1].tobytes()


def test_analyze_mock():
    c = TestClient(app)
    r = c.post("/v1/tongue/analyze", files={"file": ("t.png", fake_tongue(), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["mock"] is True and len(j["constitution"]["scores"]) == 9


def test_triage_flags_collage():
    from tongueai.preprocess import triage
    img = np.full((300, 900, 3), 120, np.uint8)
    assert "extreme_aspect_possible_collage" in triage(img)["triage_flags"]
