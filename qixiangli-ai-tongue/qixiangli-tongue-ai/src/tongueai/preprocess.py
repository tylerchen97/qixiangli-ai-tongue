"""采集质检 + 色彩归一 + 舌体分割 + 裁剪。

分割默认使用 GrabCut(无需训练数据,可作基线)。
有标注后建议换成 U-Net/SAM 的 ONNX 模型:实现 segment_tongue 同签名函数即可。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np


class TongueNotFound(ValueError):
    pass


def quality_check(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 先缩放到统一尺度再算清晰度,否则分辨率/压缩程度会主导该指标
    k = 512 / max(gray.shape)
    g512 = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(g512, cv2.CV_64F).var())
    bright = float(gray.mean())
    issues = []
    if min(gray.shape) < 224:
        issues.append("resolution_too_low")
    if blur < 15:
        issues.append("image_blurry")
    if bright < 60:
        issues.append("too_dark")
    if bright > 200:
        issues.append("overexposed")
    return {"blur_score": round(blur, 1), "brightness": round(bright, 1),
            "issues": issues, "pass": not issues}


def gray_world(bgr: np.ndarray) -> np.ndarray:
    """灰度世界白平衡,缓解不同光源色偏。有标准色卡时应改用色卡校正。"""
    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    f *= means.mean() / (means + 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def segment_tongue(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = 512 / max(h, w)
    small = cv2.resize(bgr, (int(w * s), int(h * s))) if s < 1 else bgr.copy()
    sh, sw = small.shape[:2]
    rect = (int(sw * 0.12), int(sh * 0.12), int(sw * 0.76), int(sh * 0.76))
    mask = np.zeros((sh, sw), np.uint8)
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(small, mask, rect, bg, fg, 4, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        raise TongueNotFound("no foreground found")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(lab == k, 255, 0).astype(np.uint8)
    if m.mean() / 255 < 0.03:
        raise TongueNotFound("foreground too small")
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def crop_square(bgr: np.ndarray, mask: np.ndarray, size: int = 224) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = bgr.copy()
    img[mask == 0] = 0
    img = img[y0:y1, x0:x1]
    hh, ww = img.shape[:2]
    side = max(hh, ww)
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[(side - hh) // 2:(side - hh) // 2 + hh, (side - ww) // 2:(side - ww) // 2 + ww] = img
    return cv2.cvtColor(cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def load_segmenter(spec: str | None = None):
    """spec 形如 "mypkg.mymod:fn"(环境变量 TONGUEAI_SEGMENTER);fn(bgr)->uint8 掩码(0/255)。
    用于接入 SAM / 自训 U-Net,缺省使用内置 GrabCut 基线。"""
    import importlib, os
    spec = spec or os.getenv("TONGUEAI_SEGMENTER")
    if not spec:
        return segment_tongue
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def triage(bgr: np.ndarray) -> dict:
    """入库分诊:标出不适合直接送入「舌面」模型的图片(需人工复核)。"""
    h, w = bgr.shape[:2]
    flags = []
    if max(h, w) / max(1, min(h, w)) > 1.9:
        flags.append("extreme_aspect_possible_collage")
    # 拼图常见特征:图中出现贯穿的纯色分隔带
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    col_std, row_std = g.std(0), g.std(1)
    inner_c, inner_r = col_std[int(w * .15):int(w * .85)], row_std[int(h * .15):int(h * .85)]
    if (inner_c < 3).any() or (inner_r < 3).any():
        flags.append("separator_line_possible_collage")
    if max(h, w) < 480:
        flags.append("low_resolution")
    return {"width": w, "height": h, "triage_flags": flags}


def prepare(bgr: np.ndarray, size: int = 224, segment_fn=segment_tongue):
    """返回 (RGB 舌体裁剪图, 质检结果)。"""
    q = quality_check(bgr)
    bgr = gray_world(bgr)
    mask = segment_fn(bgr)
    return crop_square(bgr, mask, size), q


def _cli():
    ap = argparse.ArgumentParser(description="批量预处理:原图 -> 舌体裁剪图")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for p in sorted(Path(a.src).rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(p))
        try:
            rgb, _ = prepare(img, a.size)
            cv2.imwrite(str(dst / f"{p.stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            ok += 1
        except Exception as e:
            bad += 1
            print("skip", p.name, e)
    print(f"done ok={ok} skipped={bad}")


if __name__ == "__main__":
    _cli()
