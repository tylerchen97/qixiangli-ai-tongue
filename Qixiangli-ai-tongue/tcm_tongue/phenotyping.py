"""模块 1：Digital Tongue Phenotyping —— 舌象数字表型提取。

流程：舌体分割 -> 颜色 / 舌苔 / 纹理 / 形态 / 裂纹 / 齿痕 / 光泽 特征。
注意：特征以像素统计为基础，未做设备色彩标定时，跨设备绝对值不可直接比较。
建议拍摄时放置标准灰卡 / 色卡，并在 qc_* 指标上做质控。
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

FEATURE_PREFIXES = ("color_", "coat_", "tex_", "shape_", "crack_", "gloss_")

FEATURE_NAMES = [
    "color_L", "color_a", "color_b", "color_L_std", "color_a_std", "color_b_std", "color_sat",
    "coat_ratio", "coat_L", "coat_a", "coat_b",
    "tex_contrast", "tex_homogeneity", "tex_energy", "tex_correlation", "tex_lapvar",
    "shape_area_ratio", "shape_aspect", "shape_solidity", "shape_circularity",
    "shape_tooth_count", "shape_tooth_index",
    "crack_index", "gloss_ratio",
]


def load_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def _resize(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out


def _a_channel_fallback(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = cv2.GaussianBlur(lab[..., 1], (9, 9), 0)
    _, m = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def segment_tongue(img_bgr: np.ndarray) -> np.ndarray:
    """舌体分割：GrabCut（中心矩形初始化），失败时回退到 Lab a* 通道 Otsu。返回 0/255 掩膜。"""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        m = np.zeros((h, w), np.uint8)
    frac = m.mean() / 255.0
    if frac < 0.05 or frac > 0.9:
        m = _a_channel_fallback(img_bgr)
    return _clean_mask(m)


def _nan_features() -> dict:
    return {k: np.nan for k in FEATURE_NAMES}


def extract_features(img_bgr: np.ndarray, mask: np.ndarray | None = None, seed: int = 0) -> dict:
    """从一张舌象图像提取数字表型特征，返回 dict（含 qc_ 质控字段）。"""
    img = _resize(img_bgr)
    if mask is None:
        mask = segment_tongue(img)
    else:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = mask > 0
    area = int(m.sum())
    total = m.size
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    qc = {"qc_mask_frac": area / total, "qc_ok": 1}
    if area < 0.03 * total:
        qc["qc_ok"] = 0
        qc["qc_sharpness"] = np.nan
        return {**_nan_features(), **qc}
    qc["qc_sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F)[m].var())

    f: dict = {}
    # ---- 颜色（CIELab，真实尺度 L:0-100，a/b 以 0 为中心）----
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[..., 0] * 100.0 / 255.0, lab[..., 1] - 128.0, lab[..., 2] - 128.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f.update(color_L=L[m].mean(), color_a=a[m].mean(), color_b=b[m].mean(),
             color_L_std=L[m].std(), color_a_std=a[m].std(), color_b_std=b[m].std(),
             color_sat=hsv[..., 1][m].mean() / 255.0)

    # ---- 舌苔：舌体像素 2-means 聚类，取 a* 较低（较不红）的簇为苔 ----
    pix = np.stack([L[m], a[m], b[m]], axis=1)
    rng = np.random.default_rng(seed)
    sample = pix[rng.choice(len(pix), size=min(4000, len(pix)), replace=False)]
    km = KMeans(n_clusters=2, n_init=3, random_state=seed).fit(sample)
    c = km.cluster_centers_
    low = int(np.argmin(c[:, 1]))
    if abs(c[0, 1] - c[1, 1]) < 5.0:  # 两簇红度差异过小 -> 视为无明显苔
        f.update(coat_ratio=0.0, coat_L=f["color_L"], coat_a=f["color_a"], coat_b=f["color_b"])
    else:
        lab_pred = km.predict(pix)
        f.update(coat_ratio=float((lab_pred == low).mean()),
                 coat_L=float(c[low, 0]), coat_a=float(c[low, 1]), coat_b=float(c[low, 2]))

    # ---- 纹理：腐蚀后的舌体内部区域做 GLCM ----
    eq_d = np.sqrt(area)
    ks = max(3, int(0.12 * eq_d))
    er = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    ys, xs = np.where(er > 0)
    if len(ys) < 100:
        ys, xs = np.where(mask > 0)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    q = (crop // 8).astype(np.uint8)
    glcm = graycomatrix(q, [1, 2], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    for name, prop in [("tex_contrast", "contrast"), ("tex_homogeneity", "homogeneity"),
                       ("tex_energy", "energy"), ("tex_correlation", "correlation")]:
        f[name] = float(graycoprops(glcm, prop).mean())
    f["tex_lapvar"] = float(cv2.Laplacian(gray, cv2.CV_64F)[er > 0].var()) if (er > 0).any() else np.nan

    # ---- 形态 + 齿痕（凸缺陷） ----
    smooth = (cv2.GaussianBlur(mask, (5, 5), 0) > 127).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    carea = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    harea = max(cv2.contourArea(hull), 1.0)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    f.update(shape_area_ratio=area / total,
             shape_aspect=float(min(rw, rh) / max(rw, rh, 1e-6)),
             shape_solidity=float(carea / harea),
             shape_circularity=float(4 * np.pi * carea / max(per ** 2, 1e-6)))
    tooth_cnt, tooth_idx = 0, 0.0
    try:
        hidx = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hidx)
        if defects is not None:
            depth = defects[:, 0, 3] / 256.0
            keep = depth > 0.025 * eq_d
            tooth_cnt = int(keep.sum())
            tooth_idx = float(depth[keep].sum() / eq_d)
    except cv2.error:
        pass
    f.update(shape_tooth_count=tooth_cnt, shape_tooth_index=tooth_idx)

    # ---- 裂纹：黑帽运算提取细暗线 ----
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    inner = er > 0
    f["crack_index"] = float(((bh > 10) & inner).sum() / max(inner.sum(), 1))

    # ---- 光泽（高亮低饱和像素占比，润燥的粗代理） ----
    f["gloss_ratio"] = float(((hsv[..., 2] >= 225) & (hsv[..., 1] <= 70) & m).sum() / area)

    f = {k: float(v) for k, v in f.items()}
    return {**_nan_features(), **f, **qc}


def extract_from_path(path) -> dict:
    return extract_features(load_image(path))
