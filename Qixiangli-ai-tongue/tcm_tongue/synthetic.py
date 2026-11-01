"""合成数据生成器：用于演示、单元测试与流水线冒烟测试（不含任何真实受试者数据）。

潜变量（证型倾向）同时驱动「舌象图像」与「客观指标」，并包含：
  * 多次随访的个体内波动（AR(1)）；
  * 部分受试者在某次访视后出现真实“事件”（湿热倾向上升），用于评估个体化监测；
  * 不同中心的相机色偏（模拟外部验证中的域偏移）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .syndrome import SYNDROMES


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def tongue_params(lat: dict, rng) -> dict:
    qi, yin, dh, pd_, st, yg = (lat[k] for k in SYNDROMES)
    red = 0.6 * yin + 0.5 * dh - 0.5 * qi - 0.5 * yg
    rgb = np.array([195 + 5 * red - 8 * st, 100 - 12 * red - 6 * st, 108 - 6 * red + 14 * st])
    return dict(
        body_rgb=np.clip(rgb, 0, 255),
        coat=_sig(0.9 * dh + 0.8 * pd_ - 0.6 * yin + rng.normal(0, 0.3)),
        yellow=_sig(1.2 * dh + rng.normal(0, 0.3)),
        tooth=_sig(0.9 * qi + 0.7 * pd_ + 0.5 * yg - 0.3 + rng.normal(0, 0.2)),
        crack=_sig(1.0 * yin - 0.3 * pd_ - 0.3 + rng.normal(0, 0.2)),
        gloss=_sig(0.6 * yg + 0.4 * dh + 0.3 * pd_ - 0.2 + rng.normal(0, 0.2)),
    )


def render_tongue(p: dict, size: int = 256, rng=None, cam_gain=(1.0, 1.0, 1.0)) -> np.ndarray:
    """按参数绘制一张合成舌象（返回 BGR uint8）。"""
    rng = rng or np.random.default_rng(0)
    img = np.zeros((size, size, 3), np.float32)
    img[:] = (60, 28, 38)
    lip = (150, 80, 85)
    cv2.rectangle(img, (0, 0), (size, int(size * .07)), lip, -1)
    cv2.rectangle(img, (0, int(size * .93)), (size, size), lip, -1)
    cx, cy = size / 2 + rng.normal(0, 3), size * 0.52 + rng.normal(0, 3)
    ax, ay = size * (0.30 + rng.normal(0, .01)), size * (0.36 + rng.normal(0, .01))
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 255, -1)

    # 齿痕：沿左右边缘“咬”出缺口
    n_tooth = int(round(2 + 6 * p["tooth"]))
    r = size * (0.012 + 0.022 * p["tooth"])
    for side in (-1, 1):
        for t in np.linspace(-0.9, 0.9, n_tooth):
            ang = (0 if side == 1 else np.pi) + t * 0.9
            bx, by = cx + ax * np.cos(ang), cy + ay * np.sin(ang)
            cv2.circle(mask, (int(bx + side * r * 0.5), int(by)), int(max(r, 1)), 0, -1)

    m = mask > 0
    yy, xx = np.mgrid[0:size, 0:size]
    shade = 1.04 - 0.14 * (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2).clip(0, 1.2)
    body = p["body_rgb"][None, None, :] * shade[..., None]
    img[m] = body[m]

    # 舌苔：中央区域覆盖，面积随 coat 增大
    coat_mask = np.zeros((size, size), np.float32)
    sc = 0.25 + 0.7 * p["coat"]
    cv2.ellipse(coat_mask, (int(cx), int(cy + size * .04)), (int(ax * .75 * sc + 2), int(ay * .8 * sc + 2)),
                0, 0, 360, 1.0, -1)
    coat_mask = cv2.GaussianBlur(coat_mask, (0, 0), 3) * 0.85 * m
    white, yellow = np.array([232, 228, 220.]), np.array([222, 200, 140.])
    coat_col = white * (1 - p["yellow"]) + yellow * p["yellow"]
    img = img * (1 - coat_mask[..., None]) + coat_col[None, None, :] * coat_mask[..., None]

    # 裂纹
    er = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0
    ys, xs = np.where(er)
    for _ in range(int(round(8 * p["crack"]))):
        i = rng.integers(len(ys))
        x0, y0 = int(xs[i]), int(ys[i])
        x1, y1 = int(x0 + rng.normal(0, 25)), int(y0 + rng.normal(0, 25))
        tmp = np.zeros((size, size), np.uint8)
        cv2.line(tmp, (x0, y0), (x1, y1), 255, 1)
        sel = (tmp > 0) & er
        img[sel] *= 0.55

    # 光泽高光
    for _ in range(int(round(6 * p["gloss"]))):
        i = rng.integers(len(ys))
        cv2.ellipse(img, (int(xs[i]), int(ys[i])), (int(rng.integers(3, 6)), int(rng.integers(2, 4))),
                    0, 0, 360, (245, 240, 238), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0) + rng.normal(0, 3, img.shape)
    img = img * np.array(cam_gain)[None, None, :]
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def generate_cohort(out_dir, n_subjects: int = 40, n_visits: int = 8, seed: int = 42,
                    sites=("A", "A", "B"), event_frac: float = 0.3, event_visit: int = 4,
                    image_size: int = 256) -> pd.DataFrame:
    """生成合成队列：图像写入 out_dir/images，返回元数据 DataFrame（含客观指标与证型标签）。"""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    site_gain = {"A": (1.0, 1.0, 1.0), "B": (0.96, 1.0, 1.05)}  # 站点 B：色偏
    rows = []
    for i in range(n_subjects):
        sid = f"S{i:03d}"
        site = sites[i % len(sites)]
        age = float(np.clip(rng.normal(50, 12), 25, 80))
        female = int(rng.random() < 0.55)
        base = {k: rng.normal(0, 1) for k in SYNDROMES}
        has_event = int(rng.random() < event_frac)
        label = max(SYNDROMES, key=lambda k: base[k] + rng.normal(0, 0.5))
        ar = {k: 0.0 for k in SYNDROMES}
        for v in range(n_visits):
            ar = {k: 0.6 * ar[k] + rng.normal(0, 0.35) for k in SYNDROMES}
            lat = {k: base[k] + ar[k] for k in SYNDROMES}
            if has_event and v >= event_visit:
                lat["damp_heat"] += 2.0
            qi, yin, dh, pd_, st, yg = (lat[k] for k in SYNDROMES)
            crp = float(np.exp(0.4 + 0.65 * dh + 0.15 * pd_ + rng.normal(0, 0.45)))
            hba1c = 5.45 + 0.22 * yin + 0.20 * pd_ + 0.012 * (age - 50) + rng.normal(0, 0.15)
            hb = 13.6 - 0.9 * qi - 0.5 * yg - 1.2 * female + rng.normal(0, 0.5)
            tg = float(np.exp(0.2 + 0.35 * pd_ + 0.1 * dh + rng.normal(0, 0.3)))
            alt = max(5.0, 24 + 6 * dh + rng.normal(0, 5))
            sbp = 122 + 4 * st + 0.3 * (age - 50) + rng.normal(0, 6)
            bmi = 24 + 1.5 * pd_ + rng.normal(0, 2)
            img = render_tongue(tongue_params(lat, rng), size=image_size, rng=rng, cam_gain=site_gain.get(site, (1, 1, 1)))
            fname = f"{sid}_v{v}.png"
            cv2.imwrite(str(out_dir / "images" / fname), img)
            rows.append(dict(subject_id=sid, site=site, visit=v, age=age, female=female, bmi=bmi,
                             crp=crp, hba1c=hba1c, hb=hb, tg=tg, alt=alt, sbp=sbp,
                             elevated_crp=int(crp > 3), high_hba1c=int(hba1c >= 5.7), anemia=int(hb < 12.5),
                             syndrome_label=label, event_subject=has_event,
                             event_visit=event_visit if has_event else -1, image_file=fname))
    return pd.DataFrame(rows)
