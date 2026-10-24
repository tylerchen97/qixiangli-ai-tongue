"""入库分诊:扫描一批原始照片,生成 manifest.csv(质检+分诊+待标注列)和人工复核用的总览图。
python scripts/make_manifest.py --src raw_images/ --out manifest.csv --sheet review.jpg
原图含人脸与健康信息,仅在受控环境处理;manifest 只含文件名与指标。"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tongueai.preprocess import quality_check, triage

COLS = ["view", "patient_id", "tongue_color", "tongue_size", "coating_color", "coating_thickness",
        "coating_moisture", "coating_greasy", "teeth_marks", "cracks", "prickles", "use_for_training", "annotator", "note"]

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", default="manifest.csv")
ap.add_argument("--sheet", default=None)
a = ap.parse_args()
rows, tiles = [], []
for i, p in enumerate(sorted(x for x in Path(a.src).rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})):
    im = cv2.imread(str(p))
    if im is None:
        rows.append({"id": i, "file": p.name, "triage_flags": "unreadable"}); continue
    q, t = quality_check(im), triage(im)
    rows.append({"id": i, "file": p.name, **{k: v for k, v in q.items() if k != "pass"},
                 "issues": ";".join(q["issues"]), "width": t["width"], "height": t["height"],
                 "triage_flags": ";".join(t["triage_flags"]), **{c: "" for c in COLS}})
    tile = cv2.resize(im, (200, 260)); cv2.putText(tile, str(i), (6, 28), 0, 0.9, (0, 255, 255), 2); tiles.append(tile)
pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig")
if a.sheet and tiles:
    while len(tiles) % 8: tiles.append(np.zeros((260, 200, 3), np.uint8))
    cv2.imwrite(a.sheet, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
print(f"{len(rows)} images -> {a.out}")
