"""python -m tongueai.train --csv labels.csv --root processed/ --out runs/exp1"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
from .dataset import TongueDataset
from .labels import SINGLE_HEADS, SCHEMA_VERSION
from .model import TongueNet

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def split_by_patient(df, val=0.15, test=0.15, seed=42):
    ids = df["patient_id"].unique()
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids); nv, nt = int(n * val), int(n * test)
    g = lambda s: df[df["patient_id"].isin(s)]
    return g(ids[nv + nt:]), g(ids[:nv]), g(ids[nv:nv + nt])


def loss_fn(out, y):
    total = 0.0
    for h in SINGLE_HEADS:
        if (y[h] >= 0).any():
            total = total + F.cross_entropy(out[h], y[h], ignore_index=-1)
    mask = y["multi"] >= 0
    if mask.any():
        total = total + F.binary_cross_entropy_with_logits(out["multi"][mask], y["multi"][mask])
    return total


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    P = {h: [] for h in SINGLE_HEADS}; Y = {h: [] for h in SINGLE_HEADS}
    for x, y in dl:
        out = model(x.to(dev))
        for h in SINGLE_HEADS:
            P[h] += out[h].argmax(1).cpu().tolist(); Y[h] += y[h].tolist()
    res = {}
    for h in SINGLE_HEADS:
        keep = [i for i, v in enumerate(Y[h]) if v >= 0]
        if keep:
            res[h] = f1_score([Y[h][i] for i in keep], [P[h][i] for i in keep], average="macro", zero_division=0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="runs/exp"); ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 注意:舌色是核心诊断信息,禁止做 hue/saturation 抖动
    tr_t = T.Compose([T.RandomResizedCrop(a.size, scale=(0.8, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(brightness=0.15, contrast=0.15), T.ToTensor(), T.Normalize(MEAN, STD)])
    ev_t = T.Compose([T.Resize((a.size, a.size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr, va, te = split_by_patient(pd.read_csv(a.csv))
    mk = lambda d, t, s: DataLoader(TongueDataset(d, a.root, t), batch_size=a.bs, shuffle=s, num_workers=4)
    dtr, dva, dte = mk(tr, tr_t, True), mk(va, ev_t, False), mk(te, ev_t, False)
    model = TongueNet(a.backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    best = -1
    for ep in range(a.epochs):
        model.train()
        for x, y in dtr:
            y = {k: v.to(dev) for k, v in y.items()}
            loss = loss_fn(model(x.to(dev)), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        r = evaluate(model, dva, dev); score = float(np.mean(list(r.values()))) if r else 0
        print(f"epoch {ep} val macroF1={score:.4f}", {k: round(v, 3) for k, v in r.items()})
        if score > best:
            best = score
            torch.save({"state": model.state_dict(), "backbone": a.backbone, "size": a.size,
                        "schema": SCHEMA_VERSION}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt")["state"])
    print("TEST (patient-level split):", evaluate(model, dte, dev))


if __name__ == "__main__":
    main()
