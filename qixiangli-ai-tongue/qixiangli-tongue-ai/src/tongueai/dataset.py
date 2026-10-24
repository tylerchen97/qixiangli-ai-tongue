from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .labels import SINGLE_HEADS, MULTI_KEYS


class TongueDataset(Dataset):
    """CSV 列见 docs/data_spec.md。缺失标签记 -1,训练时自动忽略。图片应为 preprocess 之后的裁剪图。"""
    def __init__(self, df: pd.DataFrame, root: str, tfm):
        self.df, self.root, self.tfm = df.reset_index(drop=True), Path(root), tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tfm(Image.open(self.root / r["image_path"]).convert("RGB"))
        y = {}
        for h, cls in SINGLE_HEADS.items():
            v = r.get(h)
            y[h] = cls.index(v) if isinstance(v, str) and v in cls else -1
        m = []
        for k in MULTI_KEYS:
            v = r.get(k)
            m.append(float(v) if pd.notna(v) else -1.0)
        y["multi"] = torch.tensor(m)
        return x, y
