"""中医辨证辅助模块（研究用途）。

提供两条互补路径：
  A. RuleBasedSyndrome：基于专家先验的可解释打分（舌象特征相对人群参考的 z 分数加权）；
  B. SyndromeClassifier：有专家标注时的监督学习，需在独立中心验证后再解读。

⚠️ 输出仅为研究假设与辅助参考，不构成诊断或治疗建议；规则权重为示例先验，须由临床专家审订。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SYNDROMES = ["qi_deficiency", "yin_deficiency", "damp_heat", "phlegm_dampness", "blood_stasis", "yang_deficiency"]
SYNDROME_CN = {"qi_deficiency": "气虚证", "yin_deficiency": "阴虚证", "damp_heat": "湿热证",
               "phlegm_dampness": "痰湿证", "blood_stasis": "血瘀证", "yang_deficiency": "阳虚证"}

# 特征 -> 权重（正：该特征越高越支持该证型）。示例先验，须由临床专家审订。
RULES: dict[str, dict[str, float]] = {
    "qi_deficiency":   {"color_a": -0.6, "shape_tooth_index": 0.8, "color_L": 0.2},
    "yin_deficiency":  {"color_a": 0.8, "coat_ratio": -0.6, "crack_index": 0.9, "gloss_ratio": -0.4},
    "damp_heat":       {"color_a": 0.5, "coat_ratio": 0.7, "coat_b": 0.9, "gloss_ratio": 0.3},
    "phlegm_dampness": {"coat_ratio": 0.7, "shape_tooth_index": 0.6, "shape_area_ratio": 0.4, "color_a": -0.2},
    "blood_stasis":    {"color_L": -0.7, "color_b": -0.6, "color_a": 0.2},
    "yang_deficiency": {"color_a": -0.7, "gloss_ratio": 0.6, "color_L": 0.4, "shape_tooth_index": 0.3},
}


def reference_stats(df: pd.DataFrame, features) -> dict:
    return {f: {"mean": float(df[f].mean()), "sd": float(df[f].std(ddof=0) or 1.0)} for f in features}


def save_reference(stats: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)


def load_reference(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class RuleBasedSyndrome:
    def __init__(self, reference: dict, rules: dict | None = None, temperature: float = 1.0):
        self.ref, self.rules, self.T = reference, rules or RULES, temperature

    def _z(self, row: dict | pd.Series, f: str) -> float:
        v = row.get(f, np.nan)
        if v is None or np.isnan(v) or f not in self.ref:
            return 0.0
        return (v - self.ref[f]["mean"]) / self.ref[f]["sd"]

    def score(self, row) -> dict:
        raw = {s: sum(w * self._z(row, f) for f, w in r.items()) for s, r in self.rules.items()}
        v = np.array(list(raw.values())) / self.T
        p = np.exp(v - v.max())
        p /= p.sum()
        return dict(zip(raw.keys(), p))

    def explain(self, row, top: int = 3) -> list[tuple[str, float, list[tuple[str, float]]]]:
        probs = self.score(row)
        out = []
        for s in sorted(probs, key=probs.get, reverse=True)[:top]:
            contrib = sorted(((f, w * self._z(row, f)) for f, w in self.rules[s].items()),
                             key=lambda t: abs(t[1]), reverse=True)
            out.append((s, probs[s], contrib))
        return out

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        P = pd.DataFrame([self.score(r) for _, r in df.iterrows()], index=df.index)
        P["pred"] = P.idxmax(axis=1)
        return P


class SyndromeClassifier:
    """有监督辨证分类器（多分类逻辑回归）。"""

    def __init__(self, features):
        self.features = list(features)
        self.pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                              ("m", LogisticRegression(C=0.5, max_iter=3000))])

    def fit(self, df: pd.DataFrame, label_col: str):
        self.pipe.fit(df[self.features], df[label_col])
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.pipe.predict_proba(df[self.features]),
                            columns=self.pipe.classes_, index=df.index)

    def evaluate(self, df: pd.DataFrame, label_col: str) -> dict:
        P = self.predict_proba(df)
        pred = P.idxmax(axis=1)
        y = df[label_col]
        labels = sorted(set(self.pipe.classes_) | set(y))  # 外部集可能含开发集未见的证型
        return dict(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                    classes=labels, confusion=confusion_matrix(y, pred, labels=labels).tolist(),
                    classes_unseen_in_training=sorted(set(y) - set(self.pipe.classes_)))
