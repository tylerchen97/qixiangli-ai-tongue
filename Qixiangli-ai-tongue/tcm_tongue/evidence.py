"""模块 4：Evidence-based AI Modeling —— 基于独立验证的 AI 建模。

原则（参考 TRIPOD+AI 思路）：
  1. 开发集 / 外部独立验证集严格按中心（site）划分，外部集不参与任何拟合与调参；
  2. 同一受试者多次访视必须整体划入同一折（GroupKFold），避免信息泄漏；
  3. 报告区分度（AUC+CI）、校准（斜率/截距/Brier），以及舌象相对基础协变量的增量价值（ΔAUC）；
  4. 置信区间使用以受试者为单位的聚类 bootstrap。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(kind: str = "logreg") -> Pipeline:
    if kind == "gbm":
        return Pipeline([("m", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0))])
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", LogisticRegression(C=0.5, max_iter=2000))])


def _cluster_index(groups):
    groups = np.asarray(groups)
    return {g: np.where(groups == g)[0] for g in np.unique(groups)}


def _boot_samples(cmap, n_boot, seed):
    rng = np.random.default_rng(seed)
    ids = np.array(list(cmap.keys()))
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        yield np.concatenate([cmap[g] for g in pick])


def _calibration(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def binary_metrics(y, p, groups, n_boot: int = 500, seed: int = 0) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    boots = []
    for idx in _boot_samples(_cluster_index(groups), n_boot, seed):
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    slope, intercept = _calibration(y, p)
    return dict(auc=float(auc), auc_ci_low=float(np.percentile(boots, 2.5)),
                auc_ci_high=float(np.percentile(boots, 97.5)),
                brier=float(brier_score_loss(y, p)), calib_slope=slope, calib_intercept=intercept,
                n=int(len(y)), prevalence=float(y.mean()))


def delta_auc(y, p_base, p_full, groups, n_boot: int = 500, seed: int = 0) -> dict:
    y, p_base, p_full = map(np.asarray, (y, p_base, p_full))
    obs = roc_auc_score(y, p_full) - roc_auc_score(y, p_base)
    d = []
    for idx in _boot_samples(_cluster_index(groups), n_boot, seed):
        if len(np.unique(y[idx])) == 2:
            d.append(roc_auc_score(y[idx], p_full[idx]) - roc_auc_score(y[idx], p_base[idx]))
    return dict(delta_auc=float(obs), ci_low=float(np.percentile(d, 2.5)), ci_high=float(np.percentile(d, 97.5)))


def independent_validation(dev: pd.DataFrame, ext: pd.DataFrame, features, target: str,
                           covariates=(), subject_col: str = "subject_id", model: str = "logreg",
                           n_splits: int = 5, n_boot: int = 500, seed: int = 0) -> dict:
    """开发集内部交叉验证 + 外部独立验证。比较：仅协变量 / 仅舌象 / 协变量+舌象。"""
    sets = {"tongue_only": list(features), "covariates_plus_tongue": list(covariates) + list(features)}
    if covariates:
        sets = {"covariates_only": list(covariates), **sets}
    dev = dev.dropna(subset=[target]).reset_index(drop=True)
    ext = ext.dropna(subset=[target]).reset_index(drop=True)
    yd, ye = dev[target].astype(int).to_numpy(), ext[target].astype(int).to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(dev, yd, dev[subject_col]))

    internal, external, p_ext = {}, {}, {}
    for name, cols in sets.items():
        oof = cross_val_predict(build_model(model), dev[cols], yd, cv=splits, method="predict_proba")[:, 1]
        internal[name] = binary_metrics(yd, oof, dev[subject_col], n_boot, seed)
        fitted = build_model(model).fit(dev[cols], yd)  # 仅用开发集拟合
        p_ext[name] = fitted.predict_proba(ext[cols])[:, 1]
        external[name] = binary_metrics(ye, p_ext[name], ext[subject_col], n_boot, seed)
    result = dict(target=target, model=model, n_dev=len(dev), n_ext=len(ext),
                  internal_cv=internal, external=external)
    if covariates:
        result["external_delta_auc_vs_covariates"] = delta_auc(
            ye, p_ext["covariates_only"], p_ext["covariates_plus_tongue"], ext[subject_col], n_boot, seed)
    return result


def leave_one_site_out(df: pd.DataFrame, features, target: str, site_col: str, covariates=(),
                       subject_col: str = "subject_id", model: str = "logreg", n_boot: int = 200) -> pd.DataFrame:
    """每次留出一个中心作为独立外部验证集（至少 3 个中心时有意义）。"""
    rows = []
    for s in df[site_col].unique():
        r = independent_validation(df[df[site_col] != s], df[df[site_col] == s], features, target,
                                   covariates, subject_col, model, n_boot=n_boot)
        m = r["external"]["covariates_plus_tongue"]
        rows.append(dict(held_out_site=s, **{k: m[k] for k in ("auc", "auc_ci_low", "auc_ci_high", "brier", "calib_slope")}))
    return pd.DataFrame(rows)
