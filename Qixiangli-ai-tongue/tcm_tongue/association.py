"""模块 2：Multimodal Biomarker Association —— 舌象特征与客观指标的统计关联。"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def _design(d: pd.DataFrame, x: str, covariates) -> pd.DataFrame:
    X = pd.DataFrame({x: (d[x] - d[x].mean()) / d[x].std(ddof=0)})
    if covariates:
        C = pd.get_dummies(d[list(covariates)], drop_first=True, dtype=float)
        X = pd.concat([X, C], axis=1)
    return sm.add_constant(X.astype(float), has_constant="add")


def adjusted_association(df: pd.DataFrame, features, outcomes, covariates=(),
                         cluster_col: str | None = None, fdr_alpha: float = 0.05) -> pd.DataFrame:
    """逐对 OLS：z(outcome) ~ z(feature) + covariates，聚类稳健标准误（重复测量），BH-FDR 校正。
    beta 为标准化系数（特征每增 1 SD，指标变化多少 SD）。"""
    rows = []
    for y in outcomes:
        for x in features:
            cols = [x, y, *covariates] + ([cluster_col] if cluster_col else [])
            d = df[cols].dropna()
            if len(d) < 20 or d[x].std() == 0 or d[y].std() == 0:
                continue
            yz = (d[y] - d[y].mean()) / d[y].std(ddof=0)
            X = _design(d, x, covariates)
            model = sm.OLS(yz, X)
            if cluster_col:
                groups = pd.factorize(d[cluster_col])[0]
                res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            else:
                res = model.fit(cov_type="HC3")
            lo, hi = res.conf_int().loc[x]
            rows.append(dict(outcome=y, feature=x, n=len(d), beta=res.params[x],
                             ci_low=lo, ci_high=hi, p=res.pvalues[x]))
    out = pd.DataFrame(rows)
    if len(out):
        out["q_fdr"] = multipletests(out["p"], method="fdr_bh", alpha=fdr_alpha)[1]
        out = out.sort_values("p").reset_index(drop=True)
    return out


def longitudinal_mixed_model(df: pd.DataFrame, feature: str, outcome: str, subject_col: str,
                             covariates=()) -> dict:
    """个体内 / 个体间分解的线性混合模型。
    within：同一个人舌象偏离自身均值时，指标如何同向变化（动态关联）；
    between：不同个体平均舌象差异与指标的关联（横断面关联）。"""
    d = df[[feature, outcome, subject_col, *covariates]].dropna().copy()
    d["_x_between"] = d.groupby(subject_col)[feature].transform("mean")
    d["_x_within"] = d[feature] - d["_x_between"]
    d["_y"] = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=0)
    sd = d["_x_between"].std(ddof=0) or 1.0
    d["_x_between"] = (d["_x_between"] - d["_x_between"].mean()) / sd
    d["_x_within"] = d["_x_within"] / (d["_x_within"].std(ddof=0) or 1.0)
    d = d.rename(columns={subject_col: "_sid"})
    cov = "".join(f" + {c}" for c in covariates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm(f"_y ~ _x_between + _x_within{cov}", d, groups=d["_sid"]).fit(reml=True)
    return dict(feature=feature, outcome=outcome, n_obs=len(d), n_subjects=d["_sid"].nunique(),
                beta_between=res.params["_x_between"], p_between=res.pvalues["_x_between"],
                beta_within=res.params["_x_within"], p_within=res.pvalues["_x_within"])


def cca_association(X: pd.DataFrame, Y: pd.DataFrame, n_perm: int = 200, seed: int = 0) -> dict:
    """多变量关联：第一对典型相关系数 + 置换检验 p 值（建议传入个体均值，避免重复测量）。"""
    d = pd.concat([X, Y], axis=1).dropna()
    Xs = StandardScaler().fit_transform(d[X.columns])
    Ys = StandardScaler().fit_transform(d[Y.columns])

    def first_r(Xa, Ya):
        cca = CCA(n_components=1, max_iter=1000).fit(Xa, Ya)
        u, v = cca.transform(Xa, Ya)
        return abs(np.corrcoef(u[:, 0], v[:, 0])[0, 1])

    obs = first_r(Xs, Ys)
    rng = np.random.default_rng(seed)
    null = np.array([first_r(Xs, Ys[rng.permutation(len(Ys))]) for _ in range(n_perm)])
    return dict(canonical_r=float(obs), perm_p=float((1 + (null >= obs).sum()) / (1 + n_perm)),
                n=int(len(d)), n_perm=n_perm)
