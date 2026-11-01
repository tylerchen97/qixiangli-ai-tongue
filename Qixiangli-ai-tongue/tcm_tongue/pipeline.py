"""端到端流水线：图像 -> 特征 -> 关联 -> 个体化监测 -> 独立验证 -> 辨证辅助。"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .association import adjusted_association, cca_association, longitudinal_mixed_model
from .evidence import independent_validation, leave_one_site_out
from .personalised import MonitorConfig, monitor_cohort, summarise_detection
from .phenotyping import FEATURE_PREFIXES, extract_from_path
from .syndrome import RuleBasedSyndrome, SyndromeClassifier, reference_stats, save_reference

DEFAULT_CFG = dict(
    subject_col="subject_id", time_col="visit", site_col="site",
    dev_sites=["A"], ext_sites=["B"],
    biomarkers=["crp", "hba1c", "hb", "tg", "alt", "sbp"],
    log_transform=["crp", "tg"],
    covariates=["age", "female", "bmi"],
    binary_targets=["elevated_crp", "high_hba1c", "anemia"],
    syndrome_col="syndrome_label",
    model="logreg", n_boot=300,
    monitor=dict(baseline_n=4, cusum_k=1.0, cusum_h=4.0, sd_floor_ratio=1.0, min_alarm_features=3),
)


def _py(o):
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def extract_folder(image_dir, files=None) -> pd.DataFrame:
    image_dir = Path(image_dir)
    files = files or sorted(p.name for p in image_dir.iterdir()
                            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    rows = []
    for fn in files:
        try:
            rows.append({"image_file": fn, **extract_from_path(image_dir / fn)})
        except Exception as e:  # noqa: BLE001
            rows.append({"image_file": fn, "qc_ok": 0, "error": str(e)})
    return pd.DataFrame(rows)


def run_analysis(df: pd.DataFrame, cfg: dict, outdir) -> dict:
    cfg = {**DEFAULT_CFG, **cfg}
    out = Path(outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    sid, tcol, scol = cfg["subject_col"], cfg["time_col"], cfg["site_col"]
    df = df.copy()
    if "qc_ok" in df:
        df = df[df["qc_ok"] == 1]
    feats = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    bios = []
    for b in cfg["biomarkers"]:
        if b in cfg["log_transform"]:
            df[f"log_{b}"] = np.log(df[b])
            bios.append(f"log_{b}")
        else:
            bios.append(b)
    summary: dict = {"n_images": int(len(df)), "n_subjects": int(df[sid].nunique()), "n_features": len(feats)}

    # ---- 1/2 关联分析 ----
    assoc = adjusted_association(df, feats, bios, cfg["covariates"], cluster_col=sid)
    assoc.to_csv(out / "association_results.csv", index=False)
    mixed = []
    for _, r in assoc.head(6).iterrows():
        try:
            mixed.append(longitudinal_mixed_model(df, r["feature"], r["outcome"], sid, cfg["covariates"]))
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"mixed model failed: {e}")
    pd.DataFrame(mixed).to_csv(out / "longitudinal_mixed.csv", index=False)
    sm_ = df.groupby(sid)[feats + bios].mean()
    summary["cca_subject_level"] = cca_association(sm_[feats], sm_[bios], n_perm=200)
    summary["n_fdr_significant_pairs"] = int((assoc["q_fdr"] < 0.05).sum()) if len(assoc) else 0

    # ---- 3 个体化监测 ----
    mcfg = MonitorConfig(**cfg["monitor"])
    mon = monitor_cohort(df, feats, sid, tcol, mcfg)
    mon.to_csv(out / "monitoring_timeseries.csv", index=False)
    if "event_subject" in df:
        summary["monitoring"] = summarise_detection(mon, df, sid, tcol)

    # ---- 4 独立验证 ----
    dev = df[df[scol].isin(cfg["dev_sites"])]
    ext = df[df[scol].isin(cfg["ext_sites"])]
    validation = {}
    for t in cfg["binary_targets"]:
        if t in df:
            validation[t] = independent_validation(dev, ext, feats, t, cfg["covariates"], sid,
                                                   cfg["model"], n_boot=cfg["n_boot"])
    summary["validation"] = validation
    if df[scol].nunique() >= 3:
        for t in cfg["binary_targets"]:
            leave_one_site_out(df, feats, t, scol, cfg["covariates"], sid).to_csv(
                out / f"leave_one_site_out_{t}.csv", index=False)
    (out / "validation.json").write_text(json.dumps(_py(validation), ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 辨证辅助 ----
    if cfg.get("syndrome_col") in df:
        lab = cfg["syndrome_col"]
        ref = reference_stats(dev, feats)
        save_reference(ref, out / "reference_stats.json")
        rb = RuleBasedSyndrome(ref).predict_df(ext)
        clf = SyndromeClassifier(feats).fit(dev, lab)
        joblib.dump(clf, out / "syndrome_model.joblib")
        summary["syndrome"] = dict(
            rule_based_external_accuracy=float((rb["pred"] == ext[lab]).mean()),
            supervised_external=clf.evaluate(ext, lab),
            chance_level=float(1 / df[lab].nunique()),
        )
        rb.assign(true=ext[lab].values, **{sid: ext[sid].values}).to_csv(out / "syndrome_rule_based.csv", index=False)

    _make_figures(assoc, mon, df, feats, sid, tcol, out / "figures")
    (out / "summary.json").write_text(json.dumps(_py(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out / "analysis_table.csv", index=False)
    return summary


def _make_figures(assoc, mon, df, feats, sid, tcol, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(assoc):
        B = assoc.pivot(index="feature", columns="outcome", values="beta")
        Q = assoc.pivot(index="feature", columns="outcome", values="q_fdr")
        fig, ax = plt.subplots(figsize=(1.1 * len(B.columns) + 3, 0.32 * len(B) + 2))
        im = ax.imshow(B.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(B.columns)), B.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(B.index)), B.index)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if Q.values[i, j] < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=12)
        fig.colorbar(im, label="standardised beta (* FDR<0.05)")
        ax.set_title("Tongue features vs objective biomarkers")
        fig.tight_layout()
        fig.savefig(figdir / "association_heatmap.png", dpi=150)
        plt.close(fig)

    if "event_subject" in df and (df["event_subject"] == 1).any():
        s = df.loc[df["event_subject"] == 1, sid].iloc[0]
        g = mon[mon[sid] == s]
        fz = [c for c in g.columns if c.startswith("z_")]
        top = g[fz].iloc[-1].abs().sort_values(ascending=False).index[:4]
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        for c in top:
            axes[0].plot(g[tcol], g[c], marker="o", label=c[2:])
        ev = df.loc[df[sid] == s, "event_visit"].iloc[0]
        for ax in axes:
            ax.axvline(ev, ls="--", c="gray")
        axes[0].set_ylabel("z vs personal baseline")
        axes[0].legend(fontsize=7)
        axes[1].plot(g[tcol], g["n_alarm"], marker="s", c="crimson")
        axes[1].set_ylabel("# CUSUM alarms")
        axes[1].set_xlabel("visit")
        axes[0].set_title(f"Personalised monitoring — {s} (dashed: true event)")
        fig.tight_layout()
        fig.savefig(figdir / "personal_trajectory.png", dpi=150)
        plt.close(fig)


def run_demo(out, n_subjects: int = 60, n_visits: int = 8, seed: int = 42) -> dict:
    from .synthetic import generate_cohort
    out = Path(out)
    syn = out / "synthetic"
    meta = generate_cohort(syn, n_subjects, n_visits, seed)
    meta.to_csv(syn / "metadata.csv", index=False)
    feats = extract_folder(syn / "images", meta["image_file"].tolist())
    feats.to_csv(out / "features.csv", index=False)
    df = meta.merge(feats, on="image_file")
    return run_analysis(df, DEFAULT_CFG, out)
