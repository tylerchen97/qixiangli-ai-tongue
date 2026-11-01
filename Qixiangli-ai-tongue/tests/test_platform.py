import numpy as np
import pandas as pd

from tcm_tongue.evidence import independent_validation
from tcm_tongue.personalised import MonitorConfig, monitor_cohort
from tcm_tongue.phenotyping import FEATURE_NAMES, extract_features
from tcm_tongue.syndrome import RULES, SYNDROMES, RuleBasedSyndrome
from tcm_tongue.synthetic import render_tongue


def _params(**kw):
    p = dict(body_rgb=np.array([195, 100, 108.]), coat=0.2, yellow=0.2, tooth=0.2, crack=0.2, gloss=0.2)
    p.update(kw)
    return p


def test_feature_extraction_keys_and_qc():
    f = extract_features(render_tongue(_params(), rng=np.random.default_rng(0)))
    assert f["qc_ok"] == 1 and 0.1 < f["qc_mask_frac"] < 0.8
    assert all(k in f for k in FEATURE_NAMES)


def test_coating_and_redness_are_monotonic():
    lo = extract_features(render_tongue(_params(coat=0.05), rng=np.random.default_rng(1)))
    hi = extract_features(render_tongue(_params(coat=0.95), rng=np.random.default_rng(1)))
    assert hi["coat_ratio"] > lo["coat_ratio"]
    pale = extract_features(render_tongue(_params(body_rgb=np.array([200, 140, 145.])), rng=np.random.default_rng(2)))
    red = extract_features(render_tongue(_params(body_rgb=np.array([190, 60, 70.])), rng=np.random.default_rng(2)))
    assert red["color_a"] > pale["color_a"]


def test_monitor_detects_shift():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(10):
        for v in range(10):
            shift = 4.0 if (s < 5 and v >= 5) else 0.0
            rows.append(dict(sid=s, v=v, **{f"f{i}": rng.normal(0, 1) + shift for i in range(6)}))
    df = pd.DataFrame(rows)
    mon = monitor_cohort(df, [f"f{i}" for i in range(6)], "sid", "v", MonitorConfig(baseline_n=4))
    flagged = mon.groupby("sid")["alert"].any()
    assert flagged[:5].all() and not flagged[5:].any()


def test_independent_validation_runs():
    rng = np.random.default_rng(0)
    n = 160
    df = pd.DataFrame(dict(subject_id=np.repeat(np.arange(n // 2), 2), x=rng.normal(size=n), age=rng.normal(50, 10, n)))
    df["y"] = (df["x"] + rng.normal(0, 0.5, n) > 0).astype(int)
    r = independent_validation(df.iloc[:100], df.iloc[100:], ["x"], "y", ["age"], n_boot=50)
    assert r["external"]["covariates_plus_tongue"]["auc"] > 0.8


def test_rule_based_syndrome_probabilities():
    ref = {f: {"mean": 0.0, "sd": 1.0} for r in RULES.values() for f in r}
    p = RuleBasedSyndrome(ref).score({"color_a": 2.0, "coat_ratio": -1.0, "crack_index": 2.0})
    assert abs(sum(p.values()) - 1) < 1e-9 and max(p, key=p.get) == "yin_deficiency"
    assert set(p) == set(SYNDROMES)
