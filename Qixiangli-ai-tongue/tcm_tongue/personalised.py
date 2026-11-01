"""模块 3：Personalised Tongue Monitoring —— 个体化舌象动态监测。

以个体自身前 N 次随访作为基线（稳健中位数 / MAD），对后续访视计算 z 分数、EWMA 与双侧 CUSUM，
多个特征同时报警才触发个体级提示，以控制多重比较带来的误报。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonitorConfig:
    baseline_n: int = 4          # 基线访视数
    ewma_lambda: float = 0.3
    cusum_k: float = 1.0         # 允许漂移（单位：SD）
    cusum_h: float = 4.0         # 报警阈值（单位：SD）
    sd_floor_ratio: float = 1.0  # 个体 SD 下限 = ratio * 人群个体内 SD 中位数
    min_alarm_features: int = 3  # 个体级警报需同时报警的特征数


def within_subject_sd(df: pd.DataFrame, features, subject_col: str) -> dict:
    """各特征的个体内 SD（取各人 SD 的中位数，稳健）。"""
    return {f: float(df.groupby(subject_col)[f].std(ddof=1).median()) for f in features}


def monitor_subject(g: pd.DataFrame, features, time_col: str, cfg: MonitorConfig, pop_sd: dict) -> pd.DataFrame:
    g = g.sort_values(time_col).reset_index(drop=True)
    n = len(g)
    idx = np.arange(n)
    out = g[[time_col]].copy()
    n_alarm = np.zeros(n, int)
    absz = []
    for f in features:
        x = g[f].to_numpy(float)
        base = x[:cfg.baseline_n]
        base = base[~np.isnan(base)]
        if len(base) < 2:
            continue
        mu = np.median(base)
        mad = 1.4826 * np.median(np.abs(base - mu))
        sd = max(mad, cfg.sd_floor_ratio * pop_sd.get(f, 0.0), 1e-9)
        z = np.nan_to_num((x - mu) / sd)
        ew = np.zeros(n)
        sp, sn = np.zeros(n), np.zeros(n)
        prev_e, prev_p, prev_n = 0.0, 0.0, 0.0
        for t in range(n):
            prev_e = cfg.ewma_lambda * z[t] + (1 - cfg.ewma_lambda) * prev_e
            ew[t] = prev_e
            if t >= cfg.baseline_n:  # 基线期之后才累积
                prev_p = max(0.0, prev_p + z[t] - cfg.cusum_k)
                prev_n = max(0.0, prev_n - z[t] - cfg.cusum_k)
            sp[t], sn[t] = prev_p, prev_n
        out[f"z_{f}"] = z
        out[f"ewma_{f}"] = ew
        out[f"cusum_{f}"] = np.where(sp >= sn, sp, -sn)
        n_alarm += (((sp > cfg.cusum_h) | (sn > cfg.cusum_h)) & (idx >= cfg.baseline_n)).astype(int)
        absz.append(np.abs(z))
    out["n_alarm"] = n_alarm
    out["composite_z"] = np.mean(absz, axis=0) if absz else np.nan
    out["alert"] = (n_alarm >= cfg.min_alarm_features) & (idx >= cfg.baseline_n)
    return out


def monitor_cohort(df: pd.DataFrame, features, subject_col: str, time_col: str,
                   cfg: MonitorConfig | None = None) -> pd.DataFrame:
    cfg = cfg or MonitorConfig()
    pop_sd = within_subject_sd(df, features, subject_col)
    parts = []
    for sid, g in df.groupby(subject_col):
        r = monitor_subject(g, features, time_col, cfg, pop_sd)
        r.insert(0, subject_col, sid)
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def summarise_detection(mon: pd.DataFrame, meta: pd.DataFrame, subject_col: str, time_col: str,
                        event_flag: str = "event_subject", event_time: str = "event_visit") -> dict:
    """有真值事件（如合成数据 / 干预前瞻队列）时，评估检出率、误报率与延迟。"""
    m = mon.merge(meta[[subject_col, event_flag, event_time]].drop_duplicates(subject_col), on=subject_col)
    det, delays, fa = [], [], []
    for sid, g in m.groupby(subject_col):
        first_alert = g.loc[g["alert"], time_col]
        if g[event_flag].iloc[0] == 1:
            ev = g[event_time].iloc[0]
            hit = first_alert[first_alert >= ev]
            det.append(len(hit) > 0)
            if len(hit):
                delays.append(float(hit.iloc[0] - ev))
        else:
            fa.append(len(first_alert) > 0)
    return dict(n_event_subjects=len(det), detection_rate=float(np.mean(det)) if det else np.nan,
                median_delay_visits=float(np.median(delays)) if delays else np.nan,
                n_stable_subjects=len(fa), false_alert_rate=float(np.mean(fa)) if fa else np.nan)
