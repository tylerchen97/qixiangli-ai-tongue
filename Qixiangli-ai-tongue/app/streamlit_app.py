"""Streamlit 演示：上传舌象 -> 数字表型 + 辨证辅助打分（研究用途，非诊断）。
运行：streamlit run app/streamlit_app.py
需先执行 `python -m tcm_tongue demo --out outputs` 生成 outputs/reference_stats.json。
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from tcm_tongue.phenotyping import extract_features, segment_tongue
from tcm_tongue.syndrome import SYNDROME_CN, RuleBasedSyndrome, load_reference

st.set_page_config(page_title="AI 舌象数字表型", layout="wide")
st.title("AI 舌象数字表型 · 研究演示")
st.warning("研究用途：输出不构成诊断或治疗建议。")

up = st.file_uploader("上传舌象图片（正面、自然光、含灰卡更佳）", type=["png", "jpg", "jpeg"])
if up:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    mask = segment_tongue(img)
    feats = extract_features(img)
    overlay = img.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.25).astype(np.uint8)
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
    c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="舌体分割", use_container_width=True)
    if not feats["qc_ok"]:
        st.error("质控未通过：未能稳定分割舌体，请重拍。")
    st.subheader("数字表型")
    st.dataframe(pd.Series(feats, name="value").to_frame())
    ref_path = Path("outputs/reference_stats.json")
    if ref_path.exists() and feats["qc_ok"]:
        probs = RuleBasedSyndrome(load_reference(ref_path)).score(feats)
        st.subheader("辨证辅助打分（规则先验，须专家审订）")
        st.bar_chart(pd.Series({SYNDROME_CN[k]: v for k, v in probs.items()}))
    else:
        st.info("未找到 outputs/reference_stats.json，已跳过辨证打分。")
