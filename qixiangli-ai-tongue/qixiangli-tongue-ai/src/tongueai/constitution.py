"""九种体质映射(规则引擎,依赖零)。

重要:舌象只是体质辨识的辅助依据。《中医体质分类与判定》(ZYYXH/T157-2009)以量表为主,
本模块输出的是"舌象与各体质典型表现的匹配度",不是体质判定结果。
下列权重为依据教材经验设定的【先验】,必须经中医专家评审并用带体质标签的数据校准后才能用于研究结论。
"""
CONSTITUTIONS = ["平和质", "气虚质", "阳虚质", "阴虚质", "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"]

# (特征头, 取值) -> 权重;多标签头用 True/False
RULES = {
    "平和质": {("tongue_color", "淡红"): 1.0, ("tongue_size", "正常"): 0.8, ("coating_thickness", "薄"): 0.6,
              ("coating_color", "白"): 0.4, ("coating_moisture", "润"): 0.5,
              ("teeth_marks", False): 0.5, ("cracks", False): 0.4},
    "气虚质": {("tongue_color", "淡白"): 0.6, ("tongue_color", "淡红"): 0.2, ("tongue_size", "胖大"): 0.6,
              ("teeth_marks", True): 0.7, ("coating_thickness", "薄"): 0.3, ("coating_color", "白"): 0.3},
    "阳虚质": {("tongue_color", "淡白"): 0.8, ("tongue_size", "胖大"): 0.6, ("coating_moisture", "滑"): 0.8,
              ("coating_moisture", "润"): 0.2, ("coating_color", "白"): 0.4, ("teeth_marks", True): 0.4},
    "阴虚质": {("tongue_color", "红"): 0.6, ("tongue_color", "绛"): 0.5, ("tongue_size", "瘦薄"): 0.7,
              ("coating_thickness", "少/无"): 0.9, ("coating_moisture", "燥"): 0.7, ("cracks", True): 0.7},
    "痰湿质": {("tongue_size", "胖大"): 0.7, ("coating_thickness", "厚"): 0.6, ("coating_greasy", "腻"): 0.9,
              ("coating_color", "白"): 0.4, ("teeth_marks", True): 0.5, ("coating_moisture", "滑"): 0.3},
    "湿热质": {("tongue_color", "红"): 0.5, ("coating_color", "黄"): 0.9, ("coating_greasy", "腻"): 0.8,
              ("coating_thickness", "厚"): 0.4, ("prickles", True): 0.3},
    "血瘀质": {("tongue_color", "紫暗"): 1.0, ("tongue_color", "绛"): 0.3},
    "气郁质": {("tongue_color", "淡红"): 0.3, ("coating_thickness", "薄"): 0.2, ("coating_color", "白"): 0.2},
    "特禀质": {},
}
RELIABILITY = {c: "tongue_supported" for c in CONSTITUTIONS}
RELIABILITY["气郁质"] = "low"            # 舌象特异性弱,主要依赖问诊/量表
RELIABILITY["特禀质"] = "not_assessable"  # 舌象无法评估


def _p(feats: dict, head: str, value) -> float:
    if isinstance(value, bool):
        p = feats["multi"].get(head, 0.0)
        return p if value else 1.0 - p
    return feats["single"].get(head, {}).get(value, 0.0)


def score_constitutions(feats: dict) -> dict:
    """feats = {"single": {head: {label: prob}}, "multi": {key: prob}} -> 九体质匹配度(0-1)。"""
    out = []
    for c in CONSTITUTIONS:
        rules = RULES[c]
        if not rules:
            out.append({"constitution": c, "match": None, "reliability": RELIABILITY[c]})
            continue
        s = sum(_p(feats, h, v) * w for (h, v), w in rules.items())
        out.append({"constitution": c, "match": round(s / sum(rules.values()), 4),
                    "reliability": RELIABILITY[c]})
    ranked = sorted([o for o in out if o["match"] is not None], key=lambda o: -o["match"])
    top = ranked[0]["match"]
    leading = [o["constitution"] for o in ranked
               if o["match"] >= 0.45 and top - o["match"] <= 0.12 and o["reliability"] != "low"]
    return {"scores": out, "leading": leading,
            "note": "仅为舌象匹配度,不构成体质判定;请结合体质量表与医师问诊。"}
