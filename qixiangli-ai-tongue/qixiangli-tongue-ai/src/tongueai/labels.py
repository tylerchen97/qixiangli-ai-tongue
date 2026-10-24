"""舌象标签体系 v1(单标签多头 + 多标签头)。修改标签需同步升级 SCHEMA_VERSION。"""
SCHEMA_VERSION = "tongue-schema/1.0"

SINGLE_HEADS = {
    "tongue_color": ["淡白", "淡红", "红", "绛", "紫暗"],      # 舌色
    "tongue_size": ["胖大", "正常", "瘦薄"],                   # 舌形(胖瘦)
    "coating_color": ["无苔", "白", "黄", "灰黑"],             # 苔色
    "coating_thickness": ["少/无", "薄", "厚"],                # 苔厚薄
    "coating_moisture": ["润", "滑", "燥"],                    # 苔润燥
    "coating_greasy": ["无", "腻", "腐"],                      # 腻腐
}
MULTI_KEYS = ["teeth_marks", "cracks", "prickles"]
MULTI_ZH = {"teeth_marks": "齿痕", "cracks": "裂纹", "prickles": "点刺"}

HEAD_ZH = {
    "tongue_color": "舌色", "tongue_size": "舌形", "coating_color": "苔色",
    "coating_thickness": "苔厚薄", "coating_moisture": "苔润燥", "coating_greasy": "苔腻腐",
}
