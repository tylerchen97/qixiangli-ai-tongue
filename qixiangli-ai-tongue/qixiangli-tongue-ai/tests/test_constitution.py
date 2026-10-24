from tongueai.constitution import score_constitutions
from tongueai.labels import SINGLE_HEADS, MULTI_KEYS


def onehot(**kw):
    single = {h: {c: (1.0 if kw.get(h) == c else 0.0) for c in cls} for h, cls in SINGLE_HEADS.items()}
    multi = {k: float(kw.get(k, 0)) for k in MULTI_KEYS}
    return {"single": single, "multi": multi}


def test_pinghe():
    r = score_constitutions(onehot(tongue_color="淡红", tongue_size="正常", coating_thickness="薄",
                                   coating_color="白", coating_moisture="润"))
    assert r["leading"][0] == "平和质"


def test_xueyu():
    assert "血瘀质" in score_constitutions(onehot(tongue_color="紫暗"))["leading"]


def test_shire():
    r = score_constitutions(onehot(tongue_color="红", coating_color="黄", coating_greasy="腻", coating_thickness="厚"))
    assert r["leading"][0] == "湿热质"


def test_tebing_not_assessable():
    r = score_constitutions(onehot())
    assert [s for s in r["scores"] if s["constitution"] == "特禀质"][0]["match"] is None
