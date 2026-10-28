import os, pytest, sys
sys.path.insert(0, ".")
pytestmark = pytest.mark.skipif(not os.path.exists("data/raw/hrrp_fy2026.csv"), reason="raw data not downloaded")
from src.cms import load, summarize


def test_ratio_matches_predicted_over_expected():
    s = summarize(load())
    assert s["max_ratio_mismatch"] < 1e-3


def test_suppressed_values_are_not_treated_as_zero():
    h = load()
    assert h["Number of Readmissions"].isna().sum() > 0 and (h["Excess Readmission Ratio"].dropna() > 0).all()
