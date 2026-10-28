import os, pytest, sys, json
sys.path.insert(0, ".")
pytestmark = pytest.mark.skipif(not os.path.exists("data/raw/diabetic_data.csv"), reason="raw data not downloaded")
import pandas as pd
from src.model import split, NUM, CAT
from src.prep import load


def test_no_patient_in_two_splits():
    tr, va, te = split(load())
    a, b, c = set(tr.patient_nbr), set(va.patient_nbr), set(te.patient_nbr)
    assert not (a & b) and not (a & c) and not (b & c)


def test_race_and_gender_are_not_features():
    assert not {"race", "gender"} & set(NUM + CAT)


def test_reported_results_are_consistent():
    r = json.load(open("eval/results_model.json"))
    m = r["models"]
    assert m["gradient_boosting"]["auc"] > m["rule_prior_inpatient"]["auc"]
    lo, hi = m["gradient_boosting"]["auc_ci"]
    assert lo < m["gradient_boosting"]["auc"] < hi
