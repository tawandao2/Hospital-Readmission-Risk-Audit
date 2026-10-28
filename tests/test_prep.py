import os, pytest, sys
sys.path.insert(0, ".")
pytestmark = pytest.mark.skipif(not os.path.exists("data/raw/diabetic_data.csv"), reason="raw data not downloaded")
import numpy as np
from src.prep import icd_group, load


def test_icd_groups():
    assert icd_group("250.83") == "diabetes" and icd_group("428") == "circulatory"
    assert icd_group("486") == "respiratory" and icd_group("V57") == "other" and icd_group(np.nan) == "missing"


def test_load_drops_dead_and_hospice_and_labels_outcome():
    d = load()
    assert not d.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21]).any()
    assert set(d.y.unique()) == {0, 1} and 0.10 < d.y.mean() < 0.13
