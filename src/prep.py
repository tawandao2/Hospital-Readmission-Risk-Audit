"""Load and clean the UCI Diabetes 130-US hospitals data (1999 to 2008)."""
import numpy as np
import pandas as pd

RAW = "data/raw/diabetic_data.csv"
DEAD_OR_HOSPICE = {11, 13, 14, 19, 20, 21}   # cannot be readmitted
MEDS = ["metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "insulin"]


def icd_group(code):
    """Collapse ICD9 codes to broad clinical groups."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    c = str(code)
    if c.startswith(("V", "E")):
        return "other"
    n = float(c)
    if np.floor(n) == 250:
        return "diabetes"
    if 390 <= n <= 459 or n == 785:
        return "circulatory"
    if 460 <= n <= 519 or n == 786:
        return "respiratory"
    if 520 <= n <= 579 or n == 787:
        return "digestive"
    if 580 <= n <= 629 or n == 788:
        return "genitourinary"
    if 140 <= n <= 239:
        return "neoplasms"
    if 710 <= n <= 739:
        return "musculoskeletal"
    if 800 <= n <= 999:
        return "injury"
    return "other"


def load(path=RAW):
    d = pd.read_csv(path, na_values="?", low_memory=False)
    d = d[~d.discharge_disposition_id.isin(DEAD_OR_HOSPICE)]
    d = d[d.gender != "Unknown/Invalid"].copy()
    d["y"] = (d.readmitted == "<30").astype(int)
    d["age_mid"] = d.age.str.extract(r"\[(\d+)-").astype(float) + 5
    d["race"] = d.race.fillna("Unknown")
    d["dx1"] = d.diag_1.map(icd_group)
    d["dx2"] = d.diag_2.map(icd_group)
    d["dx3"] = d.diag_3.map(icd_group)
    d["prior_visits"] = d.number_outpatient + d.number_emergency + d.number_inpatient
    d["a1c"] = d.A1Cresult.fillna("not_measured")
    d["n_meds_changed"] = sum((d[m].isin(["Up", "Down"])).astype(int) for m in MEDS)
    return d.reset_index(drop=True)
