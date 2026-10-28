"""Build the findings for one discharge. Every number the note may use is written here, already formatted."""
import json
import pandas as pd

REF = json.load(open("data/reference_rates.json"))
MODEL = json.load(open("eval/results_model.json"))


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def bucket(n):
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3 or more"


def findings_for(row):
    r = row
    base = MODEL["base_rate"]
    top = (1 - r["risk_pct_rank"]) * 100
    m = MODEL["models"]["gradient_boosting"]
    f = [dict(id="risk", kind="risk", must_cover=True, title="Estimated risk of readmission within 30 days",
              facts=[f"The model estimates a {pct(r['risk'])} chance of readmission within 30 days",
                     f"The average in the held out test group is {pct(base)}",
                     (f"This discharge ranks in the top {max(top, 1):.0f}% by estimated risk" if top <= 50
                       else f"This discharge ranks in the lowest {max(100 - top, 1):.0f}% by estimated risk")])]
    b = bucket(int(r["number_inpatient"]))
    ref = REF["by_prior_inpatient"][b]
    f.append(dict(id="prior_use", kind="history", must_cover=False, title="Hospital use in the year before this stay",
                  facts=[f"{int(r['number_inpatient'])} inpatient stays, {int(r['number_emergency'])} emergency visits and {int(r['number_outpatient'])} outpatient visits",
                         f"Among training patients in the group with {b} prior inpatient stays, {pct(ref['rate'])} were readmitted within 30 days ({ref['n']:,} stays)"]))
    dref = REF["by_disposition"].get(str(int(r["discharge_disposition_id"])))
    if dref and dref["n"] >= 200:
        f.append(dict(id="discharge", kind="history", must_cover=False, title="Where the patient went after discharge",
                      facts=[f"Discharge destination: {REF['labels'].get(str(int(r['discharge_disposition_id'])), 'not described')}",
                             f"Among training patients with the same destination, {pct(dref['rate'])} were readmitted within 30 days ({dref['n']:,} stays)"]))
    xref = REF["by_dx1"].get(r["dx1"])
    if xref and xref["n"] >= 200:
        f.append(dict(id="diagnosis", kind="history", must_cover=False, title="Main diagnosis group",
                      facts=[f"Main diagnosis group: {r['dx1']}",
                             f"Among training patients in this group, {pct(xref['rate'])} were readmitted within 30 days ({xref['n']:,} stays)"]))
    f.append(dict(id="stay", kind="history", must_cover=False, title="This stay",
                  facts=[f"{int(r['time_in_hospital'])} days in hospital (training median {REF['median_los']:.0f})",
                         f"{int(r['num_medications'])} medications (training median {REF['median_meds']:.0f})",
                         f"{int(r['number_diagnoses'])} diagnoses recorded (training median {REF['median_diagnoses']:.0f})"]))
    f.append(dict(id="limits", kind="caveat", must_cover=True, title="Limits of this estimate",
                  facts=[f"The model ranks patients only modestly: AUC {m['auc']:.2f} (95% interval {m['auc_ci'][0]:.3f} to {m['auc_ci'][1]:.3f})",
                         "The training data are hospital stays from 1999 to 2008 at 130 US hospitals",
                         "This is a ranking aid for care teams, not a diagnosis, and it does not replace clinical judgment"]))
    return {"discharge": str(int(r["encounter_id"])), "findings": f}
