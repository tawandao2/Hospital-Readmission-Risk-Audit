import json
import pandas as pd
import streamlit as st

from src.checker import RulesOnlyWriter, run_note
from src.note import findings_for
from src import pbi_view

st.set_page_config(page_title="Readmission Risk Desk", page_icon="🏥", layout="wide")


@st.cache_data
def load():
    return (pd.read_csv("data/scored_test.csv"), json.load(open("eval/results_model.json")),
            json.load(open("eval/results_cms.json")), json.load(open("eval/results_notes.json")))


d, M, C, N = load()
m = M["models"]
g = m["gradient_boosting"]

st.title("Readmission Risk Desk")
st.caption("Which discharged patients are most likely to come back within 30 days, how fair the ranking is, "
           "and how hospitals compare today. Model data: 99,340 US hospital stays from 1999 to 2008. Hospital data: CMS FY2026.")

t0, t1, t2, t3, t4, t5 = st.tabs(["Report", "Overview", "Discharge list", "Fairness audit", "Hospitals today", "Limits"])

with t0:
    pbi_view.render(d, M, C)


with t1:
    a, b, c, e = st.columns(4)
    a.metric("Model AUC", f"{g['auc']:.3f}", help="1.0 is perfect, 0.5 is a coin flip")
    b.metric("Readmissions found in top 15%", f"{g['top15']['recall'] * 100:.1f}%")
    c.metric("Hit rate in top 15%", f"{g['top15']['precision'] * 100:.1f}%", f"base rate {M['base_rate'] * 100:.1f}%")
    e.metric("Lift over random", f"{g['top15']['lift']:.2f}x")
    st.subheader("Does the model beat simple alternatives?")
    tab = pd.DataFrame([{"Model": {"rule_prior_inpatient": "Rule: count of prior inpatient stays", "logistic": "Logistic regression",
                                   "gradient_boosting": "Gradient boosting (final)"}[k],
                         "AUC": round(v["auc"], 3), "95% interval": f"{v['auc_ci'][0]:.3f} to {v['auc_ci'][1]:.3f}",
                         "Hit rate in top 15%": f"{v['top15']['precision'] * 100:.1f}%", "Lift": round(v["top15"]["lift"], 2)}
                        for k, v in m.items()])
    st.dataframe(tab, hide_index=True, width="stretch")
    st.subheader("Do predicted risks match what happened?")
    cal = pd.DataFrame(M["calibration_table"])
    cal["Risk tenth"] = cal["b"] + 1
    st.line_chart(cal.set_index("Risk tenth")[["predicted", "observed"]])
    st.caption("Each point is one tenth of the test discharges, from lowest to highest estimated risk. "
               "The two lines stay close, so the percentages can be read as rough probabilities.")
    st.info("The gain over a simple rule is real but small. This is a ranking aid, and the data are old. See the Limits tab.")

with t2:
    st.subheader("Highest risk discharges in the held out test group")
    top = d.head(200)[["encounter_id", "risk", "age", "gender", "number_inpatient", "time_in_hospital", "dx1", "y"]].copy()
    top["risk"] = (top.risk * 100).round(1)
    top["y"] = top.y.map({1: "readmitted", 0: "not readmitted"})
    top.columns = ["Encounter", "Risk %", "Age", "Gender", "Prior inpatient", "Days", "Main diagnosis", "What happened"]
    st.dataframe(top, hide_index=True, width="stretch", height=260)
    st.caption("The last column is the real outcome, shown so you can see how the ranking performed. A care team would not have it.")
    pick = st.selectbox("Open a discharge", d.head(200).encounter_id.tolist())
    row = d[d.encounter_id == pick].iloc[0]
    bundle = findings_for(row)
    res = run_note(bundle, RulesOnlyWriter())
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Checked note**")
        if res["status"] == "ready":
            for s in res["statements"]:
                st.write(s["text"])
        else:
            st.error("The checker rejected this note: " + "; ".join(res["problems"]))
        st.caption("This note is built by rules from the facts on the right. With an AI writer the same checker "
                   "validates every sentence before anyone sees it.")
    with right:
        st.markdown("**Facts the note may use**")
        for f in bundle["findings"]:
            with st.expander(f["id"]):
                for fact in f["facts"]:
                    st.write("• " + fact)

with t3:
    st.subheader("Who gets flagged?")
    st.write("Race and gender are not inputs to the model. They are used here only to check the results. "
             "'Flagged' means in the top 15% by estimated risk.")
    for key, label in [("race", "Race"), ("gender", "Gender"), ("age_band", "Age")]:
        st.markdown(f"**{label}**")
        t = pd.DataFrame(M["audit"][key])
        t = t.assign(base_rate=(t.base_rate * 100).round(1), flagged=(t.flagged * 100).round(1), recall=(t.recall * 100).round(1),
                     fpr=(t.fpr * 100).round(1), auc=t.auc.round(3))
        t.columns = ["Group", "Stays", "Readmitted %", "Flagged %", "Readmissions caught %", "False alarm %", "AUC", "Mean risk"]
        t["Mean risk"] = (t["Mean risk"] * 100).round(1)
        st.dataframe(t, hide_index=True, width="stretch")
    st.warning("Two gaps to know about. Black patients are flagged slightly more often than white patients, with more false "
               "alarms. The model catches far fewer readmissions in patients 80 and over than in younger patients. "
               "Groups under 300 stays or 20 readmissions are left out because the numbers are too small to read.")

with t4:
    st.subheader("CMS Hospital Readmissions Reduction Program, " + C["period"])
    st.write(f"{C['hospitals']:,} hospitals. {C['suppressed_pairs']:,} of {C['suppressed_pairs'] + C['reported_pairs']:,} "
             "hospital and condition results are hidden by CMS because the counts are too small, and are left out here.")
    bc = pd.DataFrame(C["by_condition"])
    bc.columns = ["Condition", "Hospitals", "Share worse than expected", "Median ratio", "Worst 10% start at"]
    bc["Share worse than expected"] = (bc["Share worse than expected"] * 100).round(1).astype(str) + "%"
    st.dataframe(bc, hide_index=True, width="stretch")
    st.caption("A ratio above 1.0 means more readmissions than expected for that hospital's patient mix.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**States with the lowest median ratio**")
        st.dataframe(pd.DataFrame(C["states_best"])[["State", "hospitals", "median_ratio"]], hide_index=True)
    with c2:
        st.markdown("**States with the highest median ratio**")
        st.dataframe(pd.DataFrame(C["states_worst"])[["State", "hospitals", "median_ratio"]], hide_index=True)
    st.caption("Only states with at least 20 hospitals. State differences are small and do not show cause.")
    st.markdown("**Do hospitals that struggle with one condition struggle with others?**")
    st.write(f"Somewhat. Heart failure and pneumonia ratios correlate at {C['hf_pn_corr']:.2f}. Of {C['hospitals_4plus_conditions']:,} "
             f"hospitals with four or more conditions reported, {C['consistently_worse_share'] * 100:.1f}% are worse than expected on every "
             f"condition they report. If each condition were independent, about {C['chance_share_all_worse_if_independent'] * 100:.1f}% would be.")

with t5:
    st.subheader("What this project does not show")
    st.markdown(f"""
- **The patient data are from 1999 to 2008.** Practice has changed. Treat the results as a demonstration of method, not a tool for use today.
- **Modest accuracy.** AUC {g['auc']:.2f}. Readmission is hard to predict from billing and stay records.
- **One health system dataset, 130 hospitals.** Results may not carry over to other hospitals.
- **Readmission means any return within 30 days in the same data.** Returns to other hospitals may be missed.
- **Patient level and hospital level data are separate.** The CMS data cannot be linked to the UCI patients, so they are shown side by side, not combined.
- **The checker is strong but not perfect.** In {N['total_tried']:,} deliberately broken notes it caught {N['total_caught']:,}. The one miss was a wrong number that happened to equal another number in the same finding. The checker confirms numbers appear in the cited findings, not that they sit in the right sentence.
- **Dropped step.** I tried isotonic recalibration. It did not improve the Brier score and gave implausible risks in sparse regions, so the raw model is used.
""")
