"""Hospital level layer: FY2026 CMS Hospital Readmissions Reduction Program."""
import json
import numpy as np
import pandas as pd

RAW = "data/raw/hrrp_fy2026.csv"
NAMES = {"READM-30-HF-HRRP": "Heart failure", "READM-30-AMI-HRRP": "Heart attack", "READM-30-PN-HRRP": "Pneumonia",
         "READM-30-COPD-HRRP": "COPD", "READM-30-HIP-KNEE-HRRP": "Hip and knee replacement", "READM-30-CABG-HRRP": "Bypass surgery"}


def load(path=RAW):
    h = pd.read_csv(path, low_memory=False)
    for c in ["Number of Discharges", "Excess Readmission Ratio", "Predicted Readmission Rate",
              "Expected Readmission Rate", "Number of Readmissions"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")       # "Too Few to Report" becomes NaN
    h["condition"] = h["Measure Name"].map(NAMES)
    return h


def summarize(h):
    ok = h.dropna(subset=["Excess Readmission Ratio"]).copy()
    out = {"period": f"{h['Start Date'].iloc[0]} to {h['End Date'].iloc[0]}",
           "hospitals": int(h["Facility ID"].nunique()),
           "reported_pairs": int(len(ok)), "suppressed_pairs": int(len(h) - len(ok))}
    ok["worse"] = ok["Excess Readmission Ratio"] > 1
    out["by_condition"] = (ok.groupby("condition").agg(hospitals=("worse", "size"), share_worse_than_expected=("worse", "mean"),
                           median_ratio=("Excess Readmission Ratio", "median"),
                           p90_ratio=("Excess Readmission Ratio", lambda s: s.quantile(.9))).round(3).reset_index().to_dict("records"))
    # size: discharges tertiles within condition
    ok["size"] = ok.groupby("condition")["Number of Discharges"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["small", "medium", "large"]))
    out["by_size"] = (ok.groupby(["condition", "size"], observed=True).agg(n=("worse", "size"), share_worse=("worse", "mean"),
                      median_ratio=("Excess Readmission Ratio", "median")).round(3).reset_index().assign(size=lambda x: x["size"].astype(str)).to_dict("records"))
    # state
    st = ok.groupby("State").agg(hospitals=("Facility ID", "nunique"), pairs=("worse", "size"), share_worse=("worse", "mean"),
                                 median_ratio=("Excess Readmission Ratio", "median")).round(3)
    st = st[st.hospitals >= 20].sort_values("median_ratio")
    out["states_best"] = st.head(5).reset_index().to_dict("records")
    out["states_worst"] = st.tail(5).iloc[::-1].reset_index().to_dict("records")
    # does a bad hospital stay bad? correlation of ratios across conditions
    w = ok.pivot_table(index="Facility ID", columns="condition", values="Excess Readmission Ratio")
    c = w.corr(min_periods=300)
    out["cross_condition_corr_mean"] = float(c.where(~np.eye(len(c), dtype=bool)).stack().mean())
    both = w.dropna(subset=["Heart failure", "Pneumonia"])
    out["hf_pn_corr"] = float(both["Heart failure"].corr(both["Pneumonia"]))
    n = w.notna().sum(axis=1)
    multi = w[n >= 4]
    out["hospitals_4plus_conditions"] = int(len(multi))
    out["consistently_worse_all"] = int((multi.gt(1) | multi.isna()).all(axis=1).sum())
    out["consistently_worse_share"] = float((multi.gt(1) | multi.isna()).all(axis=1).mean())
    # what chance would give: if each condition independent coin flip at observed share
    out["chance_share_all_worse_if_independent"] = float(np.prod([ok[ok.condition == k].worse.mean() for k in ["Heart failure", "Pneumonia", "Heart attack", "COPD"]]))
    # data quality: ratio consistent with predicted/expected
    chk = ok.dropna(subset=["Predicted Readmission Rate", "Expected Readmission Rate"])
    out["max_ratio_mismatch"] = float((chk["Predicted Readmission Rate"] / chk["Expected Readmission Rate"] - chk["Excess Readmission Ratio"]).abs().max())
    return out


if __name__ == "__main__":
    s = summarize(load())
    json.dump(s, open("eval/results_cms.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k not in ("by_size",)}, indent=1)[:3500])
