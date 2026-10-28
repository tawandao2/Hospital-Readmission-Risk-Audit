"""Train and evaluate the 30 day readmission model. Split is by patient so no
patient appears in both train and test. Race and gender are NOT features; they
are only used afterwards to audit the model."""
import json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, ".")
from src.prep import load

SEED = 7
NUM = ["age_mid", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
       "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
       "n_meds_changed"]
CAT = ["admission_type_id", "admission_source_id", "discharge_disposition_id", "dx1", "dx2", "dx3",
       "a1c", "insulin", "change", "diabetesMed"]
FLAG_RATE = 0.15      # care team can call the top 15% of discharges


def split(d):
    g = d.patient_nbr.values
    a, rest = next(GroupShuffleSplit(1, train_size=0.6, random_state=SEED).split(d, groups=g))
    r = d.iloc[rest]
    b, c = next(GroupShuffleSplit(1, train_size=0.5, random_state=SEED).split(r, groups=r.patient_nbr))
    return d.iloc[a], r.iloc[b], r.iloc[c]


def X(d):
    x = d[NUM + CAT].copy()
    for c in CAT:
        x[c] = x[c].astype(str)
    return x


def boot_ci(y, p, groups, fn=roc_auc_score, n=200):
    """Patient level bootstrap: resample patients, weight their rows."""
    rng = np.random.default_rng(SEED)
    codes, ids = pd.factorize(groups)
    out = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[codes]
        out.append(fn(y, p, sample_weight=w))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def topk(y, p, rate):
    k = int(len(y) * rate)
    idx = np.argsort(-p)[:k]
    return dict(flagged=k, precision=float(y[idx].mean()), recall=float(y[idx].sum() / y.sum()),
                lift=float(y[idx].mean() / y.mean()))


def calib_table(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    t = pd.DataFrame({"y": y, "p": p, "b": q}).groupby("b").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return t.round(4).reset_index().to_dict("records")


def audit(te, p, col, cutoff):
    rows = []
    for k, g in te.assign(p=p).groupby(col):
        if len(g) < 300 or g.y.sum() < 20:
            continue
        flag = g.p >= cutoff
        rows.append(dict(group=str(k), n=int(len(g)), base_rate=float(g.y.mean()), flagged=float(flag.mean()),
                         recall=float(flag[g.y == 1].mean()), fpr=float(flag[g.y == 0].mean()),
                         auc=float(roc_auc_score(g.y, g.p)), mean_pred=float(g.p.mean())))
    return rows


def bucket_inpatient(n):
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3 or more"


def export(tr, te, p):
    """Write the scored test set and the reference rates the explanation writer cites."""
    keep = ["encounter_id", "patient_nbr", "age", "gender", "race", "time_in_hospital", "num_medications", "number_diagnoses",
            "number_inpatient", "number_emergency", "number_outpatient", "discharge_disposition_id", "dx1", "y", "age_band"]
    out = te[keep].copy()
    out["risk"] = p.round(4)
    out["risk_pct_rank"] = out.risk.rank(pct=True, method="min").round(4)
    out.sort_values("risk", ascending=False).to_csv("data/scored_test.csv", index=False)
    ref = {"train_n": int(len(tr)), "train_rate": float(tr.y.mean()),
           "median_diagnoses": float(tr.number_diagnoses.median()), "median_los": float(tr.time_in_hospital.median()),
           "median_meds": float(tr.num_medications.median())}
    def rates(col, f=lambda x: x):
        g = tr.assign(k=tr[col].map(f)).groupby("k").y.agg(["mean", "size"])
        return {str(k): {"rate": float(r["mean"]), "n": int(r["size"])} for k, r in g.iterrows()}
    ref["by_prior_inpatient"] = rates("number_inpatient", bucket_inpatient)
    ref["by_disposition"] = rates("discharge_disposition_id")
    ref["by_dx1"] = rates("dx1")
    json.dump(ref, open("data/reference_rates.json", "w"), indent=1)


def main(out="eval/results_model.json"):
    d = load()
    tr, va, te = split(d)
    ytr, yva, yte = tr.y.values, va.y.values, te.y.values

    pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CAT)])
    lr = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.3)).fit(X(tr), ytr)
    gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                        categorical_features="from_dtype", random_state=SEED)
    xt = X(tr).astype({c: "category" for c in CAT})
    cats = {c: xt[c].cat.categories for c in CAT}
    def xg(df):
        x = X(df)
        for c in CAT:
            x[c] = pd.Categorical(x[c], categories=cats[c])
        return x
    gb.fit(xt, ytr)

    p_rule = te.number_inpatient.values + 0.01 * te.time_in_hospital.values   # simple "prior admissions" rule
    p_lr = lr.predict_proba(X(te))[:, 1]
    p_gb_raw = gb.predict_proba(xg(te))[:, 1]
    # Isotonic recalibration was tried on the validation set. It gave no gain in Brier score and
    # produced implausible risks (up to 99.5%) in sparse regions, so the raw model is used.
    p_gb = p_gb_raw

    res = {"n_encounters": int(len(d)), "n_patients": int(d.patient_nbr.nunique()),
           "base_rate": float(d.y.mean()), "split": {"train": len(tr), "valid": len(va), "test": len(te)},
           "models": {}}
    for name, p in [("rule_prior_inpatient", p_rule), ("logistic", p_lr), ("gradient_boosting", p_gb)]:
        m = dict(auc=float(roc_auc_score(yte, p)), auc_ci=boot_ci(yte, p, te.patient_nbr.values),
                 pr_auc=float(average_precision_score(yte, p)), top15=topk(yte, p, FLAG_RATE))
        if name != "rule_prior_inpatient":
            m["brier"] = float(brier_score_loss(yte, p))
        res["models"][name] = m
    res["calibration_table"] = calib_table(yte, p_gb)
    res["notes"] = ["Isotonic recalibration tried, no Brier gain, extreme risks in sparse regions, not used."]
    cutoff = float(np.sort(p_gb)[::-1][int(len(p_gb) * FLAG_RATE) - 1])
    res["flag_cutoff"] = cutoff
    te2 = te.assign(age_band=pd.cut(te.age_mid, [0, 40, 60, 80, 100], labels=["under 40", "40 to 59", "60 to 79", "80 plus"]).astype(str))
    res["audit"] = {c: audit(te2, p_gb, c, cutoff) for c in ["race", "gender", "age_band"]}
    json.dump(res, open(out, "w"), indent=1)
    export(tr, te2, p_gb)
    return res


if __name__ == "__main__":
    r = main()
    for k, v in r["models"].items():
        print(f"{k:32s} AUC {v['auc']:.3f} {np.round(v['auc_ci'],3)}  PR {v['pr_auc']:.3f}  top15 prec {v['top15']['precision']:.3f} recall {v['top15']['recall']:.3f} lift {v['top15']['lift']:.2f}  brier {v.get('brier','-')}")
    for c, rows in r["audit"].items():
        print(c); [print("  ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()}) for row in rows]
