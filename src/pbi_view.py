"""A Power BI style report page: slicers on top, KPI cards, then a grid of visuals that all respond to the slicers."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BLUE, NAVY, ORANGE, YELLOW, RED, GREY = "#118DFF", "#12239E", "#E66C37", "#D9B300", "#D64550", "#A6A6A6"

CSS = """
<style>
.stApp {background:#F3F2F1;}
.block-container {padding-top:1.2rem; max-width:1400px;}
.pbi-title {background:#12239E; color:white; padding:14px 22px; border-radius:4px; display:flex; justify-content:space-between; align-items:center;}
.pbi-title h2 {margin:0; padding:0; font-size:1.35rem !important; color:white !important; font-weight:600;}
.pbi-title span {font-size:.8rem; opacity:.85;}
.card {min-height:112px; background:white; border:1px solid #E1DFDD; border-radius:4px; padding:12px 16px; box-shadow:0 1px 2px rgba(0,0,0,.06);}
.card .lab {font-size:.78rem; color:#605E5C; margin-bottom:2px;}
.card .val {font-size:2rem; font-weight:600; color:#252423; line-height:1.15;}
.card .sub {font-size:.75rem; color:#605E5C;}
[class*="st-key-vis_"] {background:#FFFFFF !important; border:1px solid #E1DFDD !important; border-radius:4px !important; box-shadow:0 1px 2px rgba(0,0,0,.06);}
.vt {font-size:.92rem; font-weight:600; color:#252423; margin-bottom:2px;}
div[data-baseweb="select"] > div {background:white;}
</style>
"""


def auc(y, p):
    """AUC by rank formula, no sklearn needed."""
    y = np.asarray(y); p = pd.Series(p).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return float((p[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def card(col, label, value, sub=""):
    col.markdown(f'<div class="card"><div class="lab">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


_n = [0]


def vis(title):
    _n[0] += 1
    box = st.container(border=True, key=f"vis_{_n[0]}")
    box.markdown(f'<div class="vt">{title}</div>', unsafe_allow_html=True)
    return box


def style(ch, h=230):
    return ch.properties(height=h, background="#FFFFFF").configure_view(strokeWidth=0).configure_axis(labelColor="#605E5C", titleColor="#605E5C", gridColor="#EDEBE9", domainColor="#C8C6C4")


def render(d, M, C):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="pbi-title"><h2>Readmission Risk Report</h2><span>Patient data: 99,340 US stays, 1999 to 2008 &nbsp;|&nbsp; Hospital data: CMS FY2026</span></div>', unsafe_allow_html=True)
    st.write("")
    s1, s2, s3, s4 = st.columns(4)
    gender = s1.multiselect("Gender", sorted(d.gender.unique()), placeholder="All")
    race = s2.multiselect("Race", sorted(d.race.unique()), placeholder="All")
    age = s3.multiselect("Age band", ["under 40", "40 to 59", "60 to 79", "80 plus"], placeholder="All")
    dx = s4.multiselect("Main diagnosis", sorted(d.dx1.unique()), placeholder="All")
    f = d
    for col, sel in (("gender", gender), ("race", race), ("age_band", age), ("dx1", dx)):
        if sel:
            f = f[f[col].isin(sel)]
    if len(f) < 200 or f.y.sum() < 20:
        st.warning("Too few discharges for a reliable view. Clear a slicer.")
        return
    cutoff = M["flag_cutoff"]
    flag = f.risk >= cutoff
    base_all = d.y.mean()

    k = st.columns(5)
    card(k[0], "Discharges", f"{len(f):,}", f"{len(f) / len(d) * 100:.0f}% of test set")
    card(k[1], "Readmitted within 30 days", f"{f.y.mean() * 100:.1f}%", f"overall {base_all * 100:.1f}%")
    card(k[2], "Average predicted risk", f"{f.risk.mean() * 100:.1f}%", "model estimate")
    card(k[3], "AUC (this selection)", f"{auc(f.y, f.risk):.3f}", "0.5 is a coin flip")
    card(k[4], "Readmissions caught in top 15%", f"{flag[f.y == 1].mean() * 100:.1f}%", f"false alarm rate {flag[f.y == 0].mean() * 100:.1f}%")
    st.write("")

    r1a, r1b, r1c = st.columns([5, 4, 4])
    with r1a:
        box = vis("Predicted vs actual readmission rate by risk tenth")
        t = f.assign(tenth=pd.qcut(f.risk.rank(method="first"), 10, labels=False) + 1).groupby("tenth").agg(Predicted=("risk", "mean"), Actual=("y", "mean")).reset_index()
        t = t.melt("tenth", var_name="Measure", value_name="rate")
        ch = alt.Chart(t).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X("tenth:O", title="Risk tenth (1 lowest)", axis=alt.Axis(labelAngle=0)), xOffset="Measure:N",
            y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=["Predicted", "Actual"], range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["tenth", "Measure", alt.Tooltip("rate:Q", format=".1%")])
        box.altair_chart(style(ch), width="stretch")
    with r1b:
        box = vis("Readmission rate by prior inpatient stays")
        pb = f.assign(b=pd.cut(f.number_inpatient, [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3 or more"])).groupby("b", observed=True).agg(rate=("y", "mean"), n=("y", "size")).reset_index()
        ch = alt.Chart(pb).mark_bar(color=NAVY, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X("b:O", title="Prior inpatient stays, past year", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%")), tooltip=["b", alt.Tooltip("rate:Q", format=".1%"), "n"])
        lab = ch.mark_text(dy=-6, color="#252423", fontSize=11).encode(text=alt.Text("rate:Q", format=".1%"))
        box.altair_chart(style(ch + lab), width="stretch")
    with r1c:
        box = vis("Readmission rate by diagnosis (red line = overall)")
        dd = f.groupby("dx1").agg(rate=("y", "mean"), n=("y", "size")).reset_index()
        dd = dd[dd.n >= 100]
        ch = alt.Chart(dd).mark_bar(color=BLUE).encode(y=alt.Y("dx1:N", sort="-x", title=None), x=alt.X("rate:Q", title=None, axis=alt.Axis(format="%")),
                                                      tooltip=["dx1", alt.Tooltip("rate:Q", format=".1%"), "n"])
        rule = alt.Chart(pd.DataFrame({"x": [base_all]})).mark_rule(color=RED, strokeDash=[4, 3]).encode(x="x:Q")
        box.altair_chart(style(ch + rule), width="stretch")

    st.write("")
    r2a, r2b, r2c = st.columns([4, 4, 5])
    with r2a:
        box = vis("Who gets flagged (top 15%) by group")
        rows = []
        for col, lab in (("gender", "Gender"), ("age_band", "Age")):
            for key, g in f.groupby(col):
                if len(g) >= 150:
                    rows.append({"group": str(key), "flagged": float((g.risk >= cutoff).mean())})
        fl = pd.DataFrame(rows)
        ch = alt.Chart(fl).mark_bar(color=YELLOW).encode(y=alt.Y("group:N", sort="-x", title=None), x=alt.X("flagged:Q", title=None, axis=alt.Axis(format="%")),
                                                         tooltip=["group", alt.Tooltip("flagged:Q", format=".1%")])
        box.altair_chart(style(ch), width="stretch")
    with r2b:
        box = vis("Readmissions caught by age band")
        ab = []
        for key, g in f.groupby("age_band"):
            if g.y.sum() >= 15:
                ab.append({"age": key, "caught": float((g.risk >= cutoff)[g.y == 1].mean())})
        ab = pd.DataFrame(ab)
        ch = alt.Chart(ab).mark_bar(color=ORANGE).encode(x=alt.X("age:N", sort=["under 40", "40 to 59", "60 to 79", "80 plus"], title=None, axis=alt.Axis(labelAngle=0)),
                                                         y=alt.Y("caught:Q", title=None, axis=alt.Axis(format="%")), tooltip=["age", alt.Tooltip("caught:Q", format=".1%")])
        box.altair_chart(style(ch), width="stretch")
    with r2c:
        box = vis("CMS FY2026: share of hospitals worse than expected")
        bc = pd.DataFrame(C["by_condition"])
        ch = alt.Chart(bc).mark_bar(color=NAVY).encode(y=alt.Y("condition:N", sort="-x", title=None),
                                                      x=alt.X("share_worse_than_expected:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                                                      tooltip=["condition", alt.Tooltip("share_worse_than_expected:Q", format=".1%"), "hospitals"])
        box.altair_chart(style(ch), width="stretch")
    st.caption("Slicers filter every patient visual and card. The CMS visual is hospital level and cannot be linked to patients, so it does not respond to slicers.")
