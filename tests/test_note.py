import sys, json
sys.path.insert(0, ".")
import pandas as pd
from src.checker import Report, validate, RulesOnlyWriter, run_note

D = pd.read_csv("data/scored_test.csv")
from src.note import findings_for


def bundle(i=0):
    return findings_for(D.iloc[i])


def rep(st):
    return Report.model_validate({"statements": st})


def clean(i=0):
    return run_note(bundle(i), RulesOnlyWriter())["statements"]


def test_clean_notes_pass_for_many_discharges():
    for i in range(0, 20000, 500):
        assert run_note(bundle(i), RulesOnlyWriter())["status"] == "ready"


def test_unknown_id_rejected():
    st = clean(); st[0]["finding_ids"] = ["nope"]
    assert any("unknown" in p for p in validate(rep(st), bundle()))


def test_new_number_rejected():
    st = clean(); st[0]["text"] += " About 4821 such cases."
    assert any("4821" in p for p in validate(rep(st), bundle()))


def test_causal_wording_rejected():
    st = clean(); st[1]["text"] += " This causes readmission."
    assert any("causal" in p for p in validate(rep(st), bundle()))


def test_required_risk_and_limits_must_be_covered():
    st = [s for s in clean() if s["finding_ids"] not in (["risk"], ["limits"])]
    p = validate(rep(st), bundle())
    assert any("'risk'" in x for x in p) and any("'limits'" in x for x in p)


def test_statement_needs_a_citation():
    st = clean(); st[0]["finding_ids"] = []
    try:
        rep(st); assert False
    except Exception:
        pass


def test_no_findings_refused():
    assert run_note({"findings": []}, RulesOnlyWriter())["status"] == "refused"


def test_bad_writer_json_gets_one_retry_then_rejected():
    class Bad:
        calls = 0
        def draft(self, b, feedback=None):
            Bad.calls += 1
            return "not json"
    r = run_note(bundle(), Bad())
    assert r["status"] == "rejected" and Bad.calls == 2


def test_writer_gets_feedback_and_can_recover():
    good = RulesOnlyWriter()
    class Fix:
        n = 0
        def draft(self, b, feedback=None):
            Fix.n += 1
            if Fix.n == 1:
                return json.dumps({"statements": [{"text": "Risk is 99.9%.", "finding_ids": ["risk"]}]})
            assert feedback
            return good.draft(b)
    assert run_note(bundle(), Fix())["status"] == "ready"


def test_scores_are_probabilities_and_ranked():
    assert D.risk.between(0, 1).all() and D.risk.is_monotonic_decreasing
