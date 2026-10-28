"""Does the checker catch bad notes? Take clean notes for many discharges, break each in a known way, count catches."""
import json, random, sys
sys.path.insert(0, ".")
import pandas as pd
from src.checker import RulesOnlyWriter, Report, validate, NUM, run_note
from src.note import findings_for

random.seed(3)


def mutations(rep, bundle):
    st = rep["statements"]
    out = {}
    def clone():
        return json.loads(json.dumps(st))
    # wrong number
    s = clone(); t = s[0]["text"]
    m = NUM.search(t.split(".")[1]) if False else None
    import re
    nums = [x for x in re.findall(r"\d+\.\d", t)]
    if nums:
        s[0]["text"] = t.replace(nums[0], str(round(float(nums[0]) + 7.3, 1)), 1); out["wrong_number"] = s
    s = clone(); s[1]["text"] += " This is why the patient will be readmitted because of the prior stays."; out["causal_wording"] = s
    s = clone(); s[2]["finding_ids"] = ["made_up_id"]; out["unknown_id"] = s
    s = clone(); s[0]["text"] += " Roughly 1234 similar patients were seen."; out["invented_number"] = s
    s = [x for x in clone() if "risk" not in x["finding_ids"]]; out["drops_risk"] = s
    s = [x for x in clone() if "limits" not in x["finding_ids"]]; out["drops_limits"] = s
    s = clone(); s[1]["finding_ids"] = []; out["no_citation"] = s
    s = clone(); s[0]["text"] += " Leads to a poor outcome."; out["causal_leads_to"] = s
    return out


def main(n=400, out="eval/results_notes.json"):
    d = pd.read_csv("data/scored_test.csv").sample(n, random_state=3)
    clean_ok, caught, total = 0, {}, {}
    for _, row in d.iterrows():
        b = findings_for(row)
        r = run_note(b, RulesOnlyWriter())
        if r["status"] == "ready":
            clean_ok += 1
        for name, st in mutations(r, b).items():
            total[name] = total.get(name, 0) + 1
            try:
                rep = Report.model_validate({"statements": st})
                bad = bool(validate(rep, b))
            except Exception:
                bad = True
            caught[name] = caught.get(name, 0) + bad
    res = {"discharges": n, "clean_notes_passed": clean_ok, "mutations": {k: {"tried": total[k], "caught": caught[k]} for k in total},
           "total_tried": sum(total.values()), "total_caught": sum(caught.values())}
    json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
