"""Build the published (reduced) report data from the full data files.

Reads the full JSON files from reports/Q4_2026/data_private/ (gitignored) and writes the
reduced copies to reports/Q4_2026/data/, which are the only data committed and published.

Reductions:
- recon.json: per case, only the keys the notebook reads are kept. The per-subject blocks
  (subjects, subject_score, subject_scores, fc) are dropped. The BOLD trace is kept only for
  one illustrative HCP-YA subject and one illustrative OCD subject.
- exclusions.json: subject IDs are removed; only run and reason are kept.
- probes.json, runs.json, provenance.json: copied unchanged after checking for subject IDs.

Usage (from the repository root):
    python reports/Q4_2026/build/publish_data.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
SRC = REPORT / "data_private"
DST = REPORT / "data"

# Only these case keys are read by marimo_report.py.
CASE_KEYS = [
    "case", "cohort", "dataset", "model", "n_subjects", "n_windows",
    "r2_roi", "r2_roi_mean", "r2_roi_median", "corr_sample_mean", "corr_sample_sd", "corr_density",
]
# Cases that carry a BOLD trace, and the single illustrative subject each one shows.
TRACE_SUBJECT = {
    "hcp_baseline": "138837",
    "hcp_A": "138837",
    "ocd_baseline": "sub-control31",
    "ocd_A": "sub-control31",
}
SUBJECT_ID = re.compile(r"^(sub-[A-Za-z0-9]+|\d{6})$")


def load(name):
    with open(SRC / name) as f:
        return json.load(f)


def save(name, obj):
    DST.mkdir(parents=True, exist_ok=True)
    with open(DST / name, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def reduce_recon(recon):
    out = {"atlas": recon["atlas"], "cases": {}}
    for name, case in recon["cases"].items():
        kept = {k: case[k] for k in CASE_KEYS if k in case}
        # The notebook only reads the baseline and A cases (REPORT_CASES, Section II/I plots).
        # Other cases are reduced to whatever keys they have.
        if case["model"] in ("baseline", "A"):
            missing = [k for k in CASE_KEYS if k not in kept]
            if missing:
                sys.exit(f"{name}: expected keys missing from recon.json: {missing}")
        if name in TRACE_SUBJECT:
            trace = case["trace"]
            if trace["subject"] != TRACE_SUBJECT[name]:
                sys.exit(f"{name}: trace subject is {trace['subject']}, expected {TRACE_SUBJECT[name]}")
            kept["trace"] = trace
        out["cases"][name] = kept
    return out


def reduce_exclusions(excl):
    """Keep counts and reasons per cohort; drop the subject IDs of excluded subjects."""
    out = {"note": excl["note"]}
    for cohort in ("hcp_ya", "ocd"):
        block = dict(excl[cohort])
        block["excluded"] = [{"run": r["run"], "reason": r["reason"]} for r in block["excluded"]]
        out[cohort] = block
    return out


def assert_no_subject_ids(name, obj):
    """Fail if any string value in the object is a subject ID."""
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)
        elif isinstance(x, str):
            yield x
    hits = [s for s in walk(obj) if SUBJECT_ID.match(s)]
    if hits:
        sys.exit(f"{name}: subject-like IDs found: {hits[:5]}")


def main():
    recon = reduce_recon(load("recon.json"))
    save("recon.json", recon)

    excl = reduce_exclusions(load("exclusions.json"))
    save("exclusions.json", excl)

    for name in ("probes.json", "runs.json", "provenance.json"):
        obj = load(name)
        assert_no_subject_ids(name, obj)
        save(name, obj)

    assert_no_subject_ids("exclusions.json", excl)
    print("Published data written to", DST)


if __name__ == "__main__":
    main()
