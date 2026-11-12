"""LLM quality assurance — checks that need no further API calls.

1. Automated consistency checks: compare LLM output with fields that are
   known independently (registry phase, PubMed publication types, posted
   results, registry interventions, reported p-values).
2. A manual audit: a fixed random sample reviewed by hand against the source
   text; verdicts are stored in MANUAL_AUDIT so the numbers are reproducible.
"""
from __future__ import annotations

import re

import pandas as pd

# Manual audit of 13 records (random_state=7), reviewed against source text.
# Each record has 5 audited fields: drugs+roles, endpoint, efficacy, safety,
# efficacy_direction. Errors list the field and the issue.
MANUAL_AUDIT = [
    ("39002641", "PubMed", []),
    ("40480879", "PubMed", ["efficacy_direction: significant result labelled 'not reported'"]),
    ("37534936", "PubMed", []),
    ("34239138", "PubMed", []),
    ("34570916", "PubMed", []),
    ("36251165", "PubMed", []),
    ("32147075", "PubMed", ["drugs: piperine (absorption enhancer) tagged investigational"]),
    ("NCT04971785", "ClinicalTrials.gov", []),
    ("NCT03823703", "ClinicalTrials.gov", ["evidence_type: registry entry labelled RCT (now set deterministically)"]),
    ("NCT02354976", "ClinicalTrials.gov", []),
    ("NCT06104319", "ClinicalTrials.gov", ["efficacy_direction: 'positive' without a comparator (debatable)"]),
    ("NCT00303537", "ClinicalTrials.gov", []),
    ("NCT01617772", "ClinicalTrials.gov", []),
]
FIELDS_PER_RECORD = 5


def _min_p(text) -> float | None:
    if not isinstance(text, str):
        return None
    values = []
    for v in re.findall(r"p-value: [<>=≤]*\s*([0-9]*\.?[0-9]+)", text):
        try:
            values.append(float(v))
        except ValueError:
            pass
    return min(values) if values else None


def automated_checks(extraction_df: pd.DataFrame, pubmed_df: pd.DataFrame,
                     trials_df: pd.DataFrame) -> pd.DataFrame:
    x = extraction_df
    ct = x[x["data_source"] == "ClinicalTrials.gov"].merge(trials_df, left_on="source_id", right_on="nct_id")
    pm = x[x["data_source"] == "PubMed"].merge(
        pubmed_df.assign(pmid=pubmed_df["pmid"].astype(str)), left_on="source_id", right_on="pmid")

    # 1. phase agreement with registry
    phase_ok = (ct["study_phase"] == ct["phase"]).mean()

    # 2. evidence type agreement with PubMed publication types
    expected = pm["study_design"].map({
        "Meta-analysis / systematic review": "meta-analysis / systematic review",
        "Randomized controlled trial": "RCT", "Phase 2 trial": "RCT", "Phase 3 trial": "RCT"})
    has_exp = expected.notna()
    evid_ok = (pm.loc[has_exp, "evidence_type"] == expected[has_exp]).mean()

    # 3. drug capture: registry lists a non-placebo drug -> LLM finds an investigational drug
    def registry_has_drug(ivs):
        return any(i["type"] in ("DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT")
                   and "placebo" not in (i["name"] or "").lower() for i in ivs)
    reg = ct["interventions"].apply(registry_has_drug) & ct["relevant"]
    found = ct["drugs"].apply(lambda d: any(m["role"] == "investigational" for m in d))
    capture = (reg & found).sum() / reg.sum()

    # 4. no invented results for trials without posted results
    no_res = ct["results_summary"].isna()
    invented = (no_res & ct["efficacy_outcome"].notna()).sum()

    # 5. efficacy direction vs smallest reported p-value
    ct["min_p"] = ct["results_summary"].apply(_min_p)
    w = ct[ct["min_p"].notna()]
    sig = w["min_p"] < 0.05
    consistent = ((sig & w["efficacy_direction"].isin(["positive", "mixed"]))
                  | (~sig & w["efficacy_direction"].eq("negative"))).mean()

    return pd.DataFrame([
        ("Trial phase matches registry", f"{phase_ok:.0%}", f"{len(ct)} trials"),
        ("Evidence type matches PubMed publication type", f"{evid_ok:.0%}", f"{has_exp.sum()} articles"),
        ("Investigational drug captured when registry lists one", f"{capture:.1%}", f"{reg.sum()} trials"),
        ("Efficacy invented for trials without posted results", f"{invented}", f"{no_res.sum()} trials"),
        ("Efficacy direction consistent with reported p-values", f"{consistent:.0%}", f"{len(w)} trials"),
    ], columns=["check", "result", "base"])


def manual_audit_summary() -> pd.DataFrame:
    rows = [{"source_id": s, "data_source": d, "n_errors": len(e), "errors": "; ".join(e)}
            for s, d, e in MANUAL_AUDIT]
    df = pd.DataFrame(rows)
    total = len(df) * FIELDS_PER_RECORD
    errors = df["n_errors"].sum()
    print(f"Manual audit: {len(df)} records, {total} fields, {errors} errors "
          f"-> field accuracy {(total - errors) / total:.0%}")
    return df
