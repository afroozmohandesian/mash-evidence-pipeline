"""Part 2 — Trial landscape mapping from ClinicalTrials.gov (API v2).

Design choices
--------------
* All registry fields that are already structured (phase, status, sponsor,
  dates, enrollment) are taken directly from the API. They are never routed
  through the LLM: the LLM can only add noise to data that is already clean.
* For trials with posted results, a compact text summary of the primary
  outcome results and serious adverse events is built. This is what lets the
  LLM step extract efficacy and safety for registry records.
* Each trial is tagged with a disease scope so the analysis can separate
  core MASH/NASH trials from broader MASLD/NAFLD studies.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

from . import config

QUERY_PARAMS = {
    "query.cond": (
        "NASH OR MASH OR nonalcoholic steatohepatitis "
        "OR metabolic dysfunction-associated steatohepatitis"
    ),
    "filter.advanced": (
        "AREA[StudyType]INTERVENTIONAL AND "
        "(AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)"
    ),
    "countTotal": "true",
    "format": "json",
}

SPONSOR_TYPE = {
    "INDUSTRY": "Industry",
    "NIH": "Government",
    "FED": "Government",
    "OTHER_GOV": "Government",
    "OTHER": "Academic / hospital",
    "NETWORK": "Academic / hospital",
    "INDIV": "Other",
}

STATUS_GROUP = {
    "COMPLETED": "Completed",
    "RECRUITING": "Ongoing",
    "ACTIVE_NOT_RECRUITING": "Ongoing",
    "NOT_YET_RECRUITING": "Ongoing",
    "ENROLLING_BY_INVITATION": "Ongoing",
    "TERMINATED": "Stopped early",
    "WITHDRAWN": "Stopped early",
    "SUSPENDED": "Stopped early",
}

MASH_PATTERN = re.compile(
    r"\b(mash|nash)\b|steatohepatitis", re.IGNORECASE)
MASLD_PATTERN = re.compile(
    r"\b(nafld|masld|mafld)\b|fatty liver|hepatic steatosis", re.IGNORECASE)


# ---------------------------------------------------------------- collection
def collect(refresh: bool = False) -> list[dict]:
    """Page through the API and save the raw study records."""
    if config.TRIALS_RAW_FILE.exists() and not refresh:
        studies = json.loads(config.TRIALS_RAW_FILE.read_text())
        print(f"Loaded {len(studies)} cached trial records")
        return studies

    studies, token, total = [], None, None
    while True:
        params = {**QUERY_PARAMS, "pageSize": config.CTGOV_PAGE_SIZE}
        if token:
            params["pageToken"] = token
        resp = requests.get(config.CTGOV_API_URL, params=params, timeout=60)
        resp.raise_for_status()
        page = resp.json()
        total = total or page.get("totalCount")
        studies.extend(page.get("studies", []))
        token = page.get("nextPageToken")
        if not token:
            break
        time.sleep(0.3)

    config.TRIALS_RAW_FILE.write_text(json.dumps(studies, ensure_ascii=False))
    print(f"Saved {len(studies)} raw trial records "
          f"(API total {total}) -> {config.TRIALS_RAW_FILE.relative_to(config.PROJECT_ROOT)}")
    return studies


# ---------------------------------------------------------------- flattening
def _get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def format_phase(phases: list[str]) -> str:
    """['PHASE2', 'PHASE3'] -> 'Phase 2/3'."""
    nums = sorted({p.replace("PHASE", "") for p in phases if p.startswith("PHASE")})
    return "Phase " + "/".join(nums) if nums else "Not applicable"


def highest_phase(phases: list[str]) -> int | None:
    nums = [int(p[-1]) for p in phases if p.startswith("PHASE")]
    return max(nums) if nums else None


def disease_scope(text: str) -> str:
    if MASH_PATTERN.search(text):
        return "MASH/NASH"
    if MASLD_PATTERN.search(text):
        return "MASLD/NAFLD (broader)"
    return "Other"


def summarize_results(study: dict, max_chars: int = 3000) -> str | None:
    """Compact text of primary-outcome results and serious AEs, for the LLM."""
    results = study.get("resultsSection")
    if not results:
        return None

    lines = []
    for om in _get(results, "outcomeMeasuresModule", "outcomeMeasures", default=[]):
        if om.get("type") != "PRIMARY":
            continue
        groups = {g["id"]: g.get("title", g["id"]) for g in om.get("groups", [])}
        lines.append(f"PRIMARY OUTCOME: {om.get('title')} "
                     f"[{om.get('unitOfMeasure', '')}, {om.get('timeFrame', '')}]")
        for cls in om.get("classes", []):
            for cat in cls.get("categories", []):
                for m in cat.get("measurements", []):
                    lines.append(f"  {groups.get(m.get('groupId'), m.get('groupId'))}: "
                                 f"{m.get('value')}")
        for a in om.get("analyses", []):
            if a.get("pValue"):
                lines.append(f"  p-value: {a['pValue']} ({a.get('statisticalMethod', '')})")

    for g in _get(results, "adverseEventsModule", "eventGroups", default=[]):
        lines.append(f"SERIOUS AEs - {g.get('title')}: "
                     f"{g.get('seriousNumAffected')}/{g.get('seriousNumAtRisk')} affected; "
                     f"deaths {g.get('deathsNumAffected', 'NR')}")

    text = "\n".join(lines)
    return text[:max_chars] if text else None


def flatten(study: dict) -> dict:
    p = study.get("protocolSection", {})
    ident, status = p.get("identificationModule", {}), p.get("statusModule", {})
    design, sponsor = p.get("designModule", {}), p.get("sponsorCollaboratorsModule", {})
    phases = design.get("phases", [])
    conditions = _get(p, "conditionsModule", "conditions", default=[])
    interventions = _get(p, "armsInterventionsModule", "interventions", default=[])
    lead = sponsor.get("leadSponsor", {})
    overall = status.get("overallStatus")
    start = _get(status, "startDateStruct", "date")

    scope_text = " ".join([ident.get("briefTitle", ""), ident.get("officialTitle", ""),
                           *conditions])
    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "acronym": ident.get("acronym"),
        "phase": format_phase(phases),
        "highest_phase": highest_phase(phases),
        "overall_status": overall,
        "status_group": STATUS_GROUP.get(overall, "Unknown"),
        "start_date": start,
        "start_year": int(start[:4]) if start else None,
        "completion_date": _get(status, "completionDateStruct", "date"),
        "enrollment": _get(design, "enrollmentInfo", "count"),
        "sponsor_name": lead.get("name"),
        "sponsor_class": lead.get("class"),
        "sponsor_type": SPONSOR_TYPE.get(lead.get("class"), "Other"),
        "collaborators": [c.get("name") for c in sponsor.get("collaborators", [])],
        "conditions": conditions,
        "disease_scope": disease_scope(scope_text),
        "interventions": [
            {"name": i.get("name"), "type": i.get("type"),
             "description": (i.get("description") or "")[:500]}
            for i in interventions
        ],
        "primary_outcomes": [
            f"{o.get('measure')} ({o.get('timeFrame', '')})"
            for o in _get(p, "outcomesModule", "primaryOutcomes", default=[])
        ],
        "has_results": study.get("hasResults", False),
        "results_summary": summarize_results(study),
    }


def to_dataframe(studies: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([flatten(s) for s in studies]).drop_duplicates("nct_id")
    return df.reset_index(drop=True)
