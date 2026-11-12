"""Part 1 — Literature mining from PubMed (NCBI E-utilities).

Design choices
--------------
* The brief asks for *clinical* evidence on MASH *treatment*, so the query
  restricts to human studies with clinical publication types (trials, RCTs,
  meta-analyses, systematic reviews). An unfiltered query returns mostly
  mouse studies and narrative reviews.
* Relevance-sorted results cluster in the most recent years. To cover the
  whole 2020–2025 window evenly we draw a fixed sample per publication year.
* Raw MEDLINE records are written to disk untouched (deliverable #2); the
  flattening to a tidy table is a separate, re-runnable step.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
from Bio import Entrez, Medline

from . import config

Entrez.email = config.NCBI_EMAIL
if config.NCBI_API_KEY:
    Entrez.api_key = config.NCBI_API_KEY

DISEASE_TERMS = [
    '"Non-alcoholic Fatty Liver Disease"[MeSH]',
    "MASH[tiab]",
    "NASH[tiab]",
    '"steatohepatitis"[tiab]',
    '"metabolic dysfunction-associated steatohepatitis"[tiab]',
]

TREATMENT_TERMS = [
    "drug therapy[sh]",
    "treatment[tiab]",
    "therapy[tiab]",
    "efficacy[tiab]",
    "agonist[tiab]",
    "inhibitor[tiab]",
]

PRIMARY_TRIAL_TYPES = [
    "Clinical Trial[pt]",
    "Randomized Controlled Trial[pt]",
    "Clinical Trial, Phase II[pt]",
    "Clinical Trial, Phase III[pt]",
]
META_ANALYSIS_TYPES = ["Meta-Analysis[pt]", "Systematic Review[pt]"]

# Each stratum is a separate search; primary trials exclude meta-analyses so
# a record can only land in one bucket.
STRATA = {
    "primary_trial": (f"({' OR '.join(PRIMARY_TRIAL_TYPES)})"
                      f" NOT ({' OR '.join(META_ANALYSIS_TYPES)})"),
    "meta_analysis": f"({' OR '.join(META_ANALYSIS_TYPES)})",
}


def build_query(year: int, stratum: str) -> str:
    """PubMed query for one publication year and one evidence stratum."""
    return (
        f"({' OR '.join(DISEASE_TERMS)})"
        f" AND ({' OR '.join(TREATMENT_TERMS)})"
        f" AND {STRATA[stratum]}"
        " AND humans[MeSH] AND english[la] AND hasabstract"
        f' AND ("{year}/01/01"[dp] : "{year}/12/31"[dp])'
    )


def search(year: int, stratum: str, retmax: int) -> tuple[int, list[str]]:
    """Return (total hits, top-`retmax` PMIDs by relevance)."""
    with Entrez.esearch(db="pubmed", term=build_query(year, stratum),
                        retmax=retmax, sort="relevance") as handle:
        res = Entrez.read(handle)
    return int(res["Count"]), list(res["IdList"])


def fetch_records(pmids: list[str]) -> list[dict]:
    """Fetch full MEDLINE records (title, abstract, MeSH, pub types, ...)."""
    if not pmids:
        return []
    with Entrez.efetch(db="pubmed", id=",".join(pmids),
                       rettype="medline", retmode="text") as handle:
        return [dict(r) for r in Medline.parse(handle)]


def collect(per_year: dict[str, int] = config.PUBMED_PER_YEAR,
            refresh: bool = False) -> list[dict]:
    """Run the stratified search and save raw records.

    Re-uses the saved file unless `refresh=True`, so the notebook can be
    re-run without hitting the API again.
    """
    if config.PUBMED_RAW_FILE.exists() and not refresh:
        records = json.loads(config.PUBMED_RAW_FILE.read_text())
        print(f"Loaded {len(records)} cached PubMed records")
        return records

    records, log = [], []
    for year in config.PUBMED_YEARS:
        for stratum, n in per_year.items():
            total, pmids = search(year, stratum, n)
            batch = fetch_records(pmids)
            for r in batch:
                r["_search_year"], r["_stratum"] = year, stratum
            records.extend(batch)
            log.append({"year": year, "stratum": stratum,
                        "matching": total, "retrieved": len(batch)})
            time.sleep(0.4)  # stay under NCBI's 3 requests/s without a key

    config.PUBMED_RAW_FILE.write_text(json.dumps(records, ensure_ascii=False))
    print(pd.DataFrame(log).to_string(index=False))
    print(f"Saved {len(records)} raw records -> {config.PUBMED_RAW_FILE.relative_to(config.PROJECT_ROOT)}")
    return records


def _doi(record: dict) -> str | None:
    for aid in record.get("AID", []):
        if aid.endswith("[doi]"):
            return aid.split(" ")[0]
    return None


def _year(record: dict) -> int | None:
    m = re.search(r"\d{4}", record.get("DP", ""))
    return int(m.group()) if m else None


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Flatten raw MEDLINE records into one tidy row per article."""
    rows = [{
        "pmid": r.get("PMID"),
        "title": r.get("TI"),
        "abstract": r.get("AB"),
        "journal": r.get("JT"),
        "publication_year": r.get("_search_year") or _year(r),
        "stratum": r.get("_stratum"),
        "publication_types": r.get("PT", []),
        "mesh_terms": r.get("MH", []),
        "authors": r.get("AU", []),
        "doi": _doi(r),
    } for r in records]

    df = pd.DataFrame(rows).drop_duplicates("pmid")
    df = df[df["abstract"].fillna("").str.strip().ne("")]
    df["study_design"] = df["publication_types"].apply(classify_design)
    return df.reset_index(drop=True)


def classify_design(pub_types: list[str]) -> str:
    """Map PubMed publication types to one evidence tier (highest wins)."""
    pt = {p.lower() for p in pub_types}
    if "meta-analysis" in pt or "systematic review" in pt:
        return "Meta-analysis / systematic review"
    if "clinical trial, phase iii" in pt:
        return "Phase 3 trial"
    if "clinical trial, phase ii" in pt:
        return "Phase 2 trial"
    if "randomized controlled trial" in pt:
        return "Randomized controlled trial"
    if any("clinical trial" in p for p in pt):
        return "Other clinical trial"
    return "Other"
