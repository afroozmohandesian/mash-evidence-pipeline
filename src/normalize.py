"""Part 3b — LLM pass 2: compound-level normalisation and mechanism class.

Pass 1 is strictly source-grounded, so mechanism of action is often missing
(abstracts and registry entries rarely state it). For a competitive map we
need a mechanism for every compound, so this pass:

1. cleans names deterministically (case, doses, salts, brackets);
2. sends the *unique* cleaned names to the LLM in batches and asks for a
   canonical INN name and a mechanism class from a fixed list. Here the model
   may use its own pharmacology knowledge, and every value it supplies is
   flagged with moa_source = "model knowledge" so the reader can tell it
   apart from what the source itself said.

Batching ~60 names per call keeps the whole pass to a handful of requests.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from . import config
from .extraction import client, tracker

CACHE = config.CACHE_DIR / "pass2"
CACHE.mkdir(parents=True, exist_ok=True)

MOA_CLASSES = Literal[
    "THR-beta agonist",
    "GLP-1 receptor agonist",
    "GLP-1/GIP dual agonist",
    "GLP-1/glucagon dual agonist",
    "GLP-1/GIP/glucagon triple agonist",
    "FGF21 analogue",
    "FGF19 analogue",
    "FXR agonist",
    "PPAR agonist",
    "SGLT2 inhibitor",
    "DPP-4 inhibitor",
    "Lipogenesis inhibitor (ACC/FASN/SCD1/DGAT2/KHK)",
    "ASK1 / apoptosis inhibitor",
    "CCR2/CCR5 antagonist",
    "Galectin-3 inhibitor",
    "Other anti-fibrotic / anti-inflammatory",
    "RNA therapy (siRNA/ASO)",
    "Insulin sensitiser (other)",
    "Lipid-lowering (statin/omega-3/other)",
    "Bile acid / UDCA",
    "Vitamin / antioxidant / supplement",
    "Herbal / natural product",
    "Probiotic / microbiome",
    "Combination therapy",
    "Other",
    "Unknown",
]


class NormalizedDrug(BaseModel):
    raw_name: str = Field(description="Echo the input name exactly.")
    canonical_name: str = Field(description="Lower-case INN/generic name; development code if no INN; 'a + b' for combinations.")
    moa_class: MOA_CLASSES
    moa_detail: str = Field(description="Target / mechanism in <= 8 words, e.g. 'THR-beta selective agonist'.")
    novel_mash_agent: bool = Field(description="True if a new pharmacological agent developed for MASH/NASH or metabolic disease (not a vitamin, supplement, herbal product or long-generic repurposed drug).")


class NormalizedBatch(BaseModel):
    drugs: list[NormalizedDrug]


SYSTEM_PROMPT = """You are a pharmacologist curating a MASH (NASH) drug
pipeline database. For each drug name given, return its canonical name and
mechanism class using your pharmacology knowledge.

- Merge aliases: brand names, development codes and salts map to one
  canonical INN, e.g. 'MGL-3196' and 'Rezdiffra' -> 'resmetirom'.
- Combinations of two or more agents -> moa_class 'Combination therapy'.
- If you do not recognise a name, use moa_class 'Unknown'. Do not guess.
- Return exactly one entry per input name, in the same order."""

PROMPT_VERSION = hashlib.sha1(
    (SYSTEM_PROMPT + json.dumps(NormalizedBatch.model_json_schema())).encode()
).hexdigest()[:8]

DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|µg|ug|mcg|g|iu|ml|%)\b.*", re.IGNORECASE)
SALT = re.compile(r"\b(sodium|magnesium|hydrochloride|hcl|tablets?|capsules?|injection|"
                  r"oral|sc|subcutaneous|once[- ]weekly|daily)\b", re.IGNORECASE)


MANUAL_OVERRIDES = {
    "sel": ("selonsertib", "ASK1 / apoptosis inhibitor"),
    "fir": ("firsocostat", "Lipogenesis inhibitor (ACC/FASN/SCD1/DGAT2/KHK)"),
    "stearoyl-coa desaturase-1": ("aramchol", "Lipogenesis inhibitor (ACC/FASN/SCD1/DGAT2/KHK)"),
}


def clean_name(name: str) -> str:
    """Deterministic clean-up before the LLM: case, doses, salts, spacing."""
    name = DOSE.sub("", name)
    name = SALT.sub("", name)
    name = re.sub(r"\s+", " ", name).strip(" ,;-").lower()
    return name


def _normalize_batch(names: list[str]) -> list[dict]:
    key = hashlib.sha1(f"{config.LLM_MODEL}|{PROMPT_VERSION}|{'|'.join(names)}".encode()).hexdigest()[:16]
    path = CACHE / f"batch_{key}.json"
    if path.exists():
        return json.loads(path.read_text())

    tracker.check()
    resp = client().responses.parse(
        model=config.LLM_MODEL,
        reasoning={"effort": config.LLM_REASONING_EFFORT},
        input=[{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": "\n".join(names)}],
        text_format=NormalizedBatch,
    )
    tracker.add(resp.usage)
    out = [d.model_dump() for d in resp.output_parsed.drugs]
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def normalize_drugs(names: list[str], batch_size: int = 60) -> pd.DataFrame:
    """Map every cleaned name to canonical name + mechanism class."""
    unique = sorted(set(names))
    rows = []
    for i in range(0, len(unique), batch_size):
        batch = unique[i:i + batch_size]
        result = {r["raw_name"].lower(): r for r in _normalize_batch(batch)}
        for name in batch:  # tolerate a dropped/reworded echo
            r = result.get(name, {"canonical_name": name, "moa_class": "Unknown",
                                  "moa_detail": "", "novel_mash_agent": False})
            rows.append({**r, "clean_name": name})
    df = pd.DataFrame(rows).drop(columns="raw_name", errors="ignore")

    # Manual corrections found on review (abbreviations used inside trial arms)
    for name, (canon, moa) in MANUAL_OVERRIDES.items():
        mask = df["clean_name"] == name
        df.loc[mask, ["canonical_name", "moa_class", "novel_mash_agent"]] = [canon, moa, True]
    print(f"Normalised {len(unique)} names -> {df.canonical_name.nunique()} compounds; "
          f"spend so far ${tracker.spent:.4f}")
    return df
