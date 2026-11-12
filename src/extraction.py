"""Part 3a — LLM extraction, pass 1: one structured record per source.

Design choices
--------------
* The LLM only extracts what is buried in free text (drug names and their
  role, endpoint, efficacy, safety). Phase, sponsor, status and source IDs
  are attached afterwards from the structured metadata.
* Categorical fields are `Literal` enums, so the model cannot invent new
  spellings and no regex normalisation is needed later.
* Drugs carry a `role`, which separates the investigational compound from
  comparators, background therapy and non-drug interventions. Without this,
  vitamin E and metformin would show up as "competitors".
* A relevance flag lets the model drop off-topic records that slipped past
  the keyword search.
* Every response is cached on disk under a hash of (model, prompt, input),
  so re-running the notebook is free and deterministic. Changing the prompt
  invalidates the cache automatically.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from . import config

CACHE = config.CACHE_DIR / "pass1"
CACHE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- schema
class DrugMention(BaseModel):
    name: str = Field(description="Drug name exactly as written, incl. code if given, e.g. 'resmetirom (MGL-3196)'. No doses.")
    role: Literal["investigational", "active comparator", "background therapy", "non-drug intervention"] = Field(
        description="investigational = the compound being tested for MASH/NAFLD in this source.")


class RecordExtraction(BaseModel):
    relevant: bool = Field(description="True only if the source studies a treatment for MASH/NASH or MASLD/NAFLD.")
    drugs: list[DrugMention]
    mechanism_of_action: str | None = Field(description="Mechanism as stated in the source; null if not stated.")
    primary_endpoint: str | None = Field(description="Primary endpoint incl. timepoint, concise.")
    efficacy_outcome: str | None = Field(description="Main efficacy result with numbers and comparator; null if no results reported.")
    efficacy_direction: Literal["positive", "negative", "mixed", "not reported"]
    safety_signal: str | None = Field(description="Key adverse events or safety findings; null if not reported.")
    evidence_type: Literal["RCT", "single-arm trial", "meta-analysis / systematic review",
                           "registry record", "other"]
    study_phase: Literal["Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "not reported"]


SYSTEM_PROMPT = """You are a biomedical evidence analyst building a competitive
intelligence dataset on MASH (formerly NASH) treatments.

Extract the requested fields from ONE source record (a PubMed abstract or a
ClinicalTrials.gov registry entry).

Rules:
- Use only information stated in the source. Never add outside knowledge.
- drugs: list every pharmacological intervention and tag its role. Placebo is
  not a drug. Lifestyle, diet, surgery, devices -> 'non-drug intervention'.
- For a meta-analysis, the investigational drugs are the ones being pooled.
- efficacy_outcome and safety_signal: only actual results (numbers, p-values,
  event rates). A planned outcome is NOT a result -> null.
- efficacy_direction, judged on the PRIMARY endpoint:
    positive = met with statistical significance (e.g. p<0.05 vs control);
    negative = not met / no significant difference (e.g. p>=0.05);
    mixed = met for some doses or co-primary endpoints but not others;
    not reported = no results in the source.
- For registry records, results appear under RESULTS SUMMARY if posted.
- Keep free-text fields short (max ~2 sentences) and quantitative."""

PROMPT_VERSION = hashlib.sha1(
    (SYSTEM_PROMPT + json.dumps(RecordExtraction.model_json_schema())).encode()
).hexdigest()[:8]


# ---------------------------------------------------------------- inputs
def pubmed_input(row: pd.Series) -> str:
    return (f"SOURCE: PubMed (PMID {row.pmid})\n"
            f"TITLE: {row.title}\nJOURNAL: {row.journal} ({row.publication_year})\n"
            f"PUBLICATION TYPES: {', '.join(row.publication_types)}\n\n"
            f"ABSTRACT:\n{row.abstract}")


def trial_input(row: pd.Series) -> str:
    interventions = "\n".join(
        f"- {i['name']} [{i['type']}]: {i['description']}" for i in row.interventions)
    outcomes = "\n".join(f"- {o}" for o in row.primary_outcomes)
    return (f"SOURCE: ClinicalTrials.gov ({row.nct_id})\n"
            f"TITLE: {row.title}\nOFFICIAL TITLE: {row.official_title}\n"
            f"REGISTERED PHASE: {row.phase}\nCONDITIONS: {', '.join(row.conditions)}\n\n"
            f"INTERVENTIONS:\n{interventions}\n\nPRIMARY OUTCOMES:\n{outcomes}\n\n"
            f"RESULTS SUMMARY:\n{_text_or(row.results_summary, 'No results posted.')}")


def _text_or(value, default: str) -> str:
    """pandas turns missing strings into NaN (truthy), so test explicitly."""
    return value if isinstance(value, str) and value.strip() else default


def build_inputs(pubmed_df: pd.DataFrame, trials_df: pd.DataFrame) -> list[dict]:
    """One dict per source record: id, source and the text sent to the LLM."""
    items = [{"data_source": "PubMed", "source_id": str(r.pmid), "text": pubmed_input(r)}
             for r in pubmed_df.itertuples()]
    items += [{"data_source": "ClinicalTrials.gov", "source_id": r.nct_id, "text": trial_input(r)}
              for r in trials_df[trials_df.disease_scope != "Other"].itertuples()]
    return items


# ---------------------------------------------------------------- LLM call
_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _cache_path(item: dict):
    key = hashlib.sha1(f"{config.LLM_MODEL}|{PROMPT_VERSION}|{item['text']}".encode()).hexdigest()[:16]
    return CACHE / f"{item['source_id']}_{key}.json"


class BudgetExceeded(RuntimeError):
    pass


class CostTracker:
    """Thread-safe running total of API spend, with a hard cap."""

    def __init__(self, budget_usd: float):
        self.budget, self.spent, self.calls = budget_usd, 0.0, 0
        self._lock = threading.Lock()

    def check(self):
        with self._lock:
            if self.spent >= self.budget:
                raise BudgetExceeded(f"budget ${self.budget:.2f} reached")

    def add(self, usage) -> float:
        price_in, price_out = config.LLM_PRICES[config.LLM_MODEL]
        cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6
        with self._lock:
            self.spent += cost
            self.calls += 1
        return cost


tracker = CostTracker(config.LLM_BUDGET_USD)


def extract_one(item: dict, retries: int = 3) -> dict:
    """Extract one record, using the disk cache when available."""
    path = _cache_path(item)
    if path.exists():
        return json.loads(path.read_text())

    for attempt in range(retries):
        tracker.check()
        try:
            resp = client().responses.parse(
                model=config.LLM_MODEL,
                reasoning={"effort": config.LLM_REASONING_EFFORT},
                input=[{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": item["text"]}],
                text_format=RecordExtraction,
            )
            cost = tracker.add(resp.usage)
            parsed = resp.output_parsed
            if parsed is None:
                raise ValueError("empty structured output")
            result = {"data_source": item["data_source"], "source_id": item["source_id"],
                      **parsed.model_dump(),
                      "_usage": {"model": config.LLM_MODEL,
                                 "input_tokens": resp.usage.input_tokens,
                                 "output_tokens": resp.usage.output_tokens,
                                 "cost_usd": round(cost, 6)}}
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            return result
        except BudgetExceeded:
            raise
        except Exception as err:  # rate limits, timeouts, parse failures
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 5)
            print(f"retry {item['source_id']}: {err}")


def extract_all(items: list[dict], workers: int = 8) -> tuple[pd.DataFrame, list[dict]]:
    """Run pass 1 over all items in parallel. Returns (results, errors)."""
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, it): it for it in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM pass 1"):
            item = futures[fut]
            try:
                results.append(fut.result())
            except Exception as err:
                errors.append({"source_id": item["source_id"], "error": str(err)})
    print(f"Extracted {len(results)} records, {len(errors)} errors "
          f"(model {config.LLM_MODEL}, prompt {PROMPT_VERSION})")
    print(f"New API calls this run: {tracker.calls}, spend ${tracker.spent:.4f} "
          f"of ${tracker.budget:.2f} budget")
    return pd.DataFrame(results), errors


def cached_count(items: list[dict]) -> int:
    return sum(_cache_path(it).exists() for it in items)
