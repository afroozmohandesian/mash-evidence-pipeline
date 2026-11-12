"""Central configuration: paths, search windows, sample sizes and model settings.

Every other module imports from here, so a change of scope (e.g. a different
date window) is made in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------- directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"          # LLM responses, keyed by input hash
OUTPUT_DIR = PROJECT_ROOT / "outputs"   # charts, deck, final CSV
FIGURES_DIR = OUTPUT_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, OUTPUT_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- raw outputs
PUBMED_RAW_FILE = RAW_DIR / "pubmed_data.json"
TRIALS_RAW_FILE = RAW_DIR / "trials_data.json"

# ---------------------------------------------------------------- PubMed
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "mash.pipeline@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")  # optional, raises the rate limit
PUBMED_YEARS = range(2020, 2026)          # 2020–2025 inclusive
# Stratified sample per publication year: primary trials weigh more than
# secondary evidence for a competitive landscape. 6 years x 25 = up to 150.
PUBMED_PER_YEAR = {"primary_trial": 17, "meta_analysis": 8}

# ---------------------------------------------------------------- ClinicalTrials.gov
CTGOV_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_PAGE_SIZE = 100

# ---------------------------------------------------------------- LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-6-luna")
LLM_REASONING_EFFORT = "low"

# USD per 1M tokens (input, output). Reasoning tokens are billed as output.
LLM_PRICES = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6.1-sol": (2.00, 10.00),
}
# Hard spending cap for this run: new API calls stop once it is reached.
# Cached responses cost nothing and are always returned.
LLM_BUDGET_USD = float(os.getenv("LLM_BUDGET_USD", "0.60"))
