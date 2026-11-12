"""Part 3c — Assemble the landscape dataset (deliverable #4) and the
compound-level table that drives the competitive map.

landscape_data.csv : one row per (investigational compound x source record).
compounds.csv      : one row per compound, with its furthest phase, mechanism
                     class, lead sponsor and evidence counts.

Structured fields (phase, sponsor, status) come from the registry metadata;
only free-text fields come from the LLM.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .normalize import clean_name

PHASE_RANK = {"Phase 1": 1, "Phase 1/2": 1.5, "Phase 2": 2, "Phase 2/3": 2.5,
              "Phase 3": 3, "Phase 4": 4}


def explode_drugs(extraction_df: pd.DataFrame) -> pd.DataFrame:
    """One row per investigational drug mention in a relevant record."""
    df = extraction_df[extraction_df["relevant"]].copy()
    df = df.explode("drugs").dropna(subset=["drugs"])
    df["drug_name_raw"] = df["drugs"].str["name"]
    df["drug_role"] = df["drugs"].str["role"]
    df = df[df["drug_role"] == "investigational"].drop(columns="drugs")
    df["clean_name"] = df["drug_name_raw"].map(clean_name)
    return df[df["clean_name"] != ""]


def build_landscape(extraction_df: pd.DataFrame, drug_map: pd.DataFrame,
                    pubmed_df: pd.DataFrame, trials_df: pd.DataFrame) -> pd.DataFrame:
    df = explode_drugs(extraction_df).merge(drug_map, on="clean_name", how="left")

    # mechanism: prefer what the source states, fall back to model knowledge
    stated = df["mechanism_of_action"].fillna("").str.strip().ne("")
    df["moa_source"] = stated.map({True: "source-reported", False: "model knowledge"})
    df["mechanism_of_action"] = df["mechanism_of_action"].where(stated, df["moa_detail"])

    # structured metadata from the registry / PubMed tables
    trial_meta = trials_df[["nct_id", "phase", "highest_phase", "status_group",
                            "sponsor_name", "sponsor_type", "start_year",
                            "enrollment", "disease_scope"]].rename(columns={"nct_id": "source_id"})
    pub_meta = pubmed_df[["pmid", "publication_year", "journal", "study_design"]].rename(
        columns={"pmid": "source_id"})
    pub_meta["source_id"] = pub_meta["source_id"].astype(str)
    df = df.merge(trial_meta, on="source_id", how="left").merge(pub_meta, on="source_id", how="left")

    is_trial = df["data_source"].eq("ClinicalTrials.gov")
    df["trial_phase"] = df["phase"].where(is_trial, df["study_phase"])
    # evidence type of a registry entry is known, not something to extract
    df["evidence_type"] = df["evidence_type"].where(~is_trial, "registry record")
    df["sponsor_type"] = df["sponsor_type"].where(is_trial, "Not reported (literature)")
    df["year"] = df["start_year"].where(is_trial, df["publication_year"])

    columns = [
        "canonical_name", "drug_name_raw", "moa_class", "mechanism_of_action", "moa_source",
        "novel_mash_agent", "trial_phase", "primary_endpoint", "efficacy_outcome",
        "efficacy_direction", "safety_signal", "sponsor_name", "sponsor_type",
        "status_group", "evidence_type", "disease_scope", "year", "enrollment",
        "data_source", "source_id",
    ]
    out = (df[columns].rename(columns={"canonical_name": "drug_name"})
           .drop_duplicates(["drug_name", "source_id"])
           .sort_values(["drug_name", "data_source", "source_id"])
           .reset_index(drop=True))
    out.to_csv(config.OUTPUT_DIR / "landscape_data.csv", index=False)
    print(f"landscape_data.csv: {len(out)} rows, {out.drug_name.nunique()} compounds")
    return out


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if not s.empty else None


def build_compounds(landscape: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per compound for the competitive map."""
    df = landscape.copy()
    df["phase_rank"] = df["trial_phase"].map(PHASE_RANK)
    trials = df[df["data_source"] == "ClinicalTrials.gov"]
    industry = trials[trials["sponsor_type"] == "Industry"]

    g = df.groupby("drug_name")
    comp = pd.DataFrame({
        "moa_class": g["moa_class"].agg(_mode),
        "moa_detail": g["mechanism_of_action"].agg(_mode),
        "novel_mash_agent": g["novel_mash_agent"].agg(_mode),
        "highest_phase_rank": g["phase_rank"].max(),
        "n_trials": trials.groupby("drug_name")["source_id"].nunique(),
        "n_publications": df[df["data_source"] == "PubMed"].groupby("drug_name")["source_id"].nunique(),
        "n_ongoing_trials": trials[trials["status_group"] == "Ongoing"].groupby("drug_name")["source_id"].nunique(),
        # a programme counts as commercially active only if industry is still running a trial
        "n_ongoing_industry_trials": industry[industry["status_group"] == "Ongoing"]
        .groupby("drug_name")["source_id"].nunique(),
        "lead_sponsor": industry.groupby("drug_name")["sponsor_name"].agg(_mode),
        "main_sponsor_type": trials.groupby("drug_name")["sponsor_type"].agg(_mode),
        "any_positive_result": g["efficacy_direction"].agg(lambda s: (s == "positive").any()),
    }).reset_index()

    rank_to_phase = {v: k for k, v in PHASE_RANK.items()}
    comp["highest_phase"] = comp["highest_phase_rank"].map(rank_to_phase).fillna("Not reported")
    for c in ("n_trials", "n_publications", "n_ongoing_trials", "n_ongoing_industry_trials"):
        comp[c] = comp[c].fillna(0).astype(int)

    comp = comp.sort_values(["highest_phase_rank", "n_trials"], ascending=False)
    comp.to_csv(config.PROCESSED_DIR / "compounds.csv", index=False)
    print(f"compounds.csv: {len(comp)} compounds, "
          f"{comp.novel_mash_agent.fillna(False).sum()} novel MASH agents")
    return comp
