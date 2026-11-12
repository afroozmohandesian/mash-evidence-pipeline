"""Part 4a — Summary tables and charts for the findings deck.

Charts are static PNGs (they go into a PowerPoint deck). Conventions:
one hue for single-series charts, grey for "inactive", recessive axes,
direct value labels instead of gridlines, no dual axes.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config

BLUE, BLUE_LIGHT = "#2a78d6", "#86b6ef"
GREY, INK, INK_2, MUTED, AXIS = "#c3c2b7", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PHASE_ORDER = ["Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3"]
MAP_COLUMNS = {"Phase 1/2": "Phase 1/2", "Phase 2": "Phase 2",
               "Phase 2/3": "Phase 2/3 – 3", "Phase 3": "Phase 2/3 – 3"}
OTHER_LABEL = "All other classes"


def _save(fig, name: str):
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def hbar(series: pd.Series, name: str, xlabel: str = "", color=BLUE,
         width: float = 6.0, height: float | None = None, title: str | None = None):
    """Horizontal bar chart, largest on top, values labelled at bar ends.

    A folded remainder bucket (OTHER_LABEL) is always drawn last and in grey,
    so it never reads as the leading category.
    """
    rest = series[series.index == OTHER_LABEL]
    s = pd.concat([rest, series[series.index != OTHER_LABEL].sort_values()])
    colors = [GREY if i == OTHER_LABEL else color for i in s.index]
    fig, ax = plt.subplots(figsize=(width, height or 0.38 * len(s) + 0.8))
    bars = ax.barh(s.index.astype(str), s.values, color=colors, height=0.62)
    for b, v in zip(bars, s.values):
        ax.text(b.get_width() + s.max() * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, s.max() * 1.15)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    return _save(fig, name)


def vbar(series: pd.Series, name: str, color=BLUE, width=5.0, height=3.2, title=None):
    fig, ax = plt.subplots(figsize=(width, height))
    bars = ax.bar(series.index.astype(str), series.values, color=color, width=0.62)
    for b, v in zip(bars, series.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + series.max() * 0.02,
                f"{v:,}", ha="center", color=INK, fontsize=10)
    ax.set_ylim(0, series.max() * 1.15)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    return _save(fig, name)


# ---------------------------------------------------------------- tables
def core_trials(trials_df: pd.DataFrame, extraction_df: pd.DataFrame) -> pd.DataFrame:
    """MASH/NASH trials the LLM confirmed as relevant."""
    relevant = set(extraction_df.loc[extraction_df["relevant"]
                                     & extraction_df["data_source"].eq("ClinicalTrials.gov"),
                                     "source_id"])
    return trials_df[trials_df["nct_id"].isin(relevant)
                     & trials_df["disease_scope"].eq("MASH/NASH")].copy()


def trial_moa(landscape: pd.DataFrame, trial_ids) -> pd.DataFrame:
    """Unique (trial, mechanism class) pairs for novel agents."""
    d = landscape[landscape["source_id"].isin(trial_ids) & landscape["novel_mash_agent"].fillna(False)]
    return d[["source_id", "moa_class", "status_group", "trial_phase"]].drop_duplicates(["source_id", "moa_class"])


def top_n(s: pd.Series, n: int, other=OTHER_LABEL) -> pd.Series:
    """Keep the n largest named classes; fold the rest (incl. 'Other') into one bucket."""
    s = s.sort_values(ascending=False)
    if "Other" in s.index:
        s = pd.concat([s.drop("Other"), s[["Other"]]])
    if len(s) <= n:
        return s
    return pd.concat([s.iloc[:n], pd.Series({other: s.iloc[n:].sum()})])


# ---------------------------------------------------------------- charts
def evidence_charts(pubmed_df: pd.DataFrame, landscape: pd.DataFrame) -> dict:
    lit = landscape[landscape["data_source"] == "PubMed"]
    out = {
        "pub_design": hbar(pubmed_df["study_design"].value_counts(), "pub_design",
                           width=5.2, title="Articles by study design"),
        "pub_year": vbar(pubmed_df["publication_year"].value_counts().sort_index(), "pub_year",
                         width=4.6, height=2.6, title="Articles by publication year"),
        "pub_moa": hbar(top_n(lit[lit["novel_mash_agent"].fillna(False)]
                              .drop_duplicates(["source_id", "moa_class"])["moa_class"]
                              .value_counts(), 7), "pub_moa", width=5.6,
                        title="Mechanisms studied in the literature (articles)"),
    }
    return out


def trial_charts(core: pd.DataFrame, landscape: pd.DataFrame) -> dict:
    tm = trial_moa(landscape, core["nct_id"])
    moa_counts = top_n(tm["moa_class"].value_counts(), 9)
    status = core["status_group"].value_counts().reindex(
        ["Completed", "Ongoing", "Stopped early", "Unknown"]).fillna(0).astype(int)
    return {
        "trial_phase": vbar(core["phase"].value_counts().reindex(PHASE_ORDER).fillna(0).astype(int),
                            "trial_phase", width=3.8, height=2.4, title="By phase"),
        "trial_sponsor": vbar(core["sponsor_type"].value_counts()
                              .rename(index={"Academic / hospital": "Academic"}),
                              "trial_sponsor", width=3.8, height=2.4, title="By lead sponsor"),
        "trial_status": vbar(status.rename(index={"Stopped early": "Stopped"}), "trial_status",
                             width=3.8, height=2.4, title="By status"),
        "trial_moa": hbar(moa_counts, "trial_moa", width=5.8,
                          title="Trials by mechanism class (novel agents)"),
    }


def competitive_map(compounds: pd.DataFrame, max_rows: int = 12, max_per_cell: int = 6) -> tuple:
    """Pipeline grid: mechanism class (rows) x furthest phase (columns).

    Compounds with an ongoing trial are blue and bold; programmes without an
    ongoing registry trial are grey. Within a cell, active compounds come
    first, then by number of trials.
    """
    c = compounds[compounds["novel_mash_agent"].fillna(False)
                  & compounds["highest_phase"].isin(PHASE_ORDER)
                  & ~compounds["moa_class"].isin(["Unknown", "Other", "Combination therapy"])].copy()
    c["active"] = c["n_ongoing_industry_trials"] > 0
    c["column"] = c["highest_phase"].map(MAP_COLUMNS)
    c = c.sort_values(["active", "n_trials"], ascending=False)
    columns = list(dict.fromkeys(MAP_COLUMNS.values()))

    # rows: classes ordered by most advanced phase, then activity, then size
    score = c.groupby("moa_class").agg(phase=("highest_phase_rank", "max"),
                                       active=("active", "sum"), trials=("n_trials", "sum"))
    rows = score.sort_values(["phase", "active", "trials"], ascending=False).index[:max_rows].tolist()

    def short(name: str) -> str:
        name = name.removesuffix(" alfa")
        return name if len(name) <= 14 else name[:13] + "…"

    fig, ax = plt.subplots(figsize=(16, 0.55 * len(rows) + 1.0))
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(rows), 0)
    for i, col in enumerate(columns):
        ax.text(i + 0.5, -0.2, col, ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=INK)
        if i:
            ax.axvline(i, color="#e1e0d9", lw=1)
    for j, moa in enumerate(rows):
        ax.axhline(j, color="#e1e0d9", lw=0.8)
        ax.text(-0.03, j + 0.5, moa, ha="right", va="center", fontsize=10.5, color=INK)
        for i, col in enumerate(columns):
            cell = c[(c["moa_class"] == moa) & (c["column"] == col)].head(max_per_cell)
            names = list(cell.itertuples())
            for k, r in enumerate(names):
                x = i + 0.04 + (k % 3) * 0.32
                y = j + 0.3 + (k // 3) * 0.4
                has_neighbour = k % 3 < 2 and k + 1 < len(names)
                label = short(r.drug_name) if has_neighbour else r.drug_name.removesuffix(" alfa")
                ax.text(x, y, label, fontsize=9.5, va="center",
                        color=BLUE if r.active else MUTED,
                        fontweight="bold" if r.active else "normal")
    ax.axis("off")
    ax.text(0, len(rows) + 0.35, "●  Active programme (≥1 ongoing industry trial)", color=BLUE,
            fontsize=10, fontweight="bold", va="top")
    ax.text(1.5, len(rows) + 0.35, "●  No ongoing industry trial in registry", color=MUTED,
            fontsize=10, va="top")
    return _save(fig, "competitive_map"), c[c["moa_class"].isin(rows)]


def momentum_chart(core: pd.DataFrame, landscape: pd.DataFrame):
    """Ongoing vs completed/stopped trials per mechanism class (small multiples, one axis)."""
    tm = trial_moa(landscape, core["nct_id"])
    tm = tm[tm["moa_class"] != "Other"]
    keep = tm["moa_class"].value_counts().head(10).index
    tm = tm[tm["moa_class"].isin(keep)]
    tab = (pd.crosstab(tm["moa_class"], tm["status_group"])
           .reindex(columns=["Ongoing", "Completed", "Stopped early", "Unknown"], fill_value=0))
    tab["Past"] = tab[["Completed", "Stopped early", "Unknown"]].sum(axis=1)
    tab["share_ongoing"] = tab["Ongoing"] / (tab["Ongoing"] + tab["Past"])
    tab = tab.sort_values("share_ongoing")

    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(tab) + 0.9))
    ax.barh(tab.index, tab["Past"], color=GREY, height=0.62, label="Completed / stopped")
    ax.barh(tab.index, tab["Ongoing"], left=tab["Past"] + tab["Past"].max() * 0.01,
            color=BLUE, height=0.62, label="Ongoing")
    for y, (p, o) in enumerate(zip(tab["Past"], tab["Ongoing"])):
        ax.text(p + o + tab["Past"].max() * 0.03, y, f"{o} ongoing", va="center",
                fontsize=9.5, color=INK)
    ax.set_xlim(0, (tab["Past"] + tab["Ongoing"]).max() * 1.3)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title("Momentum: ongoing vs past trials by mechanism", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    return _save(fig, "momentum"), tab


# ---------------------------------------------------------------- deck facts
GRAVEYARD = ["FXR agonist", "ASK1 / apoptosis inhibitor", "CCR2/CCR5 antagonist",
             "Lipogenesis inhibitor (ACC/FASN/SCD1/DGAT2/KHK)"]


def compute_facts(pubmed_df, trials_df, extraction_df, drug_map, landscape, compounds,
                  qa: pd.DataFrame, manual_accuracy: float) -> dict:
    """Every number quoted in the deck, computed from the pipeline outputs."""
    core = core_trials(trials_df, extraction_df)
    tm = trial_moa(landscape, core["nct_id"])
    novel = landscape[landscape["novel_mash_agent"].fillna(False)]
    text = (core["title"].fillna("") + " " + core["official_title"].fillna("") + " "
            + core["conditions"].astype(str)).str.lower()
    cirrhosis = text.str.contains("cirrho")
    ongoing = core["status_group"].eq("Ongoing")

    pub = extraction_df[(extraction_df["data_source"] == "PubMed") & extraction_df["relevant"]
                        & extraction_df["efficacy_direction"].ne("not reported")]
    fgf21 = compounds[(compounds["moa_class"] == "FGF21 analogue")
                      & (compounds["highest_phase"] == "Phase 3")]["drug_name"]
    dual = novel[(novel["moa_class"] == "GLP-1/glucagon dual agonist")
                 & novel["efficacy_direction"].ne("not reported")].drop_duplicates("source_id")
    grave = tm[tm["moa_class"].isin(GRAVEYARD)]
    per_trial = novel[novel["source_id"].isin(core["nct_id"])].groupby("source_id")["drug_name"].nunique()
    usage = extraction_df["_usage"].dropna()
    q = qa.set_index("check")["result"]

    return {
        "run_date": pd.Timestamp.today().strftime("%d %b %Y"),
        "model": config.LLM_MODEL,
        "cost": sum(u["cost_usd"] for u in usage),
        "n_articles": len(pubmed_df),
        "n_primary": int((pubmed_df["stratum"] == "primary_trial").sum()),
        "n_meta": int((pubmed_df["stratum"] == "meta_analysis").sum()),
        "pub_positive": f"{(pub['efficacy_direction'] == 'positive').mean():.0%}",
        "n_trials_raw": len(trials_df),
        "n_relevant": int(extraction_df["relevant"].sum()),
        "n_raw_names": drug_map["clean_name"].nunique(),
        "n_compounds": landscape["drug_name"].nunique(),
        "qa_phase": q["Trial phase matches registry"],
        "qa_capture": q["Investigational drug captured when registry lists one"],
        "qa_invented": q["Efficacy invented for trials without posted results"],
        "qa_invented_base": int((trials_df["results_summary"].isna()
                                                   & trials_df["disease_scope"].ne("Other")).sum()),
        "qa_direction": q["Efficacy direction consistent with reported p-values"],
        "qa_manual": f"{manual_accuracy:.0%}",
        "n_core": len(core),
        "share_phase2": f"{core['highest_phase'].le(2).mean():.0%}",
        "n_phase3": int(core["highest_phase"].eq(3).sum()),
        "share_industry": f"{core['sponsor_type'].eq('Industry').mean():.0%}",
        "n_since_2020": int(core["start_year"].ge(2020).sum()),
        "n_ongoing": int(ongoing.sum()),
        "fgf21_names": ", ".join(fgf21.str.removesuffix(" alfa")),
        "fgf21_ongoing": int(tm[(tm["moa_class"] == "FGF21 analogue") & tm["status_group"].eq("Ongoing")].shape[0]),
        "dual_pos": f"{(dual['efficacy_direction'] == 'positive').sum()} of {len(dual)} studies with results",
        "graveyard_past": int(grave["status_group"].ne("Ongoing").sum()),
        "graveyard_ongoing": int(grave["status_group"].eq("Ongoing").sum()),
        "cirr_ongoing": int((cirrhosis & ongoing).sum()),
        "n_combo": int((per_trial >= 2).sum()),
        "n_trials_novel": len(per_trial),
    }
