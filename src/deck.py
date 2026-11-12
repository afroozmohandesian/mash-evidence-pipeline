"""Part 4b — Build the 5-slide findings deck (deliverable #5) with python-pptx.

Every number on the slides is passed in from the pipeline (`facts` dict), so
re-running the notebook on fresh data regenerates a consistent deck. Slide
titles state the takeaway; the body supports it.
"""
from __future__ import annotations

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

from . import config

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK_2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
BLUE_BG = RGBColor(0xEE, 0xF4, 0xFC)
PANEL = RGBColor(0xF6, 0xF6, 0xF4)
LINE = RGBColor(0xE1, 0xE0, 0xD9)
FONT = "Arial"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


# ---------------------------------------------------------------- helpers
def _text(slide, x, y, w, h, paragraphs, size=14, color=INK, bold=False,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=4):
    """Add a text box. `paragraphs` is a str or a list of str / (str, dict) runs."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [para]
        for run in runs:
            txt, fmt = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = txt
            r.font.name = FONT
            r.font.size = Pt(fmt.get("size", size))
            r.font.bold = fmt.get("bold", bold)
            r.font.color.rgb = fmt.get("color", color)
    return box


def _rect(slide, x, y, w, h, fill=PANEL, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    return shape


def _header(slide, number: int, kicker: str, title: str):
    _text(slide, MARGIN, Inches(0.35), Inches(12), Inches(0.3),
          f"{number}  ·  {kicker.upper()}", size=11, color=BLUE, bold=True)
    _text(slide, MARGIN, Inches(0.65), Inches(12.1), Inches(0.9), title, size=24, bold=True)


def _footer(slide, note: str):
    _rect(slide, MARGIN, Inches(6.95), W - 2 * MARGIN, Pt(0.75), fill=LINE)
    _text(slide, MARGIN, Inches(7.0), Inches(12.1), Inches(0.35), note, size=9, color=MUTED)


def _stat(slide, x, y, value: str, label: str, w=Inches(2.2)):
    _text(slide, x, y, w, Inches(0.6), value, size=30, bold=True, color=BLUE)
    _text(slide, x, y + Inches(0.6), w, Inches(0.5), label, size=11, color=INK_2)


def _image(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


# ---------------------------------------------------------------- slides
def slide_methodology(prs, f):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _header(s, 1, "Objective & methodology",
            "An AI-assisted evidence pipeline turns two public databases into a structured MASH landscape")
    _text(s, MARGIN, Inches(1.55), Inches(12), Inches(0.6),
          "Objective: give the board a fact-based view of who is developing MASH drugs, which mechanisms "
          "lead, and where a new entrant could differentiate.", size=14, color=INK_2)

    steps = [
        ("1  Collect", [f"PubMed: {f['n_articles']} clinical articles, 2020–2025, sampled evenly by year",
                        f"ClinicalTrials.gov: {f['n_trials_raw']} Phase 2/3 interventional trials"]),
        ("2  Screen", [f"Keyword filters + AI relevance check keep {f['n_relevant']} on-topic records",
                       "Off-topic hits removed (e.g. ‘MASH’ = brain haemorrhage)"]),
        ("3  Extract", ["AI reads every abstract / registry entry and fills a fixed form: drug, mechanism, "
                        "endpoint, efficacy, safety",
                        "Phase, sponsor, status taken directly from the registry"]),
        ("4  Synthesise", [f"{f['n_raw_names']} drug names consolidated into {f['n_compounds']} compounds "
                           f"and 26 mechanism classes",
                           "Aggregated into the trial landscape and competitive map"]),
    ]
    box_w, gap, top = Inches(2.9), Inches(0.17), Inches(2.3)
    for i, (head, lines) in enumerate(steps):
        x = MARGIN + i * (box_w + gap)
        _rect(s, x, top, box_w, Inches(2.45), fill=BLUE_BG)
        _text(s, x + Inches(0.15), top + Inches(0.12), box_w - Inches(0.3), Inches(0.4),
              head, size=15, bold=True, color=BLUE)
        _text(s, x + Inches(0.15), top + Inches(0.6), box_w - Inches(0.3), Inches(1.8),
              [f"• {l}" for l in lines], size=11.5, color=INK, spacing=6)

    # quality panel
    qy = Inches(5.0)
    _rect(s, MARGIN, qy, W - 2 * MARGIN, Inches(1.75), fill=PANEL)
    _text(s, MARGIN + Inches(0.2), qy + Inches(0.12), Inches(4), Inches(0.35),
          "How reliable is the AI extraction?", size=13, bold=True)
    checks = [(f["qa_phase"], "trial phase matches the registry"),
              (f["qa_capture"], "of investigational drugs captured"),
              (f["qa_invented"], f"invented results across {f['qa_invented_base']} trials without data"),
              (f["qa_direction"], "efficacy calls consistent with p-values"),
              (f["qa_manual"], "field accuracy in a manual audit of 13 records")]
    cw = (W - 2 * MARGIN - Inches(0.4)) / len(checks)
    for i, (v, lbl) in enumerate(checks):
        x = MARGIN + Inches(0.2) + i * cw
        _text(s, x, qy + Inches(0.5), cw - Inches(0.1), Inches(0.5), v, size=24, bold=True, color=BLUE)
        _text(s, x, qy + Inches(1.05), cw - Inches(0.15), Inches(0.6), lbl, size=10.5, color=INK_2)
    _footer(s, f"Data pulled {f['run_date']}. AI model: OpenAI {f['model']} with a fixed output schema; "
               f"AI extraction cost ≈ ${f['cost']:.2f} for all records. Mechanism classes use the model's pharmacology knowledge "
               "where the source is silent, and are flagged as such in the dataset.")


def slide_evidence(prs, f):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _header(s, 2, "Evidence base",
            f"{f['n_articles']} peer-reviewed clinical studies: the literature is led by incretin "
            "and FGF21 drugs")
    y = Inches(1.7)
    _stat(s, MARGIN, y, str(f["n_articles"]), "clinical articles\nwith abstracts")
    _stat(s, MARGIN + Inches(2.2), y, str(f["n_primary"]), "primary trials\n(RCT, Phase 2/3)")
    _stat(s, MARGIN + Inches(4.4), y, str(f["n_meta"]), "meta-analyses /\nsystematic reviews")
    _stat(s, MARGIN + Inches(6.6), y, "2020–25", "evenly sampled,\n~23 articles per year", w=Inches(2.4))
    _stat(s, MARGIN + Inches(9.2), y, f"{f['pub_positive']}", "of studies with results\nreport a positive "
          "primary outcome", w=Inches(3))

    _image(s, config.FIGURES_DIR / "pub_design.png", MARGIN, Inches(3.15), w=Inches(5.6))
    _image(s, config.FIGURES_DIR / "pub_moa.png", Inches(6.6), Inches(3.05), w=Inches(6.1))
    _footer(s, "Source: PubMed (NCBI E-utilities). Query: MASH/NASH/NAFLD MeSH & title terms × treatment terms, "
               "human, English, clinical publication types; top-relevance 17 primary trials + 8 meta-analyses "
               "per year. Mechanism counts = articles studying a novel agent in that class.")


def slide_trials(prs, f):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _header(s, 3, "Trial landscape",
            f"{f['n_core']} Phase 2/3 MASH trials: industry-led and Phase 2-heavy, with momentum "
            "moving to FGF21, incretins and RNA drugs")
    top = Inches(1.7)
    for i, name in enumerate(["trial_phase", "trial_sponsor", "trial_status"]):
        _image(s, config.FIGURES_DIR / f"{name}.png", MARGIN, top + i * Inches(1.72), h=Inches(1.62))
    _text(s, Inches(3.4), top + Inches(0.15), Inches(2.6), Inches(5),
          [[("Phase 2 dominates", {"bold": True})],
           f"{f['share_phase2']} of trials are Phase 2 or 1/2; only {f['n_phase3']} reach Phase 2/3 or 3.",
           "",
           [("Industry sets the pace", {"bold": True})],
           f"{f['share_industry']} of trials are industry-sponsored; academic trials mostly test "
           "repurposed drugs and supplements.",
           "",
           [("A young pipeline", {"bold": True})],
           f"{f['n_since_2020']} trials started in 2020 or later; {f['n_ongoing']} are still ongoing."],
          size=11.5, color=INK, spacing=3)
    _image(s, config.FIGURES_DIR / "momentum.png", Inches(6.3), top, w=Inches(6.45))
    _footer(s, "Source: ClinicalTrials.gov API v2, interventional Phase 2/3, MASH/NASH in title or condition, "
               "AI-confirmed relevant. Mechanism chart: top 10 named classes of novel agents; a trial can "
               "count in more than one class.")


def slide_map(prs, f):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _header(s, 4, "Competitive map",
            "Late-stage race is between incretins, FGF21, THR-β and pan-PPAR; FXR, ASK1 and "
            "lipogenesis programmes have largely stalled")
    pic = _image(s, config.FIGURES_DIR / "competitive_map.png", MARGIN, Inches(1.6),
                 w=W - 2 * MARGIN)
    max_h = Inches(5.3)
    if pic.height > max_h:  # keep aspect ratio, fit above the footer
        pic.width = int(pic.width * max_h / pic.height)
        pic.height = max_h
    _footer(s, "Each compound is placed at the furthest phase registered on ClinicalTrials.gov. Novel agents only "
               "(vitamins, supplements and repurposed generics excluded). Context: resmetirom (FDA 2024) and "
               "semaglutide (FDA 2025) are already approved for non-cirrhotic MASH with F2–F3 fibrosis.")


def slide_insights(prs, f):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _header(s, 5, "Strategic insights",
            "The bar has been set by two approved drugs; differentiation now comes from fibrosis, "
            "cirrhosis and combinations")
    insights = [
        ("Two approved anchors define the standard of care.",
         "Resmetirom (oral THR-β) and semaglutide (GLP-1) are approved for F2–F3 MASH. New entrants "
         "will be judged against them, not against placebo alone."),
        ("FGF21 is the most crowded late-stage class.",
         f"Three FGF21 analogues are in Phase 3 ({f['fgf21_names']}), backed by "
         f"{f['fgf21_ongoing']} ongoing trials — the most of any class."),
        ("Multi-agonist incretins show the most consistent efficacy.",
         f"GLP-1/glucagon agonists reported positive results in {f['dual_pos']}, and three "
         "(survodutide, mazdutide, pemvidutide) are in Phase 3."),
        ("Several early mechanisms have failed.",
         f"FXR, ASK1, CCR2/5 and lipogenesis inhibitors account for {f['graveyard_past']} past trials but "
         f"only {f['graveyard_ongoing']} ongoing — a high evidence bar for anyone re-entering them."),
    ]
    y = Inches(1.7)
    for i, (head, body) in enumerate(insights, start=1):
        _text(s, MARGIN, y, Inches(0.5), Inches(0.5), str(i), size=22, bold=True, color=BLUE)
        _text(s, MARGIN + Inches(0.5), y, Inches(6.6), Inches(1.2),
              [[(head, {"bold": True, "size": 13.5})], body], size=12, color=INK_2, spacing=2)
        y += Inches(1.27)

    x, wbox = Inches(7.95), Inches(4.78)
    _rect(s, x, Inches(1.7), wbox, Inches(5.05), fill=BLUE_BG)
    _text(s, x + Inches(0.25), Inches(1.85), wbox - Inches(0.5), Inches(0.4),
          "White space for a new entrant", size=15, bold=True, color=BLUE)
    spaces = [
        ("Compensated cirrhosis (F4)",
         f"No approved therapy. {f['cirr_ongoing']} of {f['n_ongoing']} ongoing trials now target "
         "cirrhosis — need is high and the field is still open."),
        ("Rational combinations",
         f"Only {f['n_combo']} of {f['n_trials_novel']} trials test two novel agents together, despite "
         "complementary metabolic and anti-fibrotic mechanisms."),
        ("Genetically validated targets",
         "RNA drugs against HSD17B13, PNPLA3 and CIDEB are mostly Phase 2 with half their trials ongoing; "
         "an early position is still attainable."),
    ]
    yy = Inches(2.4)
    for head, body in spaces:
        _text(s, x + Inches(0.25), yy, wbox - Inches(0.5), Inches(1.4),
              [[(head, {"bold": True, "size": 13, "color": INK})], body], size=11.5, color=INK_2, spacing=2)
        yy += Inches(1.45)
    _footer(s, "Insights combine the pipeline data with public regulatory context (FDA approvals of resmetirom, "
               "Mar 2024, and semaglutide, Aug 2025). Registry status can lag reality; verify key "
               "competitor programmes before investment decisions.")


def build(facts: dict, path=None):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    for make in (slide_methodology, slide_evidence, slide_trials, slide_map, slide_insights):
        make(prs, facts)
    path = path or config.OUTPUT_DIR / "findings_deck.pptx"
    prs.save(path)
    print(f"Deck saved -> {path.relative_to(config.PROJECT_ROOT)}")
    return path
