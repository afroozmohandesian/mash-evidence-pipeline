"""Build the submission ZIP: lastname_firstname_mash_case.zip

The five deliverables sit at the top level of the archive under the names the
brief uses; supporting code and intermediate data follow in sub-folders.
Secrets (.env), the virtual environment, caches and old drafts are never
included — the archive is built from an explicit allow-list, not by
excluding things from the project folder.

Usage:  python -m src.package <lastname> <firstname>
"""
from __future__ import annotations

import sys
import zipfile

from . import config

ROOT = config.PROJECT_ROOT

DELIVERABLES = {
    "pipeline.ipynb": ROOT / "pipeline.ipynb",
    "pubmed_data.json": config.PUBMED_RAW_FILE,
    "trials_data.json": config.TRIALS_RAW_FILE,
    "landscape_data.csv": config.OUTPUT_DIR / "landscape_data.csv",
    "findings_deck.pptx": config.OUTPUT_DIR / "findings_deck.pptx",
}

SUPPORTING = [
    "README.md",
    "requirements.txt",
    "outputs/findings_deck.pdf",
    "data/processed/extraction_pass1.json",
    "data/processed/drug_map.csv",
    "data/processed/compounds.csv",
]
SUPPORTING_DIRS = ["src", "outputs/figures"]
FORBIDDEN = (".env", ".venv", "__pycache__", ".ipynb_checkpoints", "legacy", "cache")


def build(lastname: str, firstname: str):
    name = f"{lastname.lower()}_{firstname.lower()}_mash_case"
    zip_path = ROOT / f"{name}.zip"
    files = dict(DELIVERABLES)
    files.update({p: ROOT / p for p in SUPPORTING if (ROOT / p).exists()})
    for d in SUPPORTING_DIRS:
        for f in sorted((ROOT / d).rglob("*")):
            if (f.is_file() and not f.name.startswith(".")
                    and not any(part in FORBIDDEN for part in f.relative_to(ROOT).parts)):
                files[str(f.relative_to(ROOT))] = f

    for arcname in files:
        if any(part in FORBIDDEN for part in arcname.split("/")) or arcname.endswith(".env"):
            raise RuntimeError(f"refusing to package {arcname}")
    secret = config.OPENAI_API_KEY
    if secret:
        for arcname, path in files.items():
            if secret.encode() in path.read_bytes():
                raise RuntimeError(f"API key found in {arcname}; remove it before packaging")
    missing = [a for a, p in DELIVERABLES.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing deliverables: {missing}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in files.items():
            zf.write(path, f"{name}/{arcname}")
    size = zip_path.stat().st_size / 1e6
    print(f"{zip_path.name}: {len(files)} files, {size:.1f} MB")
    return zip_path


if __name__ == "__main__":
    build(*sys.argv[1:3])
