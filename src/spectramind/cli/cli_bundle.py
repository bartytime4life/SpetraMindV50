import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict

import typer

from .common import (
    PROJECT_ROOT,
    REPORTS_DIR,
    logger,
    command_session,
)

app = typer.Typer(no_args_is_help=True, help="SpectraMind V50 — Bundle prediction & validator")

REQUIRED_FILES = [
    "predictions/mu.csv",
    "predictions/sigma.csv",
]

def _manifest(bundle_dir: Path) -> Dict:
    return {
        "bundle_dir": str(bundle_dir),
        "files": sorted([str(p.relative_to(PROJECT_ROOT)) for p in bundle_dir.rglob("*") if p.is_file()]),
        "created": True,
    }


@app.command("make")
def make(
    preds_dir: str = typer.Option("predictions", "--preds"),
    bundle_dir: str = typer.Option("submission_bundle", "--out"),
    zip_name: str = typer.Option("submission.zip", "--zip"),
):
    """Bundle μ/σ predictions and artifacts into a submission directory and zip."""
    with command_session("bundle.make", ["--preds", preds_dir, "--out", bundle_dir, "--zip", zip_name]):
        p_preds = Path(preds_dir)
        p_bundle = Path(bundle_dir)
        p_bundle.mkdir(parents=True, exist_ok=True)
        # Validate required files exist
        for rf in REQUIRED_FILES:
            f = PROJECT_ROOT / rf.replace("predictions", preds_dir)
            if not f.exists():
                logger.error("Missing required file: %s", f)
                raise typer.Exit(2)
        # Copy files
        for src in p_preds.glob("*"):
            if src.is_file():
                shutil.copy2(src, p_bundle / src.name)
        # Write manifest
        manifest = REPORTS_DIR / "submission_manifest.json"
        manifest.write_text(json.dumps(_manifest(p_bundle), indent=2), encoding="utf-8")
        # Zip it
        zip_path = Path(zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(p_bundle.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(p_bundle))
        logger.info("Bundle: %s", p_bundle)
        logger.info("Zip: %s", zip_path)
        raise typer.Exit(0)
