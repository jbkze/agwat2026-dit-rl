"""Paper asset registry: the only definition of what flows into the paper.

    ditrl assets stage    regenerate everything into paper_assets/ only
    ditrl assets export   regenerate everything and copy it into the paper repo
    ditrl assets check    regenerate into a temporary folder and compare (drift check)

The paper repo path comes from config/paths.local.yaml (not committed).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

from ditrl.analysis import figures, macros, tables
from ditrl.paths import LOCAL_PATHS, REPO
from ditrl.plan import Plan
from ditrl.runner import git

STAGING = REPO / "paper_assets"
PAPER_FOLDER = "generated"   # folder in the paper repo that receives the assets
HASH_LENGTH = 12

# file in paper_assets/ -> (generator, note)
ASSETS = {
    "macros.tex": (macros.generate, "All numbers used in the prose"),
    "figures/cwpf.pdf": (figures.cwpf, "Fig: yield over budget, all methods (E1)"),
    "figures/ablation.pdf": (figures.ablation, "Fig: paired differences of the ablation (E2)"),
    "figures/actions_low_budget.pdf": (figures.actions_low_budget,
                                       "Fig: actions at 100-200 mm (E4)"),
    "figures/learning_curves.pdf": (figures.learning_curves,
                                    "Fig: validation yield over training (E6)"),
    "tables/ablation.tex": (tables.ablation, "Tab: ablation, paired differences (E2)"),
    "tables/ablation_effects.tex": (tables.ablation_effects, "Tab: ablation, main effects (E2)"),
    "tables/information.tex": (tables.information, "Tab: value of weather information (E3)"),
    "tables/robustness.tex": (tables.robustness, "Tab: lower tail and noise (E5)"),
    "tables/hyperparameters.tex": (tables.hyperparameters, "Tab: PPO settings"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:HASH_LENGTH]


def _paper_dir() -> Path:
    return Path(yaml.safe_load(LOCAL_PATHS.read_text())["paper_repo"]) / PAPER_FOLDER


def _generate(plan: Plan, results: Path, target: Path, skip_missing: bool) -> dict[str, str]:
    """Build every asset into `target`; returns its content hash, or why it was not built."""
    hashes = {}
    for rel, (generate, _) in ASSETS.items():
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            generate(plan, results, out)
            hashes[rel] = _sha(out)
        except FileNotFoundError as err:
            if not skip_missing:
                raise
            hashes[rel] = f"not built: {err}"
            print(f"SKIPPED {rel}: {err}")
    return hashes


def _write_manifest(target: Path, hashes: dict[str, str]) -> None:
    lines = ["# Generated assets, do not edit by hand",
             f"Source: experiment repo @ {git('rev-parse', '--short', 'HEAD')}", "",
             "| asset | content hash | note |", "|---|---|---|"]
    lines += [f"| {rel} | {hashes[rel]} | {note} |" for rel, (_, note) in ASSETS.items()]
    (target / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage(plan: Plan, results: Path) -> None:
    """Into paper_assets/ only; an asset whose inputs are missing is reported and skipped."""
    hashes = _generate(plan, results, STAGING, skip_missing=True)
    _write_manifest(STAGING, hashes)
    built = sum(not h.startswith("not built") for h in hashes.values())
    print(f"staged {built} of {len(ASSETS)} assets in {STAGING}")


def export(plan: Plan, results: Path) -> None:
    hashes = _generate(plan, results, STAGING, skip_missing=False)
    _write_manifest(STAGING, hashes)
    shutil.copytree(STAGING, _paper_dir(), dirs_exist_ok=True)
    print(f"exported {len(ASSETS)} assets to {_paper_dir()}")


def check(plan: Plan, results: Path) -> None:
    dest = _paper_dir()
    with tempfile.TemporaryDirectory() as tmp:
        fresh = _generate(plan, results, Path(tmp), skip_missing=False)
    stale = [rel for rel, h in fresh.items() if not (dest / rel).exists() or _sha(dest / rel) != h]
    orphans = [p.relative_to(dest).as_posix() for p in dest.rglob("*")
               if p.is_file() and p.suffix != ".csv" and p.name != "MANIFEST.md"
               and p.relative_to(dest).as_posix() not in ASSETS]
    for rel in stale:
        print(f"STALE:  {rel}  (regenerate with ditrl assets export)")
    for rel in orphans:
        print(f"ORPHAN: {rel}  (in {PAPER_FOLDER}/ but not in the registry)")
    if stale or orphans:
        sys.exit(1)
    print(f"in sync: all {len(fresh)} assets match")
