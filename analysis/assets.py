"""Paper asset registry: the only definition of what flows into the paper.

    python -m analysis.assets stage    # regenerate everything into paper_assets/ only
    python -m analysis.assets export   # regenerate everything and copy it into the paper repo
    python -m analysis.assets check    # regenerate into a temp folder and compare (drift check)

The paper repo path comes from config/paths.local.yaml (not committed).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from analysis import fig_ablation, fig_actions, fig_cwpf, fig_learning, macros, tables

REPO = Path(__file__).resolve().parents[1]
STAGING = REPO / "paper_assets"

# file in paper_assets/ -> (generator, note)
ASSETS = {
    "macros.tex": (macros.generate, "All numbers used in the prose"),
    "figures/cwpf.pdf": (fig_cwpf.generate, "Fig: yield over budget, all methods (E1)"),
    "figures/ablation.pdf": (fig_ablation.generate, "Fig: main effects of the ablation (E2)"),
    "figures/actions_low_budget.pdf": (fig_actions.generate, "Fig: actions at 100-200 mm (E4)"),
    "figures/learning_curves.pdf": (fig_learning.generate, "Fig: validation yield over training (E6)"),
    "tables/ablation.tex": (tables.ablation, "Tab: ablation effects (E2)"),
    "tables/information.tex": (tables.information, "Tab: value of weather information (E3)"),
    "tables/robustness.tex": (tables.robustness, "Tab: lower tail and noise (E5)"),
    "tables/hyperparameters.tex": (tables.hyperparameters, "Tab: PPO settings (R2#3)"),
}


def _paper_repo() -> Path:
    cfg = yaml.safe_load((REPO / "config" / "paths.local.yaml").read_text())
    return Path(cfg["paper_repo"])


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _generate_all(target: Path) -> dict[str, str]:
    hashes = {}
    for rel, (generate, _) in ASSETS.items():
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        generate(out)
        hashes[rel] = _sha(out)
    return hashes


def _write_manifest(target: Path, hashes: dict[str, str]) -> None:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    lines = ["# Generated assets, do not edit by hand", f"Source: experiment repo @ {sha}", "",
             "| asset | content hash | note |", "|---|---|---|"]
    lines += [f"| {rel} | {hashes[rel]} | {note} |" for rel, (_, note) in ASSETS.items()]
    (target / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def stage() -> None:
    """Regenerate everything into paper_assets/ only; the paper repo stays untouched. An asset
    whose inputs are missing is reported and skipped, so that the others are still built."""
    hashes = {}
    for rel, (generate, _) in ASSETS.items():
        out = STAGING / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            generate(out)
            hashes[rel] = _sha(out)
        except Exception as err:  # noqa: BLE001  (report and continue with the next asset)
            hashes[rel] = f"not built: {err}"
            print(f"SKIPPED {rel}: {err}")
    _write_manifest(STAGING, hashes)
    print(f"staged {sum(not h.startswith('not built') for h in hashes.values())} of "
          f"{len(ASSETS)} assets in {STAGING}")


def export() -> None:
    hashes = _generate_all(STAGING)
    _write_manifest(STAGING, hashes)
    dest = _paper_repo() / "generated"
    shutil.copytree(STAGING, dest, dirs_exist_ok=True)
    print(f"exported {len(ASSETS)} assets to {dest}")


def check() -> None:
    dest = _paper_repo() / "generated"
    with tempfile.TemporaryDirectory() as tmp:
        fresh = _generate_all(Path(tmp))
    stale = [rel for rel, h in fresh.items() if not (dest / rel).exists() or _sha(dest / rel) != h]
    orphans = [p.relative_to(dest).as_posix() for p in dest.rglob("*")
               if p.is_file() and p.relative_to(dest).as_posix() not in ASSETS
               and p.name != "MANIFEST.md" and p.suffix != ".csv"]
    for rel in stale:
        print(f"STALE:  {rel}  (regenerate with make paper-assets)")
    for rel in orphans:
        print(f"ORPHAN: {rel}  (in generated/ but not in the registry)")
    if stale or orphans:
        sys.exit(1)
    print(f"in sync: all {len(fresh)} assets match")


if __name__ == "__main__":
    {"stage": stage, "export": export, "check": check}[sys.argv[1]]()
