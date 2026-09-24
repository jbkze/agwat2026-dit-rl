# DIT-RL revision experiments

Experiment repo for the major revision of the DIT-RL paper (Agricultural Water Management,
AGWAT-D-26-00823). Paper repo: `../DIT-RL` (Overleaf-synced). Structure follows the
`wse-experiments` conventions of the phd-vault; the project itself is independent of the PhD
(no vault submodule, no WSE admin steps).

## Goal and status

Answer the reviewer requests with new experiments: ablation (E2), heuristics under the same
budget (E1), value of weather information (E3), failure at low budgets (E4), robustness (E5).
Plan: `docs/experiment-plan.md`. Status 2026-09-24: code complete, smoke test passes, first
comparison with the published Tunis values (`check_paper`) started. Campaigns not yet run.
Oracle values come from pymoo until Niels sends the GET-OPTIS code.

## Entry points

- `make smoke`, `make test`
- `make run CONFIG=config/experiments/<exp>.yaml WORKERS=60`
- `make collect`, `make paper-assets`, `make check-drift`
- Asset registry: `analysis/assets.py`

## Conventions (enforced)

- All parameters in `config/*.yaml`; methods in `config/agents.yaml`, sites in `config/sites.yaml`.
- Runs write immutable `results/runs/<date>_<experiment>/`; never edit them, resume with `--resume`.
- One-offs go to `scratch/YYYY-MM-DD_<name>.py`; no `_v2` copies, variants are config options.
- Paper numbers, figures and tables only through the asset registry.
- Plots: `analysis/paper_style.py`; no heatmaps, no waterfalls.
- Prose, docstrings and docs follow the plain-language rules (no invented labels, every number
  with its reference noun).

## Open points

- Check the DRL2 settings (action range, interval) against Kelly's archived code.
- Check that `data/nebraska_weather.csv` equals the file behind the published results.
- E2 observation factor: repeat the KELLY cells with `normalize: true` if the effect is large.
