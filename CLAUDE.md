# DIT-RL revision experiments

Experiment repo for the major revision of the DIT-RL paper (Agricultural Water Management, AGWAT-D-26-00823). Paper repo: `../DIT-RL` (Overleaf-synced). The project is independent of the PhD vault (no submodule, no WSE admin steps).

## Goal and status

Answer the reviewer requests with new experiments: ablation (E2), rules under the same budget (E1), value of weather information (E3), failure at low budgets (E4), robustness (E5). Plan: `config/plan.yaml` (numbers) and `docs/experiment-plan.md` (reasons), in three stages. Status 2026-10-09: code rebuilt around `config/plan.yaml`, checked against the previous code on 47 reference jobs (identical to the last digit); ablation pilot done on the WSE servers; no stage run yet. The GET-OPTIS code arrived from the co-author on 2026-10-09; the oracle values are still the published ones in `data/external/`.

## Entry points

- `ditrl list|run <stage or experiment> [--shard I/N] [--workers N] [--results DIR]`
- `ditrl collect`, `ditrl assets stage|export|check`
- `make install`, `make image`

## Conventions (enforced)

- Every parameter lives in `config/plan.yaml`; code holds no unnamed numbers. A variant is a new method or experiment entry, never a code copy.
- A job writes only into its own folder `results/jobs/<experiment>/<job id>/`; never edit a finished job folder.
- No unit tests (researcher's decision). A code change is checked by rerunning reference jobs and comparing them with the previous results number by number.
- One-offs go to `scratch/` (not committed).
- Paper numbers, figures and tables only through the asset registry `src/ditrl/analysis/assets.py`. Plots use `src/ditrl/analysis/style.py`; no heatmaps, no waterfalls.
- Prose, docstrings and docs follow the plain-language rules (no invented labels, every number with its reference noun) and are never hard-wrapped.

## Open points

- Check the DRL2 settings (action range, interval) against Kelly's archived code.
- Check that `data/nebraska_weather.csv` equals the file behind the published results.
