# Input data

| File | Content | Origin |
|---|---|---|
| `nebraska_weather.csv` | daily weather 1995–2261 (min/max temperature, precipitation, reference evapotranspiration) | file `notebooks/utils/nebraska.csv` of the thesis repo `gitlab.com/jbkze/aquacrop-rl` (commit of 2024-09-10); synthetic series for Champion, Nebraska, generated with LARS-WG by Kelly et al. (2024) |
| Tunis weather | 1979–2002 | shipped with AquaCrop-OSPy (`tunis_climate.txt`), read directly from the package |
| `external/` | results from outside this repo (GET-OPTIS, Simplex heuristics by Niels) | see below |

The experiments use the first 100 years of the Nebraska series: 1995–2064 for training and 2065–2094 for testing (the years in the manuscript tables). Open point: check that this file is identical to the one used for the published DRL2 and EA-DS results (compare with the Zenodo data archive, DOI 10.5281/zenodo.18373922).

## Files in `external/`

- `getoptis_ea_ds.csv`: columns `site, year, budget_mm, yield_t_ha`, one row per case. Created on 2026-09-25 from the manuscript tables in the paper repository (the column EA-DS of `tab/Tunis_results.csv` and the EA-DS value per test year in `tab/Nebraska_results_baseline.csv`), which were only read and not changed. Replace it with new GET-OPTIS results once the code is available.
- `getoptis_ea_ss.csv`: columns `site, budget_mm, yield_t_ha`, Tunis only, from the column EA(90%)-SS of the same Tunis table.
- `simplex_heuristics.csv` (expected from Niels): columns `site, year, budget_mm, method, yield_t_ha`.
