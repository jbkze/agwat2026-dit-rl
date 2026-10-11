# Input data

| File | Content |
|---|---|
| `nebraska_weather.csv` | daily weather of the synthetic Nebraska series of Kelly et al. (2024); years 1995–2064 train, 2065–2094 test |
| Tunis weather | `tunis_climate.txt`, shipped with AquaCrop-OSPy and read from the package |
| `external/getoptis_ea_ds.csv` | best schedule per test year with known weather (`site, year, budget_mm, yield_t_ha`), taken from the manuscript tables |
| `external/getoptis_ea_ss.csv` | one robust schedule for all years, Tunis only (`site, budget_mm, yield_t_ha`), from the manuscript tables |
| `external/simplex_heuristics.csv` | Simplex rules (`site, year, budget_mm, method, yield_t_ha`), pending |
