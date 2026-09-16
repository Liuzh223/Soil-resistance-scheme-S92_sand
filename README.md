# Soil resistance scheme S92_sand

Key calculation code for the S92_sand soil surface resistance parameterization:
data preparation, physical inversion, sample selection, XGBoost feature
importance, parameter fitting, and the final scheme. Auxiliary model variables
come from **CoLM simulations using the S92 scheme**.

This repository presents the calculation methods for inspection and reuse.
Input datasets, model outputs, trained models, plotting code, and the PhySO
search process are not included. It is not a complete reproduction package for
every experiment or figure in the paper.

## Code organization

| File | Purpose | Paper connection |
| --- | --- | --- |
| `preprocess_data.py` | Combine observations, forcing, site properties, and CoLM auxiliary variables. | Input preparation |
| `derive_soil_evaporation.py` | Estimate surface conditions, specific humidity, resistances, and observed beta. | Appendices B and C |
| `select_evaporation_samples.py` | Apply E/ET and quality filters, bin wetness, and export samples. | Sample selection |
| `xgboost_feature_importance.py` | Train XGBoost with FLAML and calculate permutation importance. | Figure 3b |
| `fit_s92_sand_parameters.py` | Fit site parameters and their relationship with sand fraction. | Section 2.2.2, Figure 3d, Eq. (27) |
| `s92_sand_parameterization.py` | Evaluate the final soil surface resistance scheme. | Eqs. (28)-(32) |
| `physical_utils.py` | Supporting atmospheric and soil calculations. | Appendices B and C |
| `config.py` | Configure input locations, filenames, and output paths. | Configuration |

Useful source comments are retained in English. Redundant calculations and
unused code have been removed while retaining the effective scientific formulas.

## Use the final parameterization

The final scheme can be called independently of data preparation and fitting:

```python
from s92_sand_parameterization import soilress_S92_sand

rss = soilress_S92_sand(
    vol_liq=0.20, porsl=0.45,
    bsw=4.0, psi0=-100.0, dz=0.0175, t=298.15, hksati=0.001,
    sand=0.50, om=0.02,
)
print(rss)  # Soil surface resistance, s/m
```

`vol_liq` and `porsl` are volumetric water content and saturated water content
(m3/m3). `sand` is a volumetric fraction: use `0.50` for 50% sand. The remaining
arguments retain the original common scheme interface and do not affect this
function's result.

```text
w = vol_liq/porsl, clipped to [0.001, 1]
b = ln(1 + exp(20.45*sand - 4.12))
F1 = 1 / (1 + exp(-50*(w - 0.35)))
F2 = 1 / (1 + exp(25*(w - 0.70)))
r_ss = (1-F1)*exp(8.206 - 6*w) + F1*exp(8.206 - b*w)*F2
```

Given surface-to-canopy-air resistance `r_d` in s/m, evaporation efficiency is
`beta = 1 / (1 + r_ss/r_d)`.

## Data preparation and parameter fitting

After configuring and providing the appropriate input data, run these scripts
from the repository directory:

```sh
python preprocess_data.py
python derive_soil_evaporation.py
python select_evaporation_samples.py
python fit_s92_sand_parameters.py
```

Preprocessing requires FLUXNET half-hourly observations, flux and meteorological
NetCDF files, site properties and BADM metadata, and CoLM S92 simulation outputs.
The inversion step also reads daily FLUXNET radiation QC. The station workbook
uses `sitename`, `start_end`, `start_end_qc`, `source`, and `run_flag` fields.
Model arrays must match the variables, dimensions, soil layers, and time
conventions used in `preprocess_data.py`.

Edit `config.py` or set the following environment variables:

| Variable | Meaning |
| --- | --- |
| `S92_SAND_DATA_ROOT` | Root containing observation, forcing, site, and model data |
| `S92_SAND_STATION_FILE` | Station workbook; defaults to `site.xlsx` under the data root |
| `S92_SAND_MODEL_INPUT` | Directory of CoLM S92 simulation files |
| `S92_SAND_MODEL_PATTERN` | Model filename template: `{sitename}`, `{start_year}`, `{end_year}` |
| `S92_SAND_BASE_INPUT` | Prepared base-data directory; defaults to `outputs/base_data` |
| `S92_SAND_BASE_PATTERN` | Base filename template: `{sitename}`, `{period}` |
| `S92_SAND_DERIVED_INPUT` | Inversion results; defaults to `outputs/derived` |
| `S92_SAND_FIT_INPUT` | Selected records; defaults to `outputs/selected` |

The default `model/S92` directory and `_S92.nc` filename patterns are configurable
examples, not bundled data. To use existing prepared base files, configure their
directory and filename pattern and start with `derive_soil_evaporation.py`.

Key implementation details:

- Observed temperature is converted to K and soil moisture from percent to a fraction.
- Selection retains the source thresholds, including LAI <= 1.5 and E/ET >= 0.9.
  These are inclusive bounds, unlike the strict inequalities stated in the manuscript.
- Wetness bins have width 0.05; bins with fewer than 50 samples are removed per site.
- Fitting uses 0.35 <= wetness < 0.6 and the original unweighted `curve_fit` settings.
- Station indices `[2, 3, 5, 7]` refer to the `run_flag == 1` subset and must
  correspond to US-AR1, US-SRG, US-Ton, and US-Wkg, respectively.
- The exported sample matrix has shape `(16, N)` and row order
  `swc, porsl, aird, fc, sand, om, um, swt, dg, rd, PET, beta, Qg, bsw, psi0, hksati`.
  **It includes the target beta and must not be used in full as ML input.**

Outputs are written under `outputs/base_data`, `outputs/derived`,
`outputs/selected`, `outputs/samples`, and `outputs/fit`. Fitting writes
`S92_sand_station_parameters.csv` and `S92_sand_coefficients.csv`.
Rerunning these stages replaces their corresponding output files.

## XGBoost permutation importance

This separate workflow reads an existing train/test split. It does not train
automatically on the 16-row sample matrix. Supply these four files:

```text
X_train_hourly_20250115_beta_xgboost.npy
y_train_hourly_20250115_beta_xgboost.npy
X_test_hourly_20250115_beta_xgboost.npy
y_test_hourly_20250115_beta_xgboost.npy
```

Each X array has shape `(7, N)` with rows
`swc_porsl, fc, sand, um, swt, dg, rd`; each y array has shape `(N,)` and contains
observed beta. The script drops `fc` and `dg`, retaining relative wetness,
sand fraction, wind speed, surface temperature, and `rd`.

**The archived calculation includes `rd` in both training and permutation
importance. Figure 3b displays the other four variables.** This code preserves
that calculation; it is not a model retrained after excluding `rd`.

```sh
# Check input shapes and feature order without training.
python xgboost_feature_importance.py --data-dir /path/to/split_arrays --check-inputs

# Train and save results in a new output directory.
python xgboost_feature_importance.py --data-dir /path/to/split_arrays --run-name run_01
```

FLAML searches XGBoost regressors for 1800 seconds, using R-squared and search
seed `7654321`. Permutation importance is evaluated on the external test set
with 30 repeats, `random_state=42`, and `scoring="r2"`. Values are decreases in
R-squared, are not normalized percentages, and can exceed 1.

Results are saved in `outputs/automl/<run-name>/`: the model, predictions,
evaluation metrics, configuration, and importance CSVs.
`beta_perm_importance.csv` contains permutation importance;
`data_beta_importance.csv` contains the separate model-internal importance.
Choose a new run name for each training run.

## Scope of validation

The final formula and fitted coefficients were checked locally, and cleanup
was checked against existing local intermediate results. These checks do not
establish complete reproduction from raw observations and the exact CoLM inputs.
The original AutoML environment was not recorded, so identical retraining
results are not guaranteed.

This source-only release does not include a dependency installation manifest.
Generated outputs and Python caches are excluded from version control.
