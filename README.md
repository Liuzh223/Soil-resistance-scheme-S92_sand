# Soil resistance scheme S92_sand

| Python file | Purpose |
| --- | --- |
| `preprocess_data.py` | Prepare observations and auxiliary data from CoLM simulations using the S92 scheme. |
| `derive_soil_evaporation.py` | Calculate surface conditions, soil surface resistance, and evaporation efficiency (beta). |
| `select_evaporation_samples.py` | Select samples using E/ET and quality criteria, group them by soil wetness, and export the selected data. |
| `xgboost_feature_importance.py` | Train an XGBoost model with FLAML and calculate permutation feature importance. |
| `fit_s92_sand_parameters.py` | Fit site-specific parameters and their relationship with sand fraction. |
| `s92_sand_parameterization.py` | Calculate soil surface resistance using the final S92_sand scheme. |
| `physical_utils.py` | Provide supporting atmospheric and soil physics functions. |
| `config.py` | Configure input data locations, filename patterns, and output paths. |
