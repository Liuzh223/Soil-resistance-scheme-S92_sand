"""Shared input/output configuration for the calculation modules.

Data flow: preprocess_data -> derive_soil_evaporation
           -> select_evaporation_samples -> fit_s92_sand_parameters.
Each arrow represents files written by one script and read by the next;
importing config.py does not launch any stage or create any output directory.
The AutoML module uses --data-dir for its own inputs and shares OUTPUT_ROOT.
Environment overrides affect input locations only. Outputs always go to this
code directory's outputs folder; later stages must point to those outputs
unless existing compatible intermediate files are deliberately substituted."""
import os
from pathlib import Path

RELEASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("S92_SAND_DATA_ROOT", str(RELEASE_DIR.parent / "physo_all_data"))).resolve()
STATION_FILE = Path(os.environ.get("S92_SAND_STATION_FILE", str(DATA_ROOT / "site.xlsx"))).resolve()
OUTPUT_ROOT = RELEASE_DIR / "outputs"

# Example S92 input naming; configure the directory and pattern for the actual files.
MODEL_INPUT_DIR = Path(os.environ.get("S92_SAND_MODEL_INPUT", str(DATA_ROOT / "model/S92"))).resolve()
MODEL_FILE_PATTERN = os.environ.get("S92_SAND_MODEL_PATTERN", "{sitename}_{start_year}_{end_year}_S92.nc")
# By default, Step 2 reads the per-site files produced by preprocess_data.py.
BASE_INPUT_DIR = Path(os.environ.get("S92_SAND_BASE_INPUT", str(OUTPUT_ROOT / "base_data"))).resolve()
BASE_FILE_PATTERN = os.environ.get("S92_SAND_BASE_PATTERN", "{sitename}_{period}_S92.nc")

# Reader in select_evaporation_samples.py; producer is derive_soil_evaporation.py.
DERIVED_INPUT_DIR = Path(os.environ.get("S92_SAND_DERIVED_INPUT", str(OUTPUT_ROOT / "derived"))).resolve()
# Reader in fit_s92_sand_parameters.py; producer is select_evaporation_samples.py.
FIT_INPUT_DIR = Path(os.environ.get("S92_SAND_FIT_INPUT", str(OUTPUT_ROOT / "selected"))).resolve()
