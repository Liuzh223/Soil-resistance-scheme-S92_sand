"""Shared paths for preprocessing, inversion, selection, and fitting.
Environment variables can override input locations."""
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

DERIVED_INPUT_DIR = Path(os.environ.get("S92_SAND_DERIVED_INPUT", str(OUTPUT_ROOT / "derived"))).resolve()
FIT_INPUT_DIR = Path(os.environ.get("S92_SAND_FIT_INPUT", str(OUTPUT_ROOT / "selected"))).resolve()
