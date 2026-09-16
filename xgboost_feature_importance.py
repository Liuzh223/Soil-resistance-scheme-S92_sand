"""Train the beta model and compute XGBoost permutation importance.

This is a separate workflow, not the next stage after sample export or fitting.
load_data reads the four existing train/test NPY files specified in INPUT_FILES.
prepare_dataframes transposes no data itself: load_data already changes X from
(7, N) to (N, 7). Dropping fc/dg leaves five model inputs, including rd.
train_model uses FLAML; evaluate_model reports prediction metrics;
compute_permutation_importance evaluates the fitted estimator on the test set;
save_results writes numerical results under outputs/automl/<run-name>.
Figure 3b displays four variables, while the original calculation retains rd.
The 16-row arrays from select_evaporation_samples.py are not valid inputs here."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from config import OUTPUT_ROOT

# Inputs: wetness and sand fractions, wind (m/s), temperature (K), rd (s/m).
INPUT_FEATURES = ["swc_porsl", "fc", "sand", "um", "swt", "dg", "rd"]
REMOVE_VARS = ["fc", "dg"]
MODEL_FEATURES = ["swc_porsl", "sand", "um", "swt", "rd"]
PAPER_DISPLAY_FEATURES = ["swc_porsl", "sand", "um", "swt"]
INPUT_FILES = {
    "X_train": "X_train_hourly_20250115_beta_xgboost.npy",
    "y_train": "y_train_hourly_20250115_beta_xgboost.npy",
    "X_test": "X_test_hourly_20250115_beta_xgboost.npy",
    "y_test": "y_test_hourly_20250115_beta_xgboost.npy",
}


def load_data(base_path):
    """Read archived arrays: X is (variables, samples); y is dimensionless beta."""
    base_path = Path(base_path)
    arrays = {key: np.load(base_path / name, allow_pickle=False) for key, name in INPUT_FILES.items()}
    for subset in ("train", "test"):
        x, y = arrays[f"X_{subset}"], arrays[f"y_{subset}"]
        if x.ndim != 2 or x.shape[0] != len(INPUT_FEATURES):
            raise ValueError(f"X_{subset} must have shape (7, N) in the documented feature order; got {x.shape}")
        if y.ndim != 1 or x.shape[1] != len(y) or not len(y):
            raise ValueError(f"X/y shape mismatch or empty {subset} subset")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Nonfinite values in {subset}; inspect input rather than silently filtering samples")
    return arrays["X_train"].T, arrays["y_train"], arrays["X_test"].T, arrays["y_test"]


def prepare_dataframes(x_train, y_train, x_test, y_test, remove_vars=None):
    # Drop fc and dg in the main workflow; retain rd as in the original model.
    X_train = pd.DataFrame(x_train, columns=INPUT_FEATURES)
    X_test = pd.DataFrame(x_test, columns=INPUT_FEATURES)
    if remove_vars:
        invalid_vars = set(remove_vars) - set(INPUT_FEATURES)
        if invalid_vars:
            raise ValueError(f"Invalid variables to remove: {invalid_vars}")
        X_train = X_train.drop(columns=remove_vars)
        X_test = X_test.drop(columns=remove_vars)
    return X_train, y_train, X_test, y_test


def training_settings(output_dir):
    # Search only XGBoost, with a 1800-second budget and R-squared as the metric.
    return {
        "time_budget": 1800,
        "metric": "r2",
        "estimator_list": ["xgboost"],
        "task": "regression",
        "log_file_name": str(Path(output_dir) / "beta.log"),
        "seed": 7654321,
    }


def train_model(X_train, y_train, settings):
    from flaml import AutoML
    automl = AutoML()
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    return automl


def evaluate_model(automl, X_test, y_test):
    from flaml.automl.ml import sklearn_metric_loss_score
    y_pred = automl.predict(X_test)
    r2 = 1 - sklearn_metric_loss_score("r2", y_pred, y_test)
    mse = sklearn_metric_loss_score("mse", y_pred, y_test)
    mae = sklearn_metric_loss_score("mae", y_pred, y_test)
    return r2, mse, mae, y_pred


def compute_permutation_importance(model, X_val, y_val):
    """Mean decrease in test R-squared after permutation; values can exceed 1."""
    from sklearn.inspection import permutation_importance
    # Shuffle each feature 30 times on the test set, using random_state=42.
    result = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=42, scoring="r2")
    perm_importance = result.importances_mean
    return perm_importance


def save_results(automl, train_evaluation, test_evaluation, perm_importance, output_dir):
    output_dir = Path(output_dir)
    names = list(automl.feature_names_in_)
    if names != MODEL_FEATURES:
        raise ValueError(f"Unexpected model feature order: {names}")
    if len(perm_importance) != len(names):
        raise ValueError("Permutation importance length does not match features")
    pd.DataFrame([automl.feature_importances_], columns=names).to_csv(output_dir / "data_beta_importance.csv", index=False)
    # Save permutation importance separately from the model-internal importance.
    pd.DataFrame([perm_importance], columns=names).to_csv(output_dir / "beta_perm_importance.csv", index=False)
    np.save(output_dir / "x_pred_beta.npy", train_evaluation[3])
    np.save(output_dir / "y_pred_beta.npy", test_evaluation[3])
    pd.DataFrame([
        {"subset": "train", "r2": train_evaluation[0], "mse": train_evaluation[1], "mae": train_evaluation[2]},
        {"subset": "test", "r2": test_evaluation[0], "mse": test_evaluation[1], "mae": test_evaluation[2]},
    ]).to_csv(output_dir / "evaluation.csv", index=False)
    (output_dir / "best_config.json").write_text(json.dumps(automl.best_config, indent=2, default=_json_value)+"\n", encoding="utf-8")


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)}")


def main():
    """Load the archived split, train/evaluate the model, and save a separate run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the four original split NPY files")
    parser.add_argument("--run-name", default="original_five_features", help="New subdirectory name below outputs/automl")
    parser.add_argument("--check-inputs", action="store_true", help="Check shapes/feature order without training or writing outputs")
    args = parser.parse_args()
    if not args.run_name or args.run_name in (".", "..") or any(c in args.run_name for c in '/\\:'):
        parser.error("run-name must be a single directory name")
    x_train, y_train, x_test, y_test = load_data(args.data_dir)
    X_train, y_train, X_test, y_test = prepare_dataframes(x_train, y_train, x_test, y_test, remove_vars=REMOVE_VARS)
    print("Training/test shapes:", X_train.shape, X_test.shape)
    print("Model features:", list(X_train.columns))
    print("Paper Figure 3b display features:", PAPER_DISPLAY_FEATURES)
    # Input inspection stops here; it does not launch FLAML or write run outputs.
    if args.check_inputs:
        return
    output_dir = OUTPUT_ROOT / "automl" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = training_settings(output_dir)
    versions = {name: importlib.metadata.version(name) for name in ("flaml", "xgboost", "scikit-learn", "numpy", "pandas", "joblib")}
    provenance = {
        "settings": settings, "model_features": MODEL_FEATURES,
        "paper_display_features": PAPER_DISPLAY_FEATURES, "python": platform.python_version(),
        "package_versions": versions, "train_samples": len(y_train), "test_samples": len(y_test),
        "input_sha256": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() for name in INPUT_FILES.values()},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    automl = train_model(X_train, y_train, settings)
    import joblib
    joblib.dump(automl, output_dir / "beta_model_new_0224.pkl")
    test_evaluation = evaluate_model(automl, X_test, y_test)
    train_evaluation = evaluate_model(automl, X_train, y_train)
    # Evaluate the fitted estimator without retraining during permutation importance.
    # Use the held-out test set, not the training data or FLAML internal validation split.
    perm_importance = compute_permutation_importance(automl.model.estimator, X_test, y_test)
    save_results(automl, train_evaluation, test_evaluation, perm_importance, output_dir)
    print("Best hyperparameter configuration:", automl.best_config)
    print("Best internal validation R2:", 1 - automl.best_loss)
    print("Test R2/MSE/MAE:", test_evaluation[:3])
    print("Permutation importance:", perm_importance)
    print("Outputs:", output_dir)


if __name__ == "__main__":
    main()
