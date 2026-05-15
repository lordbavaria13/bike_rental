from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelling.common.config import (
    DPI,
    EXPERIMENTS,
    FIGSIZE,
    LABEL_SIZE,
    STATION_COL,
    TARGET_COL,
    TIME_COL,
    TITLE_SIZE,
    get_experiment_paths,
    validate_experiment,
)
from modelling.common.metrics import compute_regression_metrics
from modelling.common.plotting import (
    plot_actual_vs_predicted,
    plot_coefficients,
    plot_error_over_time,
    plot_feature_importance,
    plot_residuals_histogram,
    plot_residuals_vs_predicted,
)
from modelling.common.preprocessing import (
    extract_prediction_context,
    load_dataset,
    load_encoded_datasets,
    prepare_encoded_feature_matrices_for_model,
    prepare_feature_matrices,
)
from modelling.common.split import chronological_split
from modelling.common.utils import ensure_dirs, save_dataframe, save_json
from modelling.common.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO

ModelBuilder = Callable[[dict[str, Any]], Any]
LabelBuilder = Callable[[pd.Series], str]


def parse_experiment_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["all", *EXPERIMENTS],
        default="all",
        help="Run one experiment variant or both variants.",
    )
    return parser.parse_args()


def selected_experiments(experiment: str) -> list[str]:
    if experiment == "all":
        return list(EXPERIMENTS)
    validate_experiment(experiment)
    return [experiment]


def default_param_label(row: pd.Series) -> str:
    keys = [key for key in row.index if key not in {"train_rmse", "validation_rmse", "train_mae", "validation_mae"}]
    return " | ".join(f"{key}={row[key]}" for key in keys)


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            safe[key] = None
        elif not isinstance(value, (list, tuple, dict, str)) and pd.isna(value):
            safe[key] = None
        elif isinstance(value, tuple):
            safe[key] = list(value)
        elif isinstance(value, (np.integer,)):
            safe[key] = int(value)
        elif isinstance(value, (np.floating,)):
            safe[key] = float(value)
        elif isinstance(value, (np.bool_,)):
            safe[key] = bool(value)
        else:
            safe[key] = value
    return safe


def save_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred,
    val_pred,
    test_pred,
    output_path: Path,
) -> pd.DataFrame:
    """Save train/validation/test predictions with context for error analysis."""

    def build_split_df(df_part: pd.DataFrame, preds, split_name: str) -> pd.DataFrame:
        out = extract_prediction_context(df_part)
        out["split"] = split_name
        out["prediction"] = preds
        out["residual"] = out[TARGET_COL] - out["prediction"]
        out["absolute_error"] = out["residual"].abs()
        return out

    pred_df = pd.concat(
        [
            build_split_df(train_df, train_pred, "train"),
            build_split_df(val_df, val_pred, "validation"),
            build_split_df(test_df, test_pred, "test"),
        ],
        ignore_index=True,
    )

    save_dataframe(pred_df, output_path, index=False)
    return pred_df


def plot_search_results(
    search_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    label_builder: LabelBuilder | None = None,
) -> None:
    """Plot validation RMSE for a hyperparameter search."""
    if search_df.empty:
        return

    plot_df = search_df.copy().reset_index(drop=True)
    if label_builder is None:
        label_builder = default_param_label

    plot_df["label"] = plot_df.apply(label_builder, axis=1)
    x_pos = list(range(len(plot_df)))

    plt.figure(figsize=(max(10, len(plot_df) * 0.45), 5))
    plt.plot(x_pos, plot_df["validation_rmse"].to_numpy(), marker="o")
    plt.xticks(x_pos, plot_df["label"].tolist(), rotation=45, ha="right")
    plt.title(f"{model_name} - Validation RMSE by Parameter Setting", fontsize=TITLE_SIZE)
    plt.xlabel("parameter setting", fontsize=LABEL_SIZE)
    plt.ylabel("Validation RMSE", fontsize=LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def _load_matrices_for_experiment(experiment: str, scale_numeric: bool):
    """Load encoded data if present, otherwise fall back to raw data."""
    paths = get_experiment_paths(experiment)
    encoded_paths_available = all(
        paths[key].exists()
        for key in ["encoded_train_path", "encoded_val_path", "encoded_test_path"]
    )

    if encoded_paths_available:
        print(f"Using encoded datasets from: {paths['experiment_dir']}")
        train_df, val_df, test_df = load_encoded_datasets(
            train_path=paths["encoded_train_path"],
            val_path=paths["encoded_val_path"],
            test_path=paths["encoded_test_path"],
            target_col=TARGET_COL,
        )
        scaler, feature_names, X_train, X_val, X_test = prepare_encoded_feature_matrices_for_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            target_col=TARGET_COL,
            feature_names=None,
            scale_numeric=scale_numeric,
        )
        preprocessor = None
    else:
        print(f"Encoded datasets missing. Falling back to raw dataset: {paths['data_path']}")
        df = load_dataset(paths["data_path"])
        train_df, val_df, test_df = chronological_split(
            df=df,
            time_col=TIME_COL,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
        )
        (
            preprocessor,
            feature_names,
            X_train,
            X_val,
            X_test,
            _,
            _,
            _,
        ) = prepare_feature_matrices(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            target_col=TARGET_COL,
            categorical_cols=[STATION_COL],
            scale_numeric=scale_numeric,
        )
        scaler = None

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    return train_df, val_df, test_df, feature_names, X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, scaler


def run_regression_experiment(
    *,
    model_name: str,
    model_filename: str,
    model_base_dir: Path,
    build_model: ModelBuilder,
    param_grid: list[dict[str, Any]] | None = None,
    experiment: str = "all",
    scale_numeric: bool = False,
    station_id_encoding: str = "one_hot",
    save_coefficients: bool = False,
    save_feature_importance: bool = False,
    label_builder: LabelBuilder | None = None,
    extra_model_info: dict[str, Any] | None = None,
    plot_loss_curve: bool = False,
) -> None:
    """Run one model for one or both experiment variants."""
    for experiment_name in selected_experiments(experiment):
        print("=" * 80)
        print(f"{model_name} | experiment={experiment_name}")
        print("=" * 80)

        results_dir = model_base_dir / "results" / experiment_name
        plots_dir = results_dir / "plots"
        model_dir = model_base_dir / "model" / experiment_name
        ensure_dirs(results_dir, plots_dir, model_dir)

        (
            train_df,
            val_df,
            test_df,
            feature_names,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            preprocessor,
            scaler,
        ) = _load_matrices_for_experiment(experiment_name, scale_numeric=scale_numeric)

        print(f"Train shape: {train_df.shape}")
        print(f"Validation shape: {val_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Using {len(feature_names)} final features.")

        if param_grid is None:
            param_grid = [{}]

        search_records: list[dict[str, Any]] = []
        best_params = param_grid[0]

        if len(param_grid) > 1:
            print("Searching best hyperparameters...")
            for params in param_grid:
                model_tmp = build_model(params)
                model_tmp.fit(X_train, y_train)
                train_pred_tmp = model_tmp.predict(X_train)
                val_pred_tmp = model_tmp.predict(X_val)
                train_metrics = compute_regression_metrics(y_train, train_pred_tmp, "train")
                val_metrics = compute_regression_metrics(y_val, val_pred_tmp, "validation")
                record = {
                    **_json_safe_params(params),
                    "train_rmse": train_metrics["train_rmse"],
                    "validation_rmse": val_metrics["validation_rmse"],
                    "train_mae": train_metrics["train_mae"],
                    "validation_mae": val_metrics["validation_mae"],
                }
                search_records.append(record)

            search_df = pd.DataFrame(search_records).sort_values(
                ["validation_rmse", "train_rmse"]
            ).reset_index(drop=True)
            save_dataframe(search_df, results_dir / "hyperparameter_search.csv", index=False)
            plot_search_results(
                search_df=search_df,
                model_name=model_name,
                output_path=plots_dir / "validation_curve.png",
                label_builder=label_builder,
            )
            best_params = {
                key: search_df.iloc[0][key]
                for key in _json_safe_params(param_grid[0]).keys()
            }
            # Convert pandas/numpy values back to plain values and restore None where needed.
            best_params = _json_safe_params(best_params)
            print(f"Best params: {best_params}")
        else:
            print("No hyperparameter search for this model.")
            save_dataframe(pd.DataFrame(_json_safe_params(p) for p in param_grid), results_dir / "hyperparameter_search.csv", index=False)

        model = build_model(best_params)

        print("Training final model...")
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        print("Generating predictions...")
        pred_start = time.perf_counter()
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        predict_time = time.perf_counter() - pred_start

        metrics = {
            "model_name": model_name,
            "experiment": experiment_name,
            "uses_lag_features": bool(experiment_name == "with_lag"),
            "target": TARGET_COL,
            "n_features": int(len(feature_names)),
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
            "fit_time_seconds": float(fit_time),
            "predict_time_seconds": float(predict_time),
            "station_id_encoding": station_id_encoding,
            "scaling_used": bool(scale_numeric),
            "best_params": _json_safe_params(best_params),
        }
        if extra_model_info:
            metrics.update(extra_model_info)

        if hasattr(model, "intercept_"):
            try:
                intercept = model.intercept_
                metrics["intercept"] = float(np.ravel(intercept)[0])
            except Exception:
                pass
        if hasattr(model, "coef_"):
            coef = np.ravel(model.coef_)
            metrics["coef_l2_norm"] = float(np.sqrt(np.sum(coef ** 2)))
            metrics["n_nonzero_coefficients"] = int(np.sum(np.abs(coef) > 1e-12))

        metrics.update(compute_regression_metrics(y_train, train_pred, "train"))
        metrics.update(compute_regression_metrics(y_val, val_pred, "validation"))
        metrics.update(compute_regression_metrics(y_test, test_pred, "test"))

        save_dataframe(pd.DataFrame([metrics]), results_dir / "metrics.csv", index=False)
        save_json(metrics, results_dir / "metrics.json")

        pred_df = save_predictions(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_pred=train_pred,
            val_pred=val_pred,
            test_pred=test_pred,
            output_path=results_dir / "predictions.csv",
        )

        print("Saving model artifacts...")
        joblib.dump(model, model_dir / model_filename)
        if preprocessor is not None:
            joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
        if scaler is not None:
            joblib.dump(scaler, model_dir / "scaler.joblib")

        model_info = {
            "model_name": model_name,
            "experiment": experiment_name,
            "uses_lag_features": bool(experiment_name == "with_lag"),
            "target": TARGET_COL,
            "feature_names": feature_names,
            "n_final_features": len(feature_names),
            "station_id_encoding": station_id_encoding,
            "scaling_used": bool(scale_numeric),
            "best_params": _json_safe_params(best_params),
            "param_grid": [_json_safe_params(p) for p in param_grid],
            "results_dir": str(results_dir),
            "plots_dir": str(plots_dir),
            "model_file": str(model_dir / model_filename),
            "metrics": metrics,
        }
        if extra_model_info:
            model_info.update(extra_model_info)
        save_json(model_info, model_dir / "model_info.json")

        plot_actual_vs_predicted(
            pred_df=pred_df,
            target_col=TARGET_COL,
            output_path=plots_dir / "actual_vs_predicted_test.png",
            model_name=f"{model_name} ({experiment_name})",
            figsize=FIGSIZE,
            dpi=DPI,
            title_size=TITLE_SIZE,
            label_size=LABEL_SIZE,
        )
        plot_residuals_histogram(
            pred_df=pred_df,
            output_path=plots_dir / "residuals_histogram_test.png",
            model_name=f"{model_name} ({experiment_name})",
            figsize=FIGSIZE,
            dpi=DPI,
            title_size=TITLE_SIZE,
            label_size=LABEL_SIZE,
        )
        plot_residuals_vs_predicted(
            pred_df=pred_df,
            output_path=plots_dir / "residuals_vs_predicted_test.png",
            model_name=f"{model_name} ({experiment_name})",
            figsize=FIGSIZE,
            dpi=DPI,
            title_size=TITLE_SIZE,
            label_size=LABEL_SIZE,
        )
        plot_error_over_time(
            pred_df=pred_df,
            time_col=TIME_COL,
            output_path=plots_dir / "error_over_time_test.png",
            model_name=f"{model_name} ({experiment_name})",
            figsize=FIGSIZE,
            dpi=DPI,
            title_size=TITLE_SIZE,
            label_size=LABEL_SIZE,
        )

        if save_coefficients and hasattr(model, "coef_"):
            coef = np.ravel(model.coef_)
            coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
            coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
            coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
            save_dataframe(coef_df, results_dir / "coefficients.csv", index=False)
            plot_height = max(6, len(feature_names) * 0.20)
            plot_coefficients(
                feature_names=feature_names,
                coefficients=coef,
                output_path=plots_dir / "coefficients.png",
                model_name=f"{model_name} ({experiment_name})",
                figsize=(12, plot_height),
                dpi=DPI,
                title_size=TITLE_SIZE,
                label_size=LABEL_SIZE,
            )

        if save_feature_importance and hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            save_dataframe(importance_df, results_dir / "feature_importance.csv", index=False)
            plot_height = max(6, len(feature_names) * 0.20)
            plot_feature_importance(
                feature_names=feature_names,
                importances=model.feature_importances_,
                output_path=plots_dir / "feature_importance.png",
                model_name=f"{model_name} ({experiment_name})",
                figsize=(12, plot_height),
                dpi=DPI,
                title_size=TITLE_SIZE,
                label_size=LABEL_SIZE,
            )

        if plot_loss_curve and hasattr(model, "loss_curve_"):
            plt.figure(figsize=FIGSIZE)
            plt.plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_, marker="o", markersize=2)
            plt.title(f"{model_name} ({experiment_name}) - Training Loss Curve", fontsize=TITLE_SIZE)
            plt.xlabel("Iteration", fontsize=LABEL_SIZE)
            plt.ylabel("Loss", fontsize=LABEL_SIZE)
            plt.tight_layout()
            plt.savefig(plots_dir / "training_loss_curve.png", dpi=DPI)
            plt.close()

        print(f"Done. Results saved to: {results_dir}")
