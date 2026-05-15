from __future__ import annotations

"""
run_pipeline.py

How we use this script
----------------------
We use this script to run the whole project pipeline from one file.

Before running it:
1. Open the project root folder.
2. Activate the virtual environment:
       source .venv/bin/activate
3. Start the script:
       python run_pipeline.py

How we configure it
-------------------
We change the settings directly in the configuration block below.

Example:
- STEP_1_DATA_PROCESSING = 1   -> run data processing
- STEP_1_DATA_PROCESSING = 0   -> skip data processing

The same logic is used for all model steps and for the final comparison.

If we only want to run some models, we set them to 1.
If we want to skip a model, we set it to 0.

Why we wrote it this way
------------------------
We wanted one simple script that we can edit quickly without using command
line arguments every time. This makes it easy to rerun the full workflow
and keep all models under the same setup.
"""

import subprocess
import sys
import time
from pathlib import Path


# =============================================================================
# CONFIGURATION
# Change 1 to run a step.
# Change 0 to skip a step.
# =============================================================================

# Step 1: data processing
STEP_1_DATA_PROCESSING = 1

# Step 2: model training
RUN_00_DUMMY = 1
RUN_01_LINEAR = 1
RUN_02_RIDGE = 1
RUN_03_LASSO = 1
RUN_04_DECISION_TREE = 1
RUN_05_KNN = 1
RUN_06_RANDOM_FOREST = 1
RUN_07_GRADIENT_BOOSTING = 1
RUN_08_NEURAL_NETWORK = 1

# Step 3: final model comparison
STEP_3_MODEL_COMPARISON = 1


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXECUTABLE = sys.executable


# =============================================================================
# PREPROCESSING SCRIPTS
# These scripts rebuild the processed modelling dataset.
# They are run in this exact order.
# =============================================================================

PREPROCESSING_SCRIPTS = [
    PROJECT_ROOT / "src" / "scripts" / "02_build_top20_daily_dataset.py",
    PROJECT_ROOT / "src" / "scripts" / "03_analyze_feature_correlations.py",
    PROJECT_ROOT / "src" / "scripts" / "04_build_reduced_feature_dataset.py",
    PROJECT_ROOT / "src" / "scripts" / "05_create_encoded_dataset.py",
]


# =============================================================================
# MODEL MODULES
# These are started with "python -m ...".
# This is important because the project uses imports like:
# from modelling.common.config import ...
# =============================================================================

MODEL_STEPS = [
    (RUN_00_DUMMY, "00_dummy_regressor", "modelling.00_dummy_regressor.train_dummy"),
    (RUN_01_LINEAR, "01_linear_regression", "modelling.01_linear_regression.train_linear_regression"),
    (RUN_02_RIDGE, "02_ridge_regression", "modelling.02_ridge_regression.train_ridge"),
    (RUN_03_LASSO, "03_lasso_regression", "modelling.03_lasso_regression.train_lasso"),
    (RUN_04_DECISION_TREE, "04_decision_tree", "modelling.04_decision_tree.train_decision_tree"),
    (RUN_05_KNN, "05_knn_regressor", "modelling.05_knn_regressor.train_knn"),
    (RUN_06_RANDOM_FOREST, "06_random_forest", "modelling.06_random_forest.train_random_forest"),
    (RUN_07_GRADIENT_BOOSTING, "07_gradient_boosting", "modelling.07_gradient_boosting.train_gradient_boosting"),
    (RUN_08_NEURAL_NETWORK, "08_neural_network", "modelling.08_neural_network.train_neural_network"),
]


# =============================================================================
# FINAL MODEL COMPARISON
# =============================================================================

COMPARISON_MODULE = "modelling.99_model_comparison.model_comparison"


# =============================================================================
# HELPERS
# =============================================================================

def print_line() -> None:
    print("=" * 80)


def print_header(title: str) -> None:
    print()
    print_line()
    print(title)
    print_line()


def format_runtime(seconds: float) -> str:
    return f"{seconds:.2f} s"


def check_required_files() -> None:
    """
    Check if all preprocessing scripts exist before the pipeline starts.
    """
    missing_paths: list[Path] = []

    for script_path in PREPROCESSING_SCRIPTS:
        if not script_path.exists():
            missing_paths.append(script_path)

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Some preprocessing scripts are missing:\n"
            f"{missing_text}"
        )


def run_command(command: list[str], description: str) -> float:
    """
    Run one command inside the project root.

    We stop the whole pipeline if one step fails.
    """
    print(f"\nRunning: {description}")
    print("Command:", " ".join(command))

    start_time = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    runtime = time.perf_counter() - start_time

    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {description}\n"
            f"Return code: {result.returncode}"
        )

    print(f"Finished: {description} ({format_runtime(runtime)})")
    return runtime


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def run_data_processing() -> list[tuple[str, float]]:
    """
    Run all preprocessing scripts in the correct order.
    """
    print_header("STEP 1: DATA PROCESSING")

    timings: list[tuple[str, float]] = []

    for script_path in PREPROCESSING_SCRIPTS:
        relative_name = script_path.relative_to(PROJECT_ROOT).as_posix()
        runtime = run_command(
            [PYTHON_EXECUTABLE, str(script_path)],
            relative_name,
        )
        timings.append((relative_name, runtime))

    return timings


def run_model_training() -> list[tuple[str, float]]:
    """
    Run all selected model scripts in chronological order.
    """
    print_header("STEP 2: MODEL TRAINING")

    timings: list[tuple[str, float]] = []

    for should_run, label, module_name in MODEL_STEPS:
        if should_run != 1:
            print(f"\nSkipping model: {label}")
            continue

        runtime = run_command(
            [PYTHON_EXECUTABLE, "-m", module_name],
            f"{label} -> {module_name}",
        )
        timings.append((label, runtime))

    return timings


def run_model_comparison() -> float:
    """
    Run the final comparison script and rebuild all comparison plots.
    """
    print_header("STEP 3: MODEL COMPARISON")

    return run_command(
        [PYTHON_EXECUTABLE, "-m", COMPARISON_MODULE],
        COMPARISON_MODULE,
    )


def print_summary(
    data_processing_timings: list[tuple[str, float]],
    model_timings: list[tuple[str, float]],
    comparison_timing: float | None,
    total_runtime: float,
) -> None:
    """
    Print a short summary at the end.
    """
    print_header("PIPELINE SUMMARY")

    if data_processing_timings:
        print("Data processing:")
        for name, runtime in data_processing_timings:
            print(f"  - {name}: {format_runtime(runtime)}")
    else:
        print("Data processing: skipped")

    if model_timings:
        print("\nModels:")
        for name, runtime in model_timings:
            print(f"  - {name}: {format_runtime(runtime)}")
    else:
        print("\nModels: skipped")

    if comparison_timing is not None:
        print(f"\nModel comparison: {format_runtime(comparison_timing)}")
    else:
        print("\nModel comparison: skipped")

    print(f"\nTotal runtime: {format_runtime(total_runtime)}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_header("BIKE RENTAL PIPELINE")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {PYTHON_EXECUTABLE}")

    check_required_files()

    total_start = time.perf_counter()

    data_processing_timings: list[tuple[str, float]] = []
    model_timings: list[tuple[str, float]] = []
    comparison_timing: float | None = None

    if STEP_1_DATA_PROCESSING == 1:
        data_processing_timings = run_data_processing()
    else:
        print("\nSTEP 1: DATA PROCESSING was skipped.")

    selected_model_count = sum(1 for should_run, _, _ in MODEL_STEPS if should_run == 1)

    if selected_model_count > 0:
        model_timings = run_model_training()
    else:
        print("\nSTEP 2: MODEL TRAINING was skipped because all model flags are 0.")

    if STEP_3_MODEL_COMPARISON == 1:
        comparison_timing = run_model_comparison()
    else:
        print("\nSTEP 3: MODEL COMPARISON was skipped.")

    total_runtime = time.perf_counter() - total_start

    print_summary(
        data_processing_timings=data_processing_timings,
        model_timings=model_timings,
        comparison_timing=comparison_timing,
        total_runtime=total_runtime,
    )


if __name__ == "__main__":
    main()