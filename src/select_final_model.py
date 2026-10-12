"""Select and stage the Phase 8 final model using validation records only."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_MODEL = PROJECT_ROOT / "models" / "crop_disease_cnn.keras"
FINAL_MODEL = PROJECT_ROOT / "models" / "final_crop_disease_cnn.keras"
BASELINE_HISTORY = PROJECT_ROOT / "results" / "training_history.json"
EXPERIMENTS_CSV = PROJECT_ROOT / "results" / "hyperparameter_experiments.csv"
TUNER_SOURCE = PROJECT_ROOT / "src" / "tune.py"
SELECTION_PATH = PROJECT_ROOT / "results" / "final_model_selection.json"
PLANNED_EXPERIMENT_COUNT = 5


def main() -> None:
    """Choose by best validation accuracy, then best validation loss; copy safely."""
    required = (BASELINE_MODEL, BASELINE_HISTORY, EXPERIMENTS_CSV, TUNER_SOURCE)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required selection evidence missing: {missing}")

    baseline = json.loads(BASELINE_HISTORY.read_text(encoding="utf-8"))
    with EXPERIMENTS_CSV.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    by_name = {row["experiment_name"]: row for row in rows}
    if not rows:
        raise RuntimeError("No completed Phase 7 experiment records were found.")

    with TUNER_SOURCE.open(encoding="utf-8") as source_file:
        source = source_file.read()
    if 'TEST_DIR = PROJECT_ROOT / "data" / "test"' in source:
        raise RuntimeError("Phase 7 tuner references data/test; selection audit failed.")

    candidates: list[dict[str, Any]] = [
        {
            "model_name": "baseline",
            "experiment_name": "baseline_phase5",
            "architecture": "Baseline: Conv2D 32/64/128, Flatten, Dense128, Dropout0.5; no BatchNorm",
            "learning_rate": float(baseline["learning_rate"]),
            "batch_size": int(baseline["batch_size"]),
            "dropout": 0.5,
            "epochs_trained": int(baseline["epochs_completed"]),
            "best_validation_accuracy": float(baseline["best_validation_accuracy"]),
            "best_validation_loss": float(baseline["best_validation_loss"]),
            "best_accuracy_epoch": int(baseline["best_checkpoint_epoch"]),
            "best_loss_epoch": int(baseline["best_validation_loss_epoch"]),
            "early_stopping_triggered": bool(baseline["early_stopping_triggered"]),
            "overfitting_observed": True,
            "model_path": "models/crop_disease_cnn.keras",
            "selection_source": "results/training_history.json",
        }
    ]
    for row in rows:
        candidate_path = PROJECT_ROOT / Path(row["model_path"])
        history_path = PROJECT_ROOT / Path(row["history_path"])
        if not candidate_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint/history for {row['experiment_name']}.")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = int(row["epochs_trained"])
        candidates.append(
            {
                "model_name": row["experiment_name"],
                "experiment_name": row["experiment_name"],
                "architecture": row["architecture"],
                "learning_rate": float(row["learning_rate"]),
                "batch_size": int(row["batch_size"]),
                "dropout": float(row["dropout"]),
                "epochs_trained": epochs,
                "best_validation_accuracy": float(row["best_validation_accuracy"]),
                "best_validation_loss": float(row["best_validation_loss"]),
                "best_accuracy_epoch": int(row["best_accuracy_epoch"]),
                "best_loss_epoch": int(row["best_loss_epoch"]),
                "early_stopping_triggered": epochs < int(history["maximum_epochs"]),
                "overfitting_observed": row["overfitting_observed"].strip().lower() == "true",
                "training_time_seconds": float(row["training_time_seconds"]),
                "parameter_count": int(row["parameter_count"]),
                "model_path": row["model_path"],
                "selection_source": "results/hyperparameter_experiments.csv",
            }
        )

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["best_validation_accuracy"],
            -candidate["best_validation_loss"],
        ),
    )
    source_path = PROJECT_ROOT / Path(selected["model_path"])
    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, FINAL_MODEL)

    result = {
        "selection_criterion": "Highest best validation accuracy; lowest best validation loss breaks ties.",
        "selection_used_test_data": False,
        "phase7_completed_experiment_count": len(rows),
        "phase7_planned_experiment_count": PLANNED_EXPERIMENT_COUNT,
        "phase7_experiments_complete": len(rows) == PLANNED_EXPERIMENT_COUNT,
        "phase7_completion_warning": (
            "Only persisted completed experiment rows are ranked. Unfinished planned runs "
            "are excluded; do not interpret them as measured experiments."
            if len(rows) != PLANNED_EXPERIMENT_COUNT
            else None
        ),
        "selected": selected,
        "baseline_best_validation_accuracy": float(baseline["best_validation_accuracy"]),
        "baseline_best_validation_loss": float(baseline["best_validation_loss"]),
        "candidates": candidates,
        "final_model_path": "models/final_crop_disease_cnn.keras",
    }
    SELECTION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "final_model_path": result["final_model_path"]}, indent=2))


if __name__ == "__main__":
    main()
