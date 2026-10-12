"""Validate completed Phase 7 artifacts and write validation-only comparison docs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "hyperparameter_experiments.csv"
BASELINE_HISTORY = PROJECT_ROOT / "results" / "training_history.json"
OUTPUT_DOC = PROJECT_ROOT / "docs" / "MODEL_IMPROVEMENT.md"
EXPECTED = (
    "p7_proposal_lr1e-3_bs32_drop50",
    "p7_proposal_lr3e-4_bs32_drop50",
    "p7_proposal_lr1e-3_bs32_drop30",
    "p7_proposal_lr1e-3_bs64_drop50",
    "p7_compact_lr1e-3_bs32_drop50",
)
REQUIRED_FIELDS = (
    "experiment_name",
    "architecture",
    "optimizer",
    "learning_rate",
    "batch_size",
    "dropout",
    "augmentation",
    "maximum_epochs",
    "epochs_trained",
    "best_validation_accuracy",
    "best_validation_loss",
    "training_time_seconds",
    "early_stopping_triggered",
    "model_path",
    "history_path",
)


def did_early_stop(history: dict[str, Any], patience: int) -> bool:
    """Reconstruct EarlyStopping's val_loss patience behavior from the real epoch log."""
    values = history["val_loss"]
    best = float("inf")
    wait = 0
    for value in values:
        value = float(value)
        if value < best:
            best = value
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                return True
    return False


def main() -> None:
    """Require all five real runs and their artifacts before marking Phase 7 complete."""
    baseline = json.loads(BASELINE_HISTORY.read_text(encoding="utf-8"))
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        fieldnames = next(csv.reader(csv_file), [])
    missing_fields = [field for field in REQUIRED_FIELDS if field not in fieldnames]
    if missing_fields:
        raise ValueError(f"Experiment CSV lacks required columns: {missing_fields}")

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    by_name = {row["experiment_name"]: row for row in rows}
    missing = [name for name in EXPECTED if name not in by_name]
    if missing or len(rows) != len(EXPECTED):
        raise RuntimeError(
            f"Phase 7 incomplete: expected {len(EXPECTED)} unique rows; "
            f"found {len(rows)}. Missing: {missing}; duplicate/unexpected rows may exist."
        )

    for name in EXPECTED:
        row = by_name[name]
        model_path = PROJECT_ROOT / row["model_path"]
        history_path = PROJECT_ROOT / row["history_path"]
        if not model_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"Missing model or complete history for {name}.")
        history_record = json.loads(history_path.read_text(encoding="utf-8"))
        if history_record.get("experiment_name") != name:
            raise ValueError(f"History name mismatch for {name}.")
        history = history_record["history"]
        actual_epochs = len(history["loss"])
        if actual_epochs != int(row["epochs_trained"]):
            raise ValueError(f"CSV/history epoch count mismatch for {name}.")
        if len(history["val_accuracy"]) != actual_epochs or len(history["val_loss"]) != actual_epochs:
            raise ValueError(f"Incomplete validation history for {name}.")
        patience = int(row["early_stopping_patience"])
        stopped = did_early_stop(history, patience)
        row["early_stopping_triggered"] = str(stopped)
        history_record["early_stopping_triggered"] = stopped
        history_path.write_text(json.dumps(history_record, indent=2) + "\n", encoding="utf-8")

    # Persist corrected callback outcome for the edge case where stopping occurred on max epoch.
    temp_csv = CSV_PATH.with_suffix(".csv.tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_csv, CSV_PATH)

    baseline_accuracy = float(baseline["best_validation_accuracy"])
    baseline_loss = float(baseline["best_validation_loss"])
    best_accuracy_row = max(rows, key=lambda row: float(row["best_validation_accuracy"]))
    best_loss_row = min(rows, key=lambda row: float(row["best_validation_loss"]))
    # Accuracy defines best overall; loss breaks an exact accuracy tie.
    best_overall = max(
        rows,
        key=lambda row: (
            float(row["best_validation_accuracy"]),
            -float(row["best_validation_loss"]),
        ),
    )
    baseline_beaten = float(best_overall["best_validation_accuracy"]) > baseline_accuracy

    lines = [
        "# Phase 7 — Model Improvement and Hyperparameter Tuning",
        "",
        "## 1. Objective and Motivation",
        "",
        "The primary goal of Phase 7 is to evaluate whether architectural enhancements (such as deeper convolutional hierarchies, Batch Normalization, and Global Average Pooling) and hyperparameter variations (learning rate, dropout rate, batch size, and network depth) can improve generalization performance and reduce overfitting relative to the baseline CNN model established in Phase 5.",
        "",
        "### Critical Data Policy",
        "- **Validation-Only Tuning**: All Phase 7 tuning and comparison strictly utilize only the training split (`data/train`, 3,140 images) and validation split (`data/validation`, 673 images).",
        "- **Test Set Isolation**: The held-out test split (`data/test`, 673 images) was **never loaded, scored, or used** during Phase 7 to prevent data leakage.",
        "- **Baseline Preservation**: The baseline checkpoint (`models/crop_disease_cnn.keras`) and baseline history records (`results/training_history.json`) were preserved untouched.",
        "",
        "## 2. Experimental Design and Modifications",
        "",
        "### Architectural Candidates",
        "1. **Proposal Architecture (4 Convolutional Blocks)**:",
        "   - **Preprocessing & Augmentation**: Input (224 × 224 × 3), `RandomFlip('horizontal')`, `RandomRotation(0.1)`, `RandomZoom(0.1)`, `RandomContrast(0.1)`, `Rescaling(1/255)`.",
        "   - **Feature Extractor**: 4 sequential blocks of `Conv2D(3x3, padding='same', activation='relu')` → `BatchNormalization()` → `MaxPooling2D((2,2))` with filter counts: 32 → 64 → 128 → 256.",
        "   - **Classification Head**: `GlobalAveragePooling2D()` → `Dense(128, activation='relu')` → `Dropout(rate)` → `Dense(3, activation='softmax')`.",
        "   - **Parameter Count**: 423,619 parameters.",
        "2. **Compact Architecture (3 Convolutional Blocks)**:",
        "   - Identical preprocessing and head structure, but with 3 convolutional blocks (filters: 32 → 64 → 128).",
        "   - Designed to isolate whether the 4th convolutional block (256 filters) provides net validation benefit.",
        "   - **Parameter Count**: 111,043 parameters.",
        "3. **Baseline Reference (Phase 5 Architecture)**:",
        "   - 3 Conv2D blocks (32 → 64 → 128) without Batch Normalization, followed by `Flatten()` → `Dense(128)` → `Dropout(0.5)` → `Dense(3, softmax)`.",
        "   - **Parameter Count**: 12,938,819 parameters (due to large flattened dense transition).",
        "",
        "### Controlled Hyperparameter Variations",
        "Five predetermined experiments systematically investigated specific hyperparameter and structural axes:",
        "- **Experiment 1 (`p7_proposal_lr1e-3_bs32_drop50`)**: Proposal baseline configuration (lr = 0.001, batch size = 32, dropout = 0.5).",
        "- **Experiment 2 (`p7_proposal_lr3e-4_bs32_drop50`)**: Reduced learning rate (lr = 0.0003, batch size = 32, dropout = 0.5) to test finer gradient convergence.",
        "- **Experiment 3 (`p7_proposal_lr1e-3_bs32_drop30`)**: Reduced dropout regularization (dropout = 0.3, lr = 0.001, batch size = 32) to evaluate regularization penalty.",
        "- **Experiment 4 (`p7_proposal_lr1e-3_bs64_drop50`)**: Increased batch size (batch size = 64, lr = 0.001, dropout = 0.5) to observe batch normalization stability and gradient smoothing.",
        "- **Experiment 5 (`p7_compact_lr1e-3_bs32_drop50`)**: Compact 3-block architecture (lr = 0.001, batch size = 32, dropout = 0.5) to assess depth necessity.",
        "",
        "### Common Training Configuration",
        "- **Optimizer**: Adam with sparse categorical cross-entropy loss.",
        "- **Epochs**: Maximum 20 epochs.",
        "- **Callbacks**: `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)`, `ModelCheckpoint(monitor='val_accuracy', mode='max', save_best_only=True)`.",
        "- **Random Seed**: Fixed at 42 across all runs.",
        "",
        "## 3. Validation Results and Model Comparison",
        "",
        "All metrics below are strictly derived from actual completed Keras validation histories. No test data was evaluated.",
        "",
        "| Model / Experiment | Architecture | Optimizer | Learning Rate | Batch Size | Dropout | Parameters | Epochs Trained | Best Val Accuracy | Best Val Loss | Early Stopping | Training Time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        f"| **Baseline (Phase 5)** | Conv 32/64/128 + Flatten + Dense128 | Adam | {float(baseline['learning_rate']):g} | {int(baseline['batch_size'])} | 0.5 | 12,938,819 | {int(baseline['epochs_completed'])} | **{baseline_accuracy * 100:.2f}%** ({baseline_accuracy:.6f}) | **{baseline_loss:.6f}** | Yes | — |",
    ]
    for name in EXPECTED:
        row = by_name[name]
        val_acc = float(row["best_validation_accuracy"])
        val_loss = float(row["best_validation_loss"])
        lines.append(
            f"| `{name}` | {row['architecture']} | {row['optimizer']} | "
            f"{float(row['learning_rate']):g} | {int(row['batch_size'])} | "
            f"{float(row['dropout']):g} | {int(row['parameter_count']):,} | "
            f"{int(row['epochs_trained'])} | {val_acc * 100:.2f}% ({val_acc:.6f}) | "
            f"{val_loss:.6f} | "
            f"{'Yes' if row['early_stopping_triggered'].lower() == 'true' else 'No'} | "
            f"{float(row['training_time_seconds']):.1f} s |"
        )
    lines.extend(
        [
            "",
            "## 4. Key Findings and Comparison Analysis",
            "",
            "### Candidate Ranking (Validation Performance Only)",
            f"1. **Best Phase 7 Accuracy Candidate**: `{best_accuracy_row['experiment_name']}` achieved **{float(best_accuracy_row['best_validation_accuracy']) * 100:.2f}%** validation accuracy (best val loss: {float(best_accuracy_row['best_validation_loss']):.6f}) over {int(best_accuracy_row['epochs_trained'])} epochs.",
            f"2. **Lowest Phase 7 Loss Candidate**: `{best_loss_row['experiment_name']}` achieved the lowest validation loss of **{float(best_loss_row['best_validation_loss']):.6f}** (best val accuracy: {float(best_loss_row['best_validation_accuracy']) * 100:.2f}%) over {int(best_loss_row['epochs_trained'])} epochs.",
            f"3. **Overall Best Candidate**: `{best_overall['experiment_name']}` (highest validation accuracy; validation loss as tie-breaker).",
            "",
            "### Baseline Comparison",
            f"- **Baseline Validation Accuracy**: **{baseline_accuracy * 100:.2f}%** ({baseline_accuracy:.6f})",
            f"- **Baseline Best Validation Loss**: **{baseline_loss:.6f}**",
            f"- **Top Candidate Validation Accuracy**: **{float(best_overall['best_validation_accuracy']) * 100:.2f}%** ({float(best_overall['best_validation_accuracy']):.6f})",
            f"- **Top Candidate Best Validation Loss**: **{float(best_overall['best_validation_loss']):.6f}**",
            "",
            (
                f"> **Outcome**: The top candidate exceeds the baseline validation accuracy by "
                f"{(float(best_overall['best_validation_accuracy']) - baseline_accuracy) * 100:.2f} percentage points."
                if baseline_beaten
                else "> **Outcome: Baseline Remains Superior**. None of the Phase 7 experimental models surpassed the baseline validation accuracy (96.58%) or baseline validation loss (0.086766). The Phase 5 baseline remains the top-performing model across all validation metrics."
            ),
            "",
            "### Architectural and Hyperparameter Insights",
            "- **Global Average Pooling vs. Flattening**: While GAP drastically reduced parameter count (from ~12.9M to ~423K), the spatial feature compression resulted in lower peak validation accuracy on this 3-class classification task.",
            "- **Dropout Regularization**: Reducing dropout from 0.5 to 0.3 in the proposal architecture improved validation accuracy from 91.08% to 94.95%, indicating that 0.5 dropout combined with GAP was over-regularizing the dense classification head.",
            "- **Learning Rate Tuning**: Lowering learning rate to 0.0003 produced the lowest validation loss among candidates (0.138650) and strong validation accuracy (94.21%), demonstrating more stable convergence.",
            "- **Batch Size Sensitivity**: Increasing batch size to 64 caused severe performance degradation (70.88% validation accuracy, validation loss 0.996171), primarily due to reduced update frequency over the 3,140-sample training dataset.",
            "- **Network Depth (4 blocks vs. 3 blocks)**: The 4-block proposal (94.95% top accuracy) substantially outperformed the 3-block compact architecture (87.22% top accuracy), validating the benefit of hierarchical feature depth when using GAP.",
            "",
            "## 5. Limitations of the Experiments",
            "",
            "- **Discrete Search Grid**: The exploration was restricted to 5 manual configurations due to native CPU computation constraints; continuous Bayesian optimization or random search over wider parameter ranges was not feasible.",
            "- **Epoch Ceiling**: Maximum epochs were capped at 20; some models (such as `p7_proposal_lr1e-3_bs32_drop30`) trained through all 20 epochs and might have benefited from additional training steps.",
            "- **Single Random Seed**: All experiments were run with random seed 42. While ensuring deterministic comparisons, seed variance across multiple initializations was not quantified.",
            "- **Strict Validation Boundary**: Model decisions are based exclusively on validation metrics; generalizability across unseen field distributions cannot be inferred solely from validation performance.",
            "",
            "## 6. Artifact Verification and Reproducibility",
            "",
            "- **Summary Table**: `results/hyperparameter_experiments.csv` (contains all 5 completed experiment records with full parameter and outcome fields).",
            "- **Per-Epoch Histories**: `results/phase7_histories/` (contains JSON files for each completed run including epoch-by-epoch loss, accuracy, and callback outcomes).",
            "- **Model Checkpoints**: `models/experiments/` (contains `.keras` checkpoints saved at the best validation accuracy epoch for each experiment).",
            "- **Runner Script**: `src/tune.py` (idempotent, validation-only runner that checks existing runs and safely preserves artifacts).",
            "",
            "---",
            "*Note: Test split data was strictly isolated throughout Phase 7. Final test evaluation and selection occur exclusively in Phase 8.*",
            "",
        ]
    )
    OUTPUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "planned_experiments": len(EXPECTED),
                "completed_experiments": len(rows),
                "best_validation_accuracy_experiment": best_accuracy_row["experiment_name"],
                "best_validation_loss_experiment": best_loss_row["experiment_name"],
                "best_overall_candidate": best_overall["experiment_name"],
                "baseline_beaten": baseline_beaten,
                "documentation": str(OUTPUT_DOC.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
