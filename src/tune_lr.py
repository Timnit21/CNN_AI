"""Controlled learning rate experimentation for AgriAI baseline CNN.

This script runs controlled training experiments across candidate learning rates
(0.001, 0.0005, 0.0001) while holding all other architectural and dataset parameters
strictly identical (Batch size=32, Seed=42, Loss=sparse_categorical_crossentropy).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, build_model
from train import make_datasets, set_seeds, export_history_csv, save_training_plots

EXPERIMENT_RATES = [0.001, 0.0005, 0.0001]
EXPERIMENTS_DIR = PROJECT_ROOT / "models" / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
LR_COMPARISON_CSV = RESULTS_DIR / "learning_rate_experiments.csv"


class EpochMetricsTracker(tf.keras.callbacks.Callback):
    """Callback to record epoch duration and metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_start_time: float = 0.0
        self.epoch_records: list[dict[str, Any]] = []

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        duration = time.perf_counter() - self.epoch_start_time
        logs = logs or {}
        lr = float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer.learning_rate, "numpy") else float(self.model.optimizer.learning_rate)
        
        record = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "learning_rate": lr,
            "epoch_duration_seconds": round(duration, 3),
        }
        self.epoch_records.append(record)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for controlled learning rate tuning."""
    parser = argparse.ArgumentParser(description="Run controlled learning rate experiments.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs per run (default: 15).")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (default: 4).")
    parser.add_argument("--rates", nargs="+", type=float, default=EXPERIMENT_RATES, help="Learning rates to test.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def run_single_lr_experiment(
    lr: float,
    max_epochs: int = 15,
    patience: int = 4,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one model with a specific learning rate under controlled conditions."""
    set_seeds(seed)
    train_dataset, val_dataset, class_names, train_count, val_count = make_datasets(
        batch_size=batch_size,
        seed=seed,
    )

    exp_name = f"baseline_cnn_lr_{lr:g}"
    model_path = EXPERIMENTS_DIR / f"{exp_name}.keras"
    csv_path = RESULTS_DIR / f"{exp_name}_history.csv"
    json_path = RESULTS_DIR / f"{exp_name}_summary.json"

    model = build_model(
        num_classes=len(class_names),
        learning_rate=lr,
        use_enhanced_augmentation=False,
    )

    tracker = EpochMetricsTracker()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    print(f"\n" + "=" * 60)
    print(f" Starting Controlled Experiment: LR = {lr}")
    print("=" * 60)

    start_time = time.perf_counter()
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        callbacks=[tracker, reduce_lr, early_stopping, checkpoint],
        verbose=2,
    )
    duration = time.perf_counter() - start_time

    records = tracker.epoch_records
    export_history_csv(records, csv_path)
    save_training_plots(records, RESULTS_DIR, title_prefix=f"Baseline CNN (LR={lr})")

    best_val_acc_idx = int(np.argmax([r["val_accuracy"] for r in records]))
    best_val_loss_idx = int(np.argmin([r["val_loss"] for r in records]))

    summary = {
        "experiment_name": exp_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "maximum_epochs": max_epochs,
        "epochs_trained": len(records),
        "best_validation_accuracy": records[best_val_acc_idx]["val_accuracy"],
        "best_accuracy_epoch": records[best_val_acc_idx]["epoch"],
        "best_validation_loss": records[best_val_loss_idx]["val_loss"],
        "best_loss_epoch": records[best_val_loss_idx]["epoch"],
        "final_training_accuracy": records[-1]["accuracy"],
        "final_validation_accuracy": records[-1]["val_accuracy"],
        "final_training_loss": records[-1]["loss"],
        "final_validation_loss": records[-1]["val_loss"],
        "early_stopping_triggered": len(records) < max_epochs,
        "training_time_seconds": round(duration, 2),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "history_csv": str(csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    """Run all configured controlled learning rate experiments and compile summary table."""
    args = parse_args()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for lr in args.rates:
        summary = run_single_lr_experiment(
            lr=lr,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        summaries.append(summary)

    # Compile into comparative CSV
    fields = [
        "experiment_name",
        "learning_rate",
        "batch_size",
        "epochs_trained",
        "best_validation_accuracy",
        "best_accuracy_epoch",
        "best_validation_loss",
        "best_loss_epoch",
        "final_training_accuracy",
        "final_validation_accuracy",
        "training_time_seconds",
        "model_path",
    ]
    with LR_COMPARISON_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fields})

    print("\n" + "=" * 80)
    print(" CONTROLLED LEARNING RATE EXPERIMENTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':25s} {'LR':>8s} {'Epochs':>8s} {'Best Val Acc':>14s} {'Best Val Loss':>14s} {'Time (s)':>10s}")
    print("-" * 80)
    for s in summaries:
        print(
            f"{s['experiment_name']:25s} {s['learning_rate']:8g} {s['epochs_trained']:8d} "
            f"{s['best_validation_accuracy']:14.4%} {s['best_validation_loss']:14.6f} {s['training_time_seconds']:10.1f}"
        )
    print("=" * 80)
    print(f"Summary table saved to {LR_COMPARISON_CSV.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
