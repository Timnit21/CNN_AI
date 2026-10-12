"""AgriAI: Training pipeline for tomato leaf disease classification.

This module trains the Convolutional Neural Network (CNN) using training and
validation data splits only. The held-out test split is deliberately not loaded.
It records per-epoch metrics (loss, accuracy, learning rate, and duration),
exports an epoch history CSV, saves training curves, and checkpoints the best model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    build_model,
)

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "train"
DEFAULT_VAL_DIR = PROJECT_ROOT / "data" / "validation"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "crop_disease_cnn.keras"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_HISTORY_CSV = DEFAULT_RESULTS_DIR / "training_history.csv"
DEFAULT_HISTORY_JSON = DEFAULT_RESULTS_DIR / "training_history.json"


def format_rel_path(path: Path) -> str:
    """Format path relative to PROJECT_ROOT if within it; otherwise return absolute string."""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return str(resolved).replace("\\", "/")


class EpochMetricsTracker(tf.keras.callbacks.Callback):
    """Callback to track precise per-epoch duration, learning rate, and metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_start_time: float = 0.0
        self.epoch_records: list[dict[str, Any]] = []

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        duration = time.perf_counter() - self.epoch_start_time
        logs = logs or {}
        
        # Get current optimizer learning rate safely across Keras versions
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
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train the AgriAI Tomato Disease CNN with configurable hyperparameters and callbacks."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training and validation batch size (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="EarlyStopping patience in epochs on val_loss (default: 5).",
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience in epochs on val_loss (default: 2).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability before final dense layer (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Target checkpoint filepath (default: {DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_HISTORY_CSV,
        help="Output filepath for epoch-by-epoch history CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_HISTORY_JSON,
        help="Output filepath for training summary JSON.",
    )
    parser.add_argument(
        "--output-plots-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save training curves and visualizations.",
    )
    parser.add_argument(
        "--enhanced-augmentation",
        action="store_true",
        help="Enable enhanced data augmentation (brightness, contrast, translation, vertical flip).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline_cnn",
        help="Descriptive name of the training run.",
    )
    return parser.parse_args()


def set_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_datasets(
    train_dir: Path = DEFAULT_TRAIN_DIR,
    val_dir: Path = DEFAULT_VAL_DIR,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], int, int]:
    """Load training and validation image directories.
    
    Test data is strictly not loaded or referenced by this function.
    """
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected training directory at '{train_dir}' and validation directory at '{val_dir}'."
        )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=None,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        class_names=train_dataset.class_names,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False,
    )

    if len(train_dataset.class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes; found {len(train_dataset.class_names)}: "
            f"{train_dataset.class_names}"
        )
    if train_dataset.class_names != validation_dataset.class_names:
        raise ValueError("Train and validation class order differs.")

    class_names = list(train_dataset.class_names)
    train_count = sum(1 for _ in train_dir.glob("*/*"))
    validation_count = sum(1 for _ in val_dir.glob("*/*"))
    
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(autotune)
    validation_dataset = validation_dataset.prefetch(autotune)
    return train_dataset, validation_dataset, class_names, train_count, validation_count


def save_training_plots(
    epoch_records: list[dict[str, Any]],
    output_dir: Path,
    title_prefix: str = "CNN Training",
) -> tuple[Path, Path]:
    """Save accuracy and loss curves from measured epoch records."""
    epochs = [r["epoch"] for r in epoch_records]
    train_acc = [r["accuracy"] for r in epoch_records]
    val_acc = [r["val_accuracy"] for r in epoch_records]
    train_loss = [r["loss"] for r in epoch_records]
    val_loss = [r["val_loss"] for r in epoch_records]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy Plot
    acc_path = output_dir / "training_accuracy.png"
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_acc, marker="o", color="#1f77b4", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, val_acc, marker="s", color="#ff7f0e", linewidth=2, label="Validation Accuracy")
    plt.title(f"{title_prefix}: Accuracy vs. Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=160)
    plt.close()

    # 2. Loss Plot
    loss_path = output_dir / "training_loss.png"
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_loss, marker="o", color="#1f77b4", linewidth=2, label="Training Loss (Sparse Cross-Entropy)")
    plt.plot(epochs, val_loss, marker="s", color="#d62728", linewidth=2, label="Validation Loss (Sparse Cross-Entropy)")
    plt.title(f"{title_prefix}: Loss vs. Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Negative Log-Likelihood)", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()

    return acc_path, loss_path


def export_history_csv(epoch_records: list[dict[str, Any]], csv_path: Path) -> None:
    """Export epoch-by-epoch metrics to a structured CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
        "epoch_duration_seconds",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in epoch_records:
            writer.writerow({k: record.get(k, "") for k in fields})


def main() -> None:
    """Execute training pipeline with full logging, callback management, and artifact persistence."""
    args = parse_args()
    set_seeds(args.seed)

    print("=" * 80)
    print(f" AgriAI CNN Training Pipeline: '{args.name}'")
    print("=" * 80)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Available Devices: {tf.config.list_physical_devices()}")
    print(f"Initial Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size} | Max Epochs: {args.epochs}")
    print(f"EarlyStopping Patience: {args.patience} | ReduceLR Patience: {args.reduce_lr_patience}")
    print(f"Augmentation Mode: {'Enhanced' if args.enhanced_augmentation else 'Baseline'}")
    print(f"Target Checkpoint: {args.output_model}")
    print("Test data is strictly isolated and NOT loaded during training.")
    print("-" * 80)

    train_dataset, val_dataset, class_names, train_count, val_count = make_datasets(
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Training Set Size: {train_count} images")
    print(f"Validation Set Size: {val_count} images")
    print(f"Class Mapping: {dict(enumerate(class_names))}")
    print("-" * 80)

    # Build model
    model = build_model(
        num_classes=len(class_names),
        learning_rate=args.lr,
        use_enhanced_augmentation=args.enhanced_augmentation,
        dropout_rate=args.dropout,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_plots_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    tracker = EpochMetricsTracker()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=args.reduce_lr_patience,
        min_lr=1e-6,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(args.output_model),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    callbacks = [tracker, reduce_lr, early_stopping, checkpoint]

    start_time = time.perf_counter()
    print("\nStarting Model Fit...\n")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    total_training_time = time.perf_counter() - start_time

    records = tracker.epoch_records
    if not records:
        raise RuntimeError("No epoch records were captured during training.")

    # Export history CSV
    export_history_csv(records, args.output_csv)

    # Save visualization plots
    acc_plot, loss_plot = save_training_plots(
        records,
        args.output_plots_dir,
        title_prefix=args.name,
    )

    # Compute best epoch metrics
    best_val_acc_idx = int(np.argmax([r["val_accuracy"] for r in records]))
    best_val_loss_idx = int(np.argmin([r["val_loss"] for r in records]))
    best_epoch_acc = records[best_val_acc_idx]["epoch"]
    best_epoch_loss = records[best_val_loss_idx]["epoch"]
    stopped_early = len(records) < args.epochs

    summary = {
        "experiment_name": args.name,
        "dataset": "PlantVillage tomato 3-class subset",
        "class_names": class_names,
        "training_images": train_count,
        "validation_images": val_count,
        "test_data_loaded": False,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "epochs_completed": len(records),
        "random_seed": args.seed,
        "optimizer": "Adam",
        "initial_learning_rate": args.lr,
        "loss_function": "sparse_categorical_crossentropy",
        "augmentation_mode": "enhanced" if args.enhanced_augmentation else "baseline",
        "dropout_rate": args.dropout,
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": args.patience,
        "early_stopping_triggered": stopped_early,
        "early_stopping_epoch": len(records) if stopped_early else None,
        "reduce_lr_monitor": "val_loss",
        "reduce_lr_patience": args.reduce_lr_patience,
        "reduce_lr_factor": 0.5,
        "checkpoint_monitor": "val_accuracy",
        "checkpoint_path": format_rel_path(args.output_model),
        "best_accuracy_epoch": best_epoch_acc,
        "best_validation_accuracy": records[best_val_acc_idx]["val_accuracy"],
        "best_loss_epoch": best_epoch_loss,
        "best_validation_loss": records[best_val_loss_idx]["val_loss"],
        "final_training_accuracy": records[-1]["accuracy"],
        "final_validation_accuracy": records[-1]["val_accuracy"],
        "final_training_loss": records[-1]["loss"],
        "final_validation_loss": records[-1]["val_loss"],
        "total_training_duration_seconds": round(total_training_time, 2),
        "average_epoch_duration_seconds": round(
            float(np.mean([r["epoch_duration_seconds"] for r in records])), 2
        ),
        "history_csv_path": format_rel_path(args.output_csv),
        "plots": {
            "accuracy_plot": format_rel_path(acc_plot),
            "loss_plot": format_rel_path(loss_plot),
        },
        "cost_function_note": (
            "The sparse categorical cross-entropy cost function penalizes probabilistic error "
            "as L = -sum(log(p_true)). It reflects confidence divergence rather than just discrete errors."
        ),
    }

    # Save summary JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" TRAINING SUMMARY & OUTCOMES")
    print("=" * 80)
    print(f"Total Epochs Completed: {len(records)} / {args.epochs}")
    print(f"Early Stopping Triggered: {stopped_early} (at Epoch {len(records)})")
    print(f"Best Validation Accuracy: {summary['best_validation_accuracy']:.4%} (Epoch {best_epoch_acc})")
    print(f"Best Validation Loss: {summary['best_validation_loss']:.6f} (Epoch {best_epoch_loss})")
    print(f"Final Training Accuracy: {summary['final_training_accuracy']:.4%}")
    print(f"Final Validation Accuracy: {summary['final_validation_accuracy']:.4%}")
    print(f"Total Training Duration: {total_training_time:.2f}s (~{total_training_time/60:.1f} mins)")
    print(f"Artifacts Saved:")
    print(f"  - Model Checkpoint: {summary['checkpoint_path']}")
    print(f"  - History CSV:      {summary['history_csv_path']}")
    print(f"  - Summary JSON:     {format_rel_path(args.output_json)}")
    print(f"  - Accuracy Plot:    {summary['plots']['accuracy_plot']}")
    print(f"  - Loss Plot:        {summary['plots']['loss_plot']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
