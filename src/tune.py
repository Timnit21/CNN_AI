"""Run Phase 7 CNN experiments using training and validation data only."""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
MODEL_DIR = PROJECT_ROOT / "models" / "experiments"
HISTORY_DIR = PROJECT_ROOT / "results" / "phase7_histories"
CSV_PATH = PROJECT_ROOT / "results" / "hyperparameter_experiments.csv"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CLASSES = 3
RANDOM_SEED = 42
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 2


@dataclass(frozen=True)
class Experiment:
    """One validation-only experiment configuration."""

    name: str
    architecture: str
    filters: tuple[int, ...]
    learning_rate: float
    batch_size: int
    dropout: float


EXPERIMENTS = (
    Experiment(
        name="p7_proposal_lr1e-3_bs32_drop50",
        architecture="Proposal: Conv-BN-MaxPool blocks 32/64/128/256, GAP, Dense128",
        filters=(32, 64, 128, 256),
        learning_rate=0.001,
        batch_size=32,
        dropout=0.5,
    ),
    Experiment(
        name="p7_proposal_lr3e-4_bs32_drop50",
        architecture="Proposal: Conv-BN-MaxPool blocks 32/64/128/256, GAP, Dense128",
        filters=(32, 64, 128, 256),
        learning_rate=0.0003,
        batch_size=32,
        dropout=0.5,
    ),
    Experiment(
        name="p7_proposal_lr1e-3_bs32_drop30",
        architecture="Proposal: Conv-BN-MaxPool blocks 32/64/128/256, GAP, Dense128",
        filters=(32, 64, 128, 256),
        learning_rate=0.001,
        batch_size=32,
        dropout=0.3,
    ),
    Experiment(
        name="p7_proposal_lr1e-3_bs64_drop50",
        architecture="Proposal: Conv-BN-MaxPool blocks 32/64/128/256, GAP, Dense128",
        filters=(32, 64, 128, 256),
        learning_rate=0.001,
        batch_size=64,
        dropout=0.5,
    ),
    Experiment(
        name="p7_compact_lr1e-3_bs32_drop50",
        architecture="Compact: Conv-BN-MaxPool blocks 32/64/128, GAP, Dense128",
        filters=(32, 64, 128),
        learning_rate=0.001,
        batch_size=32,
        dropout=0.5,
    ),
)

CSV_FIELDS = (
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
    "early_stopping_triggered",
    "early_stopping_monitor",
    "early_stopping_patience",
    "reduce_lr_monitor",
    "reduce_lr_factor",
    "reduce_lr_patience",
    "checkpoint_monitor",
    "random_seed",
    "overfitting_observed",
    "training_time_seconds",
    "best_accuracy_epoch",
    "best_loss_epoch",
    "parameter_count",
    "model_path",
    "history_path",
)

AUGMENTATION = (
    "RandomFlip(horizontal); RandomRotation(0.1); RandomZoom(0.1); "
    "RandomContrast(0.1); Rescaling(1/255)"
)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Reset random generators before each independent run."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_datasets(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load only training and validation images, preserving one class order."""
    if not TRAIN_DIR.is_dir() or not VALIDATION_DIR.is_dir():
        raise FileNotFoundError("Expected data/train and data/validation directories.")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="int",
        class_names=train_dataset.class_names,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False,
    )
    if len(train_dataset.class_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, got {train_dataset.class_names}.")
    if train_dataset.class_names != validation_dataset.class_names:
        raise ValueError("Train and validation class order differs.")

    autotune = tf.data.AUTOTUNE
    return (
        train_dataset.prefetch(autotune),
        validation_dataset.prefetch(autotune),
        list(train_dataset.class_names),
    )


def build_model(experiment: Experiment) -> tf.keras.Model:
    """Build an experiment model from the proposal or compact block schedule."""
    layers: list[tf.keras.layers.Layer] = [
        tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="image"),
        tf.keras.layers.RandomFlip("horizontal", name="random_flip"),
        tf.keras.layers.RandomRotation(0.1, name="random_rotation"),
        tf.keras.layers.RandomZoom(0.1, name="random_zoom"),
        tf.keras.layers.RandomContrast(0.1, name="random_contrast"),
        tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling"),
    ]
    for index, filters in enumerate(experiment.filters, start=1):
        layers.extend(
            [
                tf.keras.layers.Conv2D(
                    filters,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    name=f"conv2d_{filters}",
                ),
                tf.keras.layers.BatchNormalization(name=f"batch_normalization_{index}"),
                tf.keras.layers.MaxPooling2D((2, 2), name=f"max_pooling_{index}"),
            ]
        )
    layers.extend(
        [
            tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
            tf.keras.layers.Dropout(experiment.dropout, name="dropout"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class_probabilities"),
        ]
    )
    model = tf.keras.Sequential(layers, name=experiment.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=experiment.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def overfitting_observed(history: dict[str, list[float]]) -> bool:
    """Apply a stated late-run divergence rule to the recorded curves."""
    best_loss = min(history["val_loss"])
    best_accuracy = max(history["val_accuracy"])
    final_val_loss = history["val_loss"][-1]
    final_val_accuracy = history["val_accuracy"][-1]
    final_train_accuracy = history["accuracy"][-1]
    loss_divergence = (
        final_val_loss > best_loss * 1.25 and final_train_accuracy > final_val_accuracy
    )
    accuracy_regression = best_accuracy - final_val_accuracy >= 0.05
    return loss_divergence or accuracy_regression


def run_experiment(
    experiment: Experiment, model_path: Path, history_path: Path
) -> dict[str, Any]:
    """Train one model, checkpoint its best validation-accuracy weights, and log results."""
    set_seeds()
    train_dataset, validation_dataset, class_names = make_datasets(experiment.batch_size)
    model = build_model(experiment)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    print(f"\n=== {experiment.name} ===", flush=True)
    print(
        f"Architecture={experiment.architecture}; lr={experiment.learning_rate}; "
        f"batch={experiment.batch_size}; dropout={experiment.dropout}; seed={RANDOM_SEED}",
        flush=True,
    )
    started = time.perf_counter()
    history_object = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    training_time = time.perf_counter() - started
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    best_accuracy_epoch = int(np.argmax(history["val_accuracy"])) + 1
    best_loss_epoch = int(np.argmin(history["val_loss"])) + 1
    record: dict[str, Any] = {
        "experiment_name": experiment.name,
        "architecture": experiment.architecture,
        "optimizer": "Adam",
        "learning_rate": experiment.learning_rate,
        "batch_size": experiment.batch_size,
        "dropout": experiment.dropout,
        "augmentation": AUGMENTATION,
        "maximum_epochs": MAX_EPOCHS,
        "epochs_trained": len(history["loss"]),
        "best_validation_accuracy": max(history["val_accuracy"]),
        "best_validation_loss": min(history["val_loss"]),
        "early_stopping_triggered": len(history["loss"]) < MAX_EPOCHS,
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_monitor": "val_loss",
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "checkpoint_monitor": "val_accuracy",
        "random_seed": RANDOM_SEED,
        "overfitting_observed": overfitting_observed(history),
        "training_time_seconds": round(training_time, 3),
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_loss_epoch": best_loss_epoch,
        "parameter_count": model.count_params(),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "history_path": str(history_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "class_names": class_names,
        "history": history,
    }
    history_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def append_csv_record(record: dict[str, Any]) -> None:
    """Append a completed run after its model and full history have been saved."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record[field] for field in CSV_FIELDS})


def migrate_existing_records() -> list[dict[str, str]]:
    """Add factual run metadata to the CSV/history while preserving old metric values."""
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        old_rows = list(csv.DictReader(csv_file))
    if not old_rows:
        return []

    rows: list[dict[str, str]] = []
    for old_row in old_rows:
        row = {field: old_row.get(field, "") for field in CSV_FIELDS}
        history_path = PROJECT_ROOT / Path(row["history_path"])
        model_path = PROJECT_ROOT / Path(row["model_path"])
        if not history_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                f"Completed record lacks its saved model/history: {row['experiment_name']}"
            )
        history_record = json.loads(history_path.read_text(encoding="utf-8"))
        derived: dict[str, Any] = {
            "optimizer": "Adam",
            "augmentation": AUGMENTATION,
            "maximum_epochs": int(history_record.get("maximum_epochs", MAX_EPOCHS)),
            "early_stopping_triggered": (
                int(row["epochs_trained"])
                < int(history_record.get("maximum_epochs", MAX_EPOCHS))
            ),
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": int(
                history_record.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)
            ),
            "reduce_lr_monitor": "val_loss",
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": int(
                history_record.get("reduce_lr_patience", REDUCE_LR_PATIENCE)
            ),
            "checkpoint_monitor": "val_accuracy",
            "random_seed": int(history_record.get("random_seed", RANDOM_SEED)),
        }
        for field, value in derived.items():
            if not row[field]:
                row[field] = str(value)
        # Enrich the existing JSON with configuration metadata, retaining its measured history.
        for field, value in derived.items():
            history_record.setdefault(field, value)
        history_record.setdefault("early_stopping_triggered", derived["early_stopping_triggered"])
        history_record.setdefault("early_stopping_monitor", "val_loss")
        history_record.setdefault("reduce_lr_monitor", "val_loss")
        history_record.setdefault("reduce_lr_factor", 0.5)
        history_record.setdefault("checkpoint_monitor", "val_accuracy")
        history_path.write_text(json.dumps(history_record, indent=2) + "\n", encoding="utf-8")
        rows.append(row)

    temporary_csv = CSV_PATH.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, CSV_PATH)
    return rows


def unique_run_paths(experiment: Experiment) -> tuple[Path, Path]:
    """Use the canonical paths when free; otherwise preserve artifacts with a fresh suffix."""
    model_path = MODEL_DIR / f"{experiment.name}.keras"
    history_path = HISTORY_DIR / f"{experiment.name}.json"
    if not model_path.exists() and not history_path.exists():
        return model_path, history_path

    suffix = 1
    while True:
        model_path = MODEL_DIR / f"{experiment.name}_rerun{suffix}.keras"
        history_path = HISTORY_DIR / f"{experiment.name}_rerun{suffix}.json"
        if not model_path.exists() and not history_path.exists():
            return model_path, history_path
        suffix += 1


def main() -> None:
    """Complete missing planned runs without reading test data or overwriting artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    existing_rows = migrate_existing_records()
    completed_names = [row["experiment_name"] for row in existing_rows]
    if len(completed_names) != len(set(completed_names)):
        raise ValueError("Duplicate completed experiment names already exist in the CSV.")
    unknown_names = set(completed_names).difference(experiment.name for experiment in EXPERIMENTS)
    if unknown_names:
        raise ValueError(f"CSV contains unplanned experiment(s): {sorted(unknown_names)}")

    print(f"TensorFlow {tf.__version__}; devices: {tf.config.list_physical_devices()}")
    print(f"Training and validation directories only; maximum epochs per run: {MAX_EPOCHS}.")
    for experiment in EXPERIMENTS:
        if experiment.name in completed_names:
            print(f"Skipping persisted completed run: {experiment.name}", flush=True)
            continue
        model_path, history_path = unique_run_paths(experiment)
        try:
            record = run_experiment(experiment, model_path, history_path)
            append_csv_record(record)
            completed_names.append(experiment.name)
            print(
                f"Completed {experiment.name}: epochs={record['epochs_trained']}, "
                f"best val_accuracy={record['best_validation_accuracy']:.6f}, "
                f"best val_loss={record['best_validation_loss']:.6f}, "
                f"time={record['training_time_seconds']:.1f}s",
                flush=True,
            )
        finally:
            tf.keras.backend.clear_session()
            gc.collect()


if __name__ == "__main__":
    main()