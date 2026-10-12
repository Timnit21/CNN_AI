"""Evaluate the selected final checkpoint once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "test"
FINAL_MODEL = PROJECT_ROOT / "models" / "final_crop_disease_cnn.keras"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


def parse_args() -> argparse.Namespace:
    """Parse selected model path; default to the Phase 8 final checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=FINAL_MODEL,
        help="Selected model checkpoint; defaults to models/final_crop_disease_cnn.keras.",
    )
    return parser.parse_args()


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    """Calculate per-class, macro, and support-weighted metrics."""
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(len(CLASS_NAMES), dtype=float),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(CLASS_NAMES), dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(CLASS_NAMES), dtype=float),
        where=(precision + recall) != 0,
    )
    total = int(support.sum())

    def average(values: np.ndarray, weighted: bool) -> float:
        if total == 0:
            return 0.0
        return float(np.average(values, weights=support if weighted else None))

    accuracy = float(true_positive.sum() / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": average(precision, weighted=True),
        "recall": average(recall, weighted=True),
        "f1_score": average(f1, weighted=True),
        "macro_average": {
            "precision": average(precision, weighted=False),
            "recall": average(recall, weighted=False),
            "f1_score": average(f1, weighted=False),
            "support": total,
        },
        "weighted_average": {
            "precision": average(precision, weighted=True),
            "recall": average(recall, weighted=True),
            "f1_score": average(f1, weighted=True),
            "support": total,
        },
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def make_report(metrics: dict[str, Any]) -> str:
    """Format the actual predictions into a readable classification report."""
    lines = [
        "Final model classification report (held-out test set)",
        "",
        f"{'Class':32s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s} {'Support':>9s}",
        "-" * 76,
    ]
    for class_name in CLASS_NAMES:
        item = metrics["per_class"][class_name]
        lines.append(
            f"{class_name:32s} {item['precision']:10.4f} {item['recall']:10.4f} "
            f"{item['f1_score']:10.4f} {item['support']:9d}"
        )
    lines.extend(
        [
            "-" * 76,
            f"{'Accuracy':32s} {'':>10s} {'':>10s} {metrics['accuracy']:10.4f} "
            f"{metrics['weighted_average']['support']:9d}",
        ]
    )
    for label, key in (("Macro average", "macro_average"), ("Weighted average", "weighted_average")):
        item = metrics[key]
        lines.append(
            f"{label:32s} {item['precision']:10.4f} {item['recall']:10.4f} "
            f"{item['f1_score']:10.4f} {item['support']:9d}"
        )
    return "\n".join(lines) + "\n"


def save_confusion_matrix(confusion: np.ndarray, output_path: Path) -> None:
    """Save the 3-class confusion matrix with readable names and counts."""
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Number of test images")
    short_names = [name.replace("Tomato___", "") for name in CLASS_NAMES]
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=short_names,
        yticklabels=short_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Final Model — Held-out Test Set Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    threshold = confusion.max() / 2
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(
                column,
                row,
                f"{confusion[row, column]:,}",
                ha="center",
                va="center",
                color="white" if confusion[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load the selected model, evaluate once, and persist Phase 8 outputs."""
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected final model does not exist: {model_path}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Held-out test directory does not exist: {TEST_DIR}")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    if list(test_dataset.class_names) != CLASS_NAMES:
        raise ValueError(f"Unexpected test class order: {test_dataset.class_names}")

    model = tf.keras.models.load_model(model_path)
    if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected input shape: {model.input_shape}")
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(f"Expected 3 model outputs; got {model.output_shape[-1]}")

    actual_batches: list[np.ndarray] = []
    predicted_batches: list[np.ndarray] = []
    loss_sum = 0.0
    sample_count = 0
    for images, labels in test_dataset:
        probabilities = model(images, training=False)
        batch_labels = labels.numpy()
        batch_probabilities = probabilities.numpy()
        actual_batches.append(batch_labels)
        predicted_batches.append(np.argmax(batch_probabilities, axis=1))
        per_image_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, probabilities
        )
        loss_sum += float(tf.reduce_sum(per_image_loss).numpy())
        sample_count += len(batch_labels)
    actual_labels = np.concatenate(actual_batches)
    predicted_labels = np.concatenate(predicted_batches)
    if sample_count != len(actual_labels) or len(actual_labels) != len(predicted_labels):
        raise ValueError("Prediction count does not match test image count.")

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (actual_labels, predicted_labels), 1)
    if int(confusion.sum()) != len(actual_labels) or len(actual_labels) != 673:
        raise ValueError(
            f"Expected exactly 673 test predictions; got {len(actual_labels)} "
            f"(confusion matrix total {int(confusion.sum())})."
        )
    metrics = metrics_from_confusion(confusion)
    metrics["test_loss"] = loss_sum / sample_count
    metrics["test_accuracy"] = metrics["accuracy"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "final_classification_report.txt"
    confusion_path = RESULTS_DIR / "final_confusion_matrix.png"
    results_path = RESULTS_DIR / "final_test_results.json"
    report_path.write_text(make_report(metrics), encoding="utf-8")
    save_confusion_matrix(confusion, confusion_path)
    results = {
        "checkpoint": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "test_data_directory": "data/test",
        "test_data_images": int(len(actual_labels)),
        "test_data_used_for_training": False,
        "test_data_used_for_validation": False,
        "class_names": CLASS_NAMES,
        "image_size": [*IMAGE_SIZE, 3],
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape),
        "test_loss": metrics["test_loss"],
        "test_accuracy": metrics["test_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "macro_average": metrics["macro_average"],
        "weighted_average": metrics["weighted_average"],
        "per_class": metrics["per_class"],
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": {"rows": "actual classes", "columns": "predicted classes"},
        "phase8_test_passes": 1,
        "historical_baseline_test_evaluation_exists": True,
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key != "per_class"}, indent=2))
    print(f"Saved {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {confusion_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {results_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
