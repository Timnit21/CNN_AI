"""Evaluate the saved Phase 5 checkpoint on the held-out test split only."""

from __future__ import annotations

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
MODEL_PATH = PROJECT_ROOT / "models" / "crop_disease_cnn.keras"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]
VALIDATION_ACCURACY = 0.9658246636390686


def metric_values(confusion: np.ndarray) -> dict[str, Any]:
    """Calculate class, macro, weighted, and overall metrics from a confusion matrix."""
    support = confusion.sum(axis=1)
    true_positive = np.diag(confusion)
    predicted = confusion.sum(axis=0)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive, dtype=float), where=predicted != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    total = int(support.sum())

    def average(values: np.ndarray, weighted: bool) -> float:
        if total == 0:
            return 0.0
        return float(np.average(values, weights=support if weighted else None))

    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
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
    }


def make_classification_report(metrics: dict[str, Any]) -> str:
    """Format a readable report containing all requested classes and averages."""
    lines = [
        "Classification report (test set)",
        "",
        f"{'Class':32s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s} {'Support':>9s}",
        "-" * 76,
    ]
    for class_name in CLASS_NAMES:
        result = metrics["per_class"][class_name]
        lines.append(
            f"{class_name:32s} {result['precision']:10.4f} {result['recall']:10.4f} "
            f"{result['f1_score']:10.4f} {result['support']:9d}"
        )
    lines.append("-" * 76)
    lines.append(f"{'Accuracy':32s} {'':>10s} {'':>10s} {metrics['accuracy']:10.4f} {metrics['macro_average']['support']:9d}")
    for title, key in (("Macro average", "macro_average"), ("Weighted average", "weighted_average")):
        result = metrics[key]
        lines.append(
            f"{title:32s} {result['precision']:10.4f} {result['recall']:10.4f} "
            f"{result['f1_score']:10.4f} {result['support']:9d}"
        )
    return "\n".join(lines) + "\n"


def save_confusion_matrix(confusion: np.ndarray) -> None:
    """Save a labeled 3x3 confusion-matrix visualization."""
    short_names = [name.replace("Tomato___", "") for name in CLASS_NAMES]
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Number of images")
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=short_names,
        yticklabels=short_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Held-out Test Set Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    threshold = confusion.max() / 2 if confusion.size else 0
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(
                column,
                row,
                f"{confusion[row, column]:,}",
                ha="center",
                va="center",
                color="white" if confusion[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Evaluate the checkpoint and persist only measured test-set results."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved checkpoint not found: {MODEL_PATH}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

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
        raise ValueError(f"Test class names/order mismatch: {test_dataset.class_names}")

    model = tf.keras.models.load_model(MODEL_PATH)
    if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(f"Model output has {model.output_shape[-1]} classes, expected 3")

    evaluation = model.evaluate(test_dataset, verbose=0, return_dict=True)
    predictions = model.predict(test_dataset, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)
    if len(actual_labels) != len(predicted_labels):
        raise ValueError("Prediction count does not match test image count")

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (actual_labels, predicted_labels), 1)
    metrics = metric_values(confusion)
    metrics["test_loss"] = float(evaluation["loss"])
    metrics["test_accuracy"] = float(evaluation["compile_metrics"] if "compile_metrics" in evaluation else evaluation["accuracy"])
    # Report accuracy from both Keras evaluation and the confusion matrix to catch inconsistencies.
    if not np.isclose(metrics["test_accuracy"], metrics["accuracy"], atol=1e-7):
        raise ValueError("Keras test accuracy disagrees with confusion-matrix accuracy")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(confusion)
    report = make_classification_report(metrics)
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    class_metrics = metrics["per_class"]
    best_class = max(CLASS_NAMES, key=lambda name: class_metrics[name]["f1_score"])
    hardest_class = min(CLASS_NAMES, key=lambda name: class_metrics[name]["f1_score"])
    off_diagonal = [
        (int(confusion[row, column]), CLASS_NAMES[row], CLASS_NAMES[column])
        for row in range(len(CLASS_NAMES))
        for column in range(len(CLASS_NAMES))
        if row != column and confusion[row, column] > 0
    ]
    strongest_confusions = sorted(off_diagonal, reverse=True)
    comparison_gap = metrics["test_accuracy"] - VALIDATION_ACCURACY
    results = {
        "checkpoint": "models/crop_disease_cnn.keras",
        "test_data_directory": "data/test",
        "test_data_used_for_training": False,
        "test_data_used_for_validation": False,
        "test_images": int(len(actual_labels)),
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "class_names": CLASS_NAMES,
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape),
        "test_loss": metrics["test_loss"],
        "test_accuracy": metrics["test_accuracy"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["weighted_average"]["precision"],
        "recall": metrics["weighted_average"]["recall"],
        "f1_score": metrics["weighted_average"]["f1_score"],
        "macro_average": metrics["macro_average"],
        "weighted_average": metrics["weighted_average"],
        "per_class": metrics["per_class"],
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": {"rows": "actual classes", "columns": "predicted classes"},
        "validation_comparison": {
            "best_validation_accuracy": VALIDATION_ACCURACY,
            "test_minus_validation_accuracy": float(comparison_gap),
            "test_minus_validation_percentage_points": float(comparison_gap * 100),
        },
        "analysis": {
            "best_class_by_f1": best_class,
            "most_difficult_class_by_f1": hardest_class,
            "nonzero_misclassifications": [
                {"count": count, "actual": actual, "predicted": predicted}
                for count, actual, predicted in strongest_confusions
            ],
            "generalization_note": (
                "Test accuracy is compared with the best validation accuracy. This is a single "
                "held-out split; conclusions are limited to this dataset."
            ),
        },
    }
    (RESULTS_DIR / "test_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("TEST RESULTS")
    print("------------")
    print(f"Test images: {len(actual_labels)}")
    print(f"Test loss: {metrics['test_loss']:.6f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4%}")
    print(f"Precision (weighted): {metrics['weighted_average']['precision']:.4%}")
    print(f"Recall (weighted): {metrics['weighted_average']['recall']:.4%}")
    print(f"F1-score (weighted): {metrics['weighted_average']['f1_score']:.4%}")
    print(f"Macro precision: {metrics['macro_average']['precision']:.4%}")
    print(f"Macro recall: {metrics['macro_average']['recall']:.4%}")
    print(f"Macro F1-score: {metrics['macro_average']['f1_score']:.4%}")
    print(f"Best validation accuracy: {VALIDATION_ACCURACY:.4%}")
    print(f"Test minus best validation: {comparison_gap * 100:+.2f} percentage points")
    print("\nCLASSIFICATION REPORT")
    print(report, end="")
    print("\nCONFUSION MATRIX (rows=actual, columns=predicted)")
    print(confusion)
    print(f"\nBest class by F1: {best_class}")
    print(f"Most difficult class by F1: {hardest_class}")
    if strongest_confusions:
        print("Nonzero confusions (actual -> predicted):")
        for count, actual, predicted in strongest_confusions:
            print(f"  {actual} -> {predicted}: {count}")
    else:
        print("No misclassifications in the test set.")
    print("\nSaved: results/confusion_matrix.png")
    print("Saved: results/classification_report.txt")
    print("Saved: results/test_results.json")


if __name__ == "__main__":
    main()
