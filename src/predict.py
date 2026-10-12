"""AgriAI: Inference and Evaluation CLI for Tomato Leaf Disease Classification.

This module provides two core capabilities:
1. Single-image prediction: Classifies an uploaded leaf image, displaying predicted
   disease class, confidence, and full probability distribution.
2. Dataset evaluation: Evaluates a labeled test directory, computing cross-entropy
   loss, overall accuracy, per-class precision/recall/F1, and confusion matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "model_config.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_crop_disease_cnn.keras"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "crop_disease_cnn.keras"

DEFAULT_IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
DEFAULT_MIN_IMAGE_DIMENSION = 64  # Minimum width and height in pixels

# Provisional uncertainty thresholds for flagging borderline or ambiguous predictions.
# Note: These are provisional heuristic guardrails and do not constitute
# a validated out-of-distribution detection mechanism for non-tomato images.
DEFAULT_MIN_SCORE_THRESHOLD = 0.60
DEFAULT_MIN_MARGIN_THRESHOLD = 0.15

DEFAULT_CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]
DEFAULT_READABLE_LABELS = {
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___healthy": "Healthy",
}


class PredictionResult(NamedTuple):
    """Container for single-image prediction output and uncertainty assessment."""

    predicted_class: str
    top_score: float
    class_probabilities: dict[str, float]
    uncertainty: dict[str, Any]


def load_config() -> dict[str, Any]:
    """Load settings from model_config.json or return defaults."""
    if CONFIG_PATH.is_file():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_model_path(custom_path: Path | None = None) -> Path:
    """Resolve model checkpoint path with fallback handling."""
    if custom_path is not None and custom_path.is_file():
        return custom_path
    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH
    if FALLBACK_MODEL_PATH.is_file():
        return FALLBACK_MODEL_PATH
    raise FileNotFoundError(
        f"No model checkpoint found at '{DEFAULT_MODEL_PATH}' or '{FALLBACK_MODEL_PATH}'."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prediction or dataset evaluation."""
    parser = argparse.ArgumentParser(
        description="AgriAI: Predict tomato leaf diseases from single images or evaluate a labeled test dataset."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=Path,
        help="Path to a single tomato-leaf image file to classify.",
    )
    group.add_argument(
        "--eval-dir",
        type=Path,
        help="Path to a labeled evaluation dataset folder (e.g., data/test) to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained Keras model checkpoint (defaults to final/baseline checkpoint).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for dataset evaluation (default: 32).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE_THRESHOLD,
        help=f"Provisional minimum top score threshold (default: {DEFAULT_MIN_SCORE_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN_THRESHOLD,
        help=f"Provisional minimum margin threshold between top two classes (default: {DEFAULT_MIN_MARGIN_THRESHOLD}).",
    )
    return parser.parse_args()


def validate_probabilities(probabilities: np.ndarray, expected_num_classes: int) -> None:
    """Validate that model outputs are finite, non-negative, and represent a valid probability distribution."""
    if probabilities.shape != (expected_num_classes,):
        raise ValueError(
            f"Unexpected prediction shape {probabilities.shape}; expected ({expected_num_classes},)."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model output contains non-finite values (NaN or Inf).")
    if np.any(probabilities < -1e-5):
        raise ValueError("Model output contains negative score values.")
    prob_sum = float(np.sum(probabilities))
    if not (0.95 <= prob_sum <= 1.05):
        raise ValueError(
            f"Model output scores do not sum to approximately 1.0 (got sum {prob_sum:.4f})."
        )


def evaluate_uncertainty(
    probabilities: np.ndarray,
    class_names: list[str],
    min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
    min_margin_threshold: float = DEFAULT_MIN_MARGIN_THRESHOLD,
) -> dict[str, Any]:
    """Calculate highest score, second-highest score, margin, and conclusiveness status.

    Note on limitations:
    - Score and margin thresholds (provisional defaults: 0.60 score, 0.15 margin)
      filter out borderline or ambiguous predictions among the 3 supported classes.
    - Softmax scores are relative exponential outputs, NOT calibrated Bayesian probabilities.
    - These thresholds alone do NOT reliably detect out-of-domain or non-tomato images,
      as this model lacks a validated out-of-distribution rejection mechanism.
    """
    validate_probabilities(probabilities, len(class_names))

    sorted_indices = np.argsort(probabilities)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx

    top1_class = class_names[top1_idx]
    top2_class = class_names[top2_idx]

    top1_score = float(probabilities[top1_idx])
    top2_score = float(probabilities[top2_idx])
    margin = float(top1_score - top2_score)

    reasons: list[str] = []
    if top1_score < min_score_threshold:
        reasons.append(
            f"Highest model output score ({top1_score:.2%}) is below provisional threshold ({min_score_threshold:.2%})."
        )
    if margin < min_margin_threshold:
        reasons.append(
            f"Score margin ({margin:.2%}) between '{top1_class}' and '{top2_class}' is below provisional threshold ({min_margin_threshold:.2%})."
        )

    is_conclusive = len(reasons) == 0

    return {
        "is_conclusive": is_conclusive,
        "predicted_class": top1_class,
        "top_score": top1_score,
        "second_class": top2_class,
        "second_score": top2_score,
        "margin": margin,
        "status_label": top1_class if is_conclusive else "Inconclusive — please try another image",
        "reasons": reasons,
        "min_score_threshold": float(min_score_threshold),
        "min_margin_threshold": float(min_margin_threshold),
    }


def prepare_image(
    image_path: Path,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    min_dimension: int = DEFAULT_MIN_IMAGE_DIMENSION,
) -> np.ndarray:
    """Load an image as RGB, resize it, and return a raw-pixel batch tensor (1, H, W, 3).

    Validates that the file exists, has a non-zero size, does not exceed maximum size,
    meets minimum resolution requirements, and uses a supported image format.
    Handles corrupted, truncated, or unreadable images gracefully with user-friendly errors.

    The model includes its internal Rescaling(1/255) layer, so raw pixel values in [0, 255]
    are preserved without external normalization to avoid double-scaling.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: '{image_path.name}'")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported image format '{image_path.suffix}'. Supported formats are: {supported_str}."
        )

    try:
        file_size = image_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Could not inspect file attributes for '{image_path.name}': {exc}") from exc

    if file_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): '{image_path.name}'.")

    if file_size > max_file_size_bytes:
        max_mb = max_file_size_bytes / (1024 * 1024)
        raise ValueError(
            f"Image file exceeds maximum allowed size of {max_mb:.1f} MB: '{image_path.name}'."
        )

    try:
        with Image.open(image_path) as img:
            img.load()  # Force decode all pixel data to catch truncated or corrupted files
            if img.width < min_dimension or img.height < min_dimension:
                raise ValueError(
                    f"Image resolution is too small ({img.width}×{img.height} px). "
                    f"Minimum required dimensions are {min_dimension}×{min_dimension} pixels: '{image_path.name}'."
                )
            if img.mode != "RGB":
                img = img.convert("RGB")
            resized = img.resize(target_size, resample=Image.Resampling.BILINEAR)
            image_array = np.array(resized, dtype=np.float32)
    except UnidentifiedImageError as exc:
        raise ValueError(f"File is not a valid or readable image: '{image_path.name}'.") from exc
    except Image.DecompressionBombError as exc:
        raise ValueError(f"Image exceeds safe size limits (decompression bomb risk): '{image_path.name}'.") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, ValueError) and ("too small" in str(exc) or "maximum allowed" in str(exc)):
            raise
        raise ValueError(f"Corrupted, truncated, or unreadable image file '{image_path.name}'.") from exc

    if image_array.shape != (*target_size, 3):
        raise ValueError(
            f"Unexpected image tensor shape {image_array.shape}; expected {(*target_size, 3)}."
        )
    return np.expand_dims(image_array, axis=0)


def predict_image(
    image_path: Path,
    model_path: Path | None = None,
    min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
    min_margin_threshold: float = DEFAULT_MIN_MARGIN_THRESHOLD,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    min_dimension: int = DEFAULT_MIN_IMAGE_DIMENSION,
) -> PredictionResult:
    """Classify a single leaf image and return predicted class, top score, probabilities, and uncertainty analysis."""
    resolved_model_path = get_model_path(model_path)
    model = tf.keras.models.load_model(resolved_model_path)

    config = load_config()
    classes_section = config.get("classes", {})
    class_names = classes_section.get("class_names", DEFAULT_CLASS_NAMES)
    input_spec = config.get("input_specification", {})
    img_h = input_spec.get("image_height", DEFAULT_IMAGE_SIZE[0])
    img_w = input_spec.get("image_width", DEFAULT_IMAGE_SIZE[1])
    target_size = (img_h, img_w)

    image_batch = prepare_image(
        image_path,
        target_size=target_size,
        max_file_size_bytes=max_file_size_bytes,
        min_dimension=min_dimension,
    )
    probabilities = np.asarray(model.predict(image_batch, verbose=0)[0], dtype=np.float64)

    uncertainty = evaluate_uncertainty(
        probabilities,
        class_names,
        min_score_threshold=min_score_threshold,
        min_margin_threshold=min_margin_threshold,
    )

    class_probabilities = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(class_names)
    }
    predicted_class = uncertainty["predicted_class"]
    top_score = uncertainty["top_score"]

    return PredictionResult(
        predicted_class=predicted_class,
        top_score=top_score,
        class_probabilities=class_probabilities,
        uncertainty=uncertainty,
    )


def evaluate_dataset(
    data_dir: Path,
    model_path: Path | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate a labeled dataset directory and calculate loss, accuracy, and per-class metrics."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory does not exist: {data_dir}")

    resolved_model_path = get_model_path(model_path)
    model = tf.keras.models.load_model(resolved_model_path)

    config = load_config()
    class_names = config.get("classes", {}).get("class_names", DEFAULT_CLASS_NAMES)
    input_spec = config.get("input_specification", {})
    target_size = (
        input_spec.get("image_height", DEFAULT_IMAGE_SIZE[0]),
        input_spec.get("image_width", DEFAULT_IMAGE_SIZE[1]),
    )

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False,
    )

    actual_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    for images, labels in dataset:
        batch_preds = model(images, training=False)
        batch_labels = labels.numpy()
        batch_probs = batch_preds.numpy()

        actual_list.append(batch_labels)
        pred_list.append(np.argmax(batch_probs, axis=1))

        per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, batch_preds)
        total_loss += float(tf.reduce_sum(per_sample_loss).numpy())
        total_samples += len(batch_labels)

    actual = np.concatenate(actual_list)
    predicted = np.concatenate(pred_list)

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(confusion, (actual, predicted), 1)

    support = confusion.sum(axis=1)
    true_positives = np.diag(confusion)
    predicted_totals = confusion.sum(axis=0)

    precision = np.divide(
        true_positives,
        predicted_totals,
        out=np.zeros(len(class_names), dtype=float),
        where=predicted_totals != 0,
    )
    recall = np.divide(
        true_positives,
        support,
        out=np.zeros(len(class_names), dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(class_names), dtype=float),
        where=(precision + recall) != 0,
    )

    overall_accuracy = float(true_positives.sum() / total_samples) if total_samples else 0.0
    mean_loss = total_loss / total_samples if total_samples else 0.0

    per_class_metrics = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    macro_avg = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1_score": float(np.mean(f1)),
        "support": total_samples,
    }

    weighted_avg = {
        "precision": float(np.average(precision, weights=support)) if total_samples else 0.0,
        "recall": float(np.average(recall, weights=support)) if total_samples else 0.0,
        "f1_score": float(np.average(f1, weights=support)) if total_samples else 0.0,
        "support": total_samples,
    }

    resolved_data_dir = data_dir.resolve()
    rel_data_dir = str(resolved_data_dir.relative_to(PROJECT_ROOT)).replace("\\", "/") if resolved_data_dir.is_relative_to(PROJECT_ROOT) else str(resolved_data_dir)
    rel_model_path = str(resolved_model_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if resolved_model_path.is_relative_to(PROJECT_ROOT) else str(resolved_model_path)

    return {
        "model_evaluated": rel_model_path,
        "dataset_directory": rel_data_dir,
        "total_images": total_samples,
        "loss": mean_loss,
        "accuracy": overall_accuracy,
        "per_class": per_class_metrics,
        "macro_average": macro_avg,
        "weighted_average": weighted_avg,
        "confusion_matrix": confusion.tolist(),
        "class_names": class_names,
    }


def print_evaluation_report(results: dict[str, Any]) -> None:
    """Format and print a complete evaluation report to terminal."""
    class_names = results["class_names"]
    print("\n" + "=" * 78)
    print(" AgriAI Model Evaluation Report")
    print("=" * 78)
    print(f"Model Checkpoint:    {results['model_evaluated']}")
    print(f"Evaluation Dataset:  {results['dataset_directory']}")
    print(f"Total Test Images:   {results['total_images']}")
    print(f"Sparse Cross-Entropy Loss: {results['loss']:.6f}")
    print(f"Overall Accuracy:          {results['accuracy']:.4%} ({int(round(results['accuracy'] * results['total_images']))} / {results['total_images']})")
    print("-" * 78)
    print(f"{'Class Name':32s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>9s}")
    print("-" * 78)
    for name in class_names:
        m = results["per_class"][name]
        print(f"{name:32s} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1_score']:10.4f} {m['support']:9d}")
    print("-" * 78)
    print(f"{'Accuracy':32s} {'':>10s} {'':>10s} {results['accuracy']:10.4f} {results['total_images']:9d}")
    macro = results["macro_average"]
    print(f"{'Macro Average':32s} {macro['precision']:10.4f} {macro['recall']:10.4f} {macro['f1_score']:10.4f} {macro['support']:9d}")
    weighted = results["weighted_average"]
    print(f"{'Weighted Average':32s} {weighted['precision']:10.4f} {weighted['recall']:10.4f} {weighted['f1_score']:10.4f} {weighted['support']:9d}")
    print("-" * 78)
    print("\nConfusion Matrix (Rows = Actual, Columns = Predicted):")
    conf = np.array(results["confusion_matrix"])
    short_names = [n.replace("Tomato___", "") for n in class_names]
    header = f"{'':18s}" + "".join(f"{name:>16s}" for name in short_names)
    print(header)
    for i, name in enumerate(short_names):
        row_str = "".join(f"{conf[i, j]:16d}" for j in range(len(short_names)))
        print(f"{name:18s}{row_str}")
    print("=" * 78 + "\n")


def main() -> int:
    """Execute prediction or evaluation CLI mode."""
    args = parse_args()
    
    if args.image is not None:
        try:
            result = predict_image(
                args.image,
                model_path=args.model,
                min_score_threshold=args.min_score,
                min_margin_threshold=args.min_margin,
            )
        except Exception as exc:
            print(f"Error during prediction: {exc}", file=sys.stderr)
            return 1

        config = load_config()
        display_names = config.get("classes", {}).get("display_names", DEFAULT_READABLE_LABELS)
        predicted_class = result.predicted_class
        readable_label = display_names.get(predicted_class, predicted_class)
        uncertainty = result.uncertainty
        is_conclusive = uncertainty["is_conclusive"]

        print("\n" + "=" * 54)
        print(" AgriAI Single-Image Diagnosis")
        print("=" * 54)
        print(f"Image File:          {args.image}")
        if is_conclusive:
            print("Diagnostic Status:   CONCLUSIVE")
            print(f"Predicted Class:     {predicted_class} ({readable_label})")
        else:
            print("Diagnostic Status:   INCONCLUSIVE -- Please try another image")
            print(f"Tentative Class:     {predicted_class} ({readable_label}) [Uncertain]")

        print(f"Model Output Score:  {result.top_score:.2%}")
        second_name = display_names.get(uncertainty['second_class'], uncertainty['second_class'])
        print(f"Top Class Margin:    {uncertainty['margin']:.2%} (vs {second_name}: {uncertainty['second_score']:.2%})")

        if not is_conclusive:
            print("-" * 54)
            print("Flagged Uncertainty Reasons:")
            for reason in uncertainty["reasons"]:
                print(f"  * {reason}")
            print("\nAdvisory Notice:")
            print("  Definitive disease management or chemical treatment should NOT")
            print("  be applied based on an inconclusive classification.")

        print("-" * 54)
        print("Relative Model Output Score Breakdown:")
        for cls_name, prob in result.class_probabilities.items():
            label = display_names.get(cls_name, cls_name)
            bar = "#" * int(prob * 30)
            print(f"  {label:15s} [{prob:6.2%}] {bar}")
        print("-" * 54)
        print("Reliability & Domain Limitations:")
        print("* Model output scores indicate uncalibrated relative outputs among")
        print("  the 3 supported classes and are not guaranteed real-world probabilities.")
        print("* This model only supports 3 tomato leaf categories and does NOT reliably")
        print("  detect non-tomato leaves, other plant diseases, or arbitrary images.")
        print("* Educational prototype; not a substitute for certified agronomist diagnosis.")
        print("=" * 54 + "\n")
        return 0

    elif args.eval_dir is not None:
        try:
            results = evaluate_dataset(
                args.eval_dir,
                model_path=args.model,
                batch_size=args.batch_size,
            )
            print_evaluation_report(results)
            return 0
        except Exception as exc:
            print(f"Error during evaluation: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
