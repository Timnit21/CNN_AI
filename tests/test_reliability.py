"""Automated reliability tests for AgriAI Tomato Leaf Disease Classification.

Uses Python's built-in unittest framework without third-party test runners.
Lightweight tests use small synthetic images and synthetic probability vectors
to execute rapidly without repeatedly reloading the full CNN checkpoint.
A dedicated integration test verifies the primary model checkpoint architecture and forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.predict import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DEFAULT_MIN_IMAGE_DIMENSION,
    DEFAULT_MIN_MARGIN_THRESHOLD,
    DEFAULT_MIN_SCORE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    SUPPORTED_EXTENSIONS,
    evaluate_uncertainty,
    prepare_image,
    validate_probabilities,
)
from app import CLASS_NAMES as APP_CLASS_NAMES, CONFIG_PATH as APP_CONFIG_PATH


class TestLightweightPreprocessingAndValidation(unittest.TestCase):
    """Fast unit tests for image decoding, format checks, and error handling."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_rgb_image_dimensions_and_dtype(self) -> None:
        """Valid JPEG image must resize to (1, 224, 224, 3) float32."""
        img_path = self.temp_path / "valid_leaf.jpg"
        img = Image.new("RGB", (300, 200), color=(80, 160, 40))
        img.save(img_path, format="JPEG")

        tensor = prepare_image(img_path, target_size=(224, 224))
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_raw_pixel_scaling_preserved_no_double_scaling(self) -> None:
        """Raw pixel values in [0, 255] must be preserved for internal Rescaling layer."""
        img_path = self.temp_path / "test_pixel_scale.png"
        img = Image.new("RGB", (100, 100), color=(200, 150, 100))
        img.save(img_path, format="PNG")

        tensor = prepare_image(img_path, target_size=(224, 224))
        # Tensor values should remain near 200, 150, 100 — not pre-normalized to [0, 1]
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 200.0, delta=1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 1]), 150.0, delta=1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 2]), 100.0, delta=1.0)
        self.assertGreater(float(np.max(tensor)), 1.0)

    def test_rgba_converted_to_three_channel_rgb(self) -> None:
        """RGBA image must have alpha channel stripped and produce 3 channels."""
        img_path = self.temp_path / "transparent_leaf.png"
        img = Image.new("RGBA", (100, 100), color=(50, 120, 30, 128))
        img.save(img_path, format="PNG")

        tensor = prepare_image(img_path, target_size=(224, 224))
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_grayscale_converted_to_three_channel_rgb(self) -> None:
        """Single-channel grayscale image must convert to 3 identical RGB channels."""
        img_path = self.temp_path / "gray_leaf.png"
        img = Image.new("L", (80, 80), color=128)
        img.save(img_path, format="PNG")

        tensor = prepare_image(img_path, target_size=(224, 224))
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        # Verify all 3 channels have equal pixel intensity
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), float(tensor[0, 0, 0, 1]), delta=1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 1]), float(tensor[0, 0, 0, 2]), delta=1.0)

    def test_missing_file_raises_filenotfound(self) -> None:
        """Attempting to prepare a non-existent file path must raise FileNotFoundError."""
        missing_path = self.temp_path / "non_existent_file.jpg"
        with self.assertRaises(FileNotFoundError) as ctx:
            prepare_image(missing_path)
        self.assertIn("does not exist", str(ctx.exception))

    def test_empty_zero_byte_file_raises_valueerror(self) -> None:
        """Empty 0-byte image file must raise a clear ValueError."""
        empty_path = self.temp_path / "empty_image.jpg"
        empty_path.write_bytes(b"")

        with self.assertRaises(ValueError) as ctx:
            prepare_image(empty_path)
        self.assertIn("empty (0 bytes)", str(ctx.exception).lower())

    def test_unsupported_file_extension_raises_valueerror(self) -> None:
        """Non-supported file format (.txt, .bmp, .pdf) must raise ValueError."""
        unsupported_path = self.temp_path / "leaf_data.txt"
        unsupported_path.write_text("not an image")

        with self.assertRaises(ValueError) as ctx:
            prepare_image(unsupported_path)
        self.assertIn("unsupported image format", str(ctx.exception).lower())

    def test_corrupted_truncated_file_raises_valueerror(self) -> None:
        """Corrupted or truncated file bytes must raise a descriptive ValueError."""
        corrupted_path = self.temp_path / "corrupted_leaf.jpg"
        # Write valid JPEG header followed by abrupt EOF
        corrupted_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")

        with self.assertRaises(ValueError) as ctx:
            prepare_image(corrupted_path)
        err_msg = str(ctx.exception).lower()
        self.assertTrue(
            "corrupted" in err_msg or "unreadable" in err_msg or "cannot identify" in err_msg
        )

    def test_image_smaller_than_minimum_dimensions_raises_valueerror(self) -> None:
        """Images below the minimum dimensions (64x64) must raise ValueError."""
        tiny_path = self.temp_path / "tiny_icon.png"
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        img.save(tiny_path, format="PNG")

        with self.assertRaises(ValueError) as ctx:
            prepare_image(tiny_path)
        self.assertIn("too small", str(ctx.exception).lower())

    def test_image_exceeding_max_file_size_raises_valueerror(self) -> None:
        """Files exceeding the maximum file size limit must raise ValueError."""
        img_path = self.temp_path / "oversized_leaf.jpg"
        img = Image.new("RGB", (100, 100), color=(100, 150, 50))
        img.save(img_path, format="JPEG")

        # Set max_file_size_bytes to 50 bytes (smaller than the file)
        with self.assertRaises(ValueError) as ctx:
            prepare_image(img_path, max_file_size_bytes=50)
        self.assertIn("maximum allowed size", str(ctx.exception).lower())

    def test_safe_error_messages_do_not_expose_absolute_paths(self) -> None:
        """Error messages must only reference the filename, not full filesystem paths."""
        missing_path = self.temp_path / "private_folder" / "nonexistent.jpg"
        with self.assertRaises(FileNotFoundError) as ctx:
            prepare_image(missing_path)
        self.assertIn("nonexistent.jpg", str(ctx.exception))
        self.assertNotIn(str(self.temp_path), str(ctx.exception))

    def test_decompression_bomb_raises_valueerror(self) -> None:
        """Images exceeding PIL's MAX_IMAGE_PIXELS must trigger decompression bomb handling."""
        img_path = self.temp_path / "bomb.png"
        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        img.save(img_path, format="PNG")

        old_max = Image.MAX_IMAGE_PIXELS
        try:
            # PIL raises DecompressionBombError when pixels > 2 * MAX_IMAGE_PIXELS (100x100 = 10,000 > 2 * 3,000)
            Image.MAX_IMAGE_PIXELS = 3000
            with self.assertRaises(ValueError) as ctx:
                prepare_image(img_path)
            self.assertIn("decompression bomb", str(ctx.exception).lower())
        finally:
            Image.MAX_IMAGE_PIXELS = old_max


class TestConfigAndClassMappingConsistency(unittest.TestCase):
    """Verify consistency between model_config.json, predict.py, and app.py."""

    def test_class_names_and_indices_match_across_system(self) -> None:
        """Class names and alphabetical ordering must match identically across all files."""
        self.assertTrue(APP_CONFIG_PATH.is_file(), "model_config.json must exist")
        with APP_CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)

        config_classes = config.get("classes", {}).get("class_names", [])
        expected_classes = [
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___healthy",
        ]

        self.assertEqual(config_classes, expected_classes)
        self.assertEqual(DEFAULT_CLASS_NAMES, expected_classes)
        self.assertEqual(APP_CLASS_NAMES, expected_classes)

    def test_supported_extensions(self) -> None:
        """Supported extensions must include standard web image formats."""
        self.assertIn(".jpg", SUPPORTED_EXTENSIONS)
        self.assertIn(".jpeg", SUPPORTED_EXTENSIONS)
        self.assertIn(".png", SUPPORTED_EXTENSIONS)


class TestProbabilityValidation(unittest.TestCase):
    """Unit tests for probability distribution verification."""

    def test_valid_probabilities(self) -> None:
        """Finite, non-negative, normalized array must pass validation."""
        valid_probs = np.array([0.70, 0.20, 0.10], dtype=np.float64)
        validate_probabilities(valid_probs, 3)

    def test_non_finite_nan_raises_valueerror(self) -> None:
        """NaN output must raise ValueError."""
        nan_probs = np.array([np.nan, 0.50, 0.50], dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            validate_probabilities(nan_probs, 3)
        self.assertIn("non-finite", str(ctx.exception).lower())

    def test_non_finite_inf_raises_valueerror(self) -> None:
        """Inf output must raise ValueError."""
        inf_probs = np.array([np.inf, 0.0, 0.0], dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            validate_probabilities(inf_probs, 3)
        self.assertIn("non-finite", str(ctx.exception).lower())

    def test_negative_values_raise_valueerror(self) -> None:
        """Negative scores must raise ValueError."""
        neg_probs = np.array([-0.05, 0.55, 0.50], dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            validate_probabilities(neg_probs, 3)
        self.assertIn("negative", str(ctx.exception).lower())

    def test_sum_mismatch_raises_valueerror(self) -> None:
        """Scores failing to sum near 1.0 must raise ValueError."""
        bad_sum = np.array([0.20, 0.20, 0.20], dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            validate_probabilities(bad_sum, 3)
        self.assertIn("approximately 1.0", str(ctx.exception).lower())

    def test_wrong_shape_raises_valueerror(self) -> None:
        """Unexpected array length must raise ValueError."""
        wrong_len = np.array([0.50, 0.50], dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            validate_probabilities(wrong_len, 3)
        self.assertIn("unexpected prediction shape", str(ctx.exception).lower())


class TestUncertaintyEvaluation(unittest.TestCase):
    """Unit tests for top score, margin calculation, and provisional thresholds."""

    def setUp(self) -> None:
        self.classes = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]

    def test_provisional_threshold_constants_documented(self) -> None:
        """Verify baseline provisional threshold values."""
        self.assertEqual(DEFAULT_MIN_SCORE_THRESHOLD, 0.60)
        self.assertEqual(DEFAULT_MIN_MARGIN_THRESHOLD, 0.15)

    def test_conclusive_prediction(self) -> None:
        """High score and wide margin result in conclusive diagnosis."""
        probs = np.array([0.85, 0.10, 0.05], dtype=np.float64)
        result = evaluate_uncertainty(probs, self.classes)

        self.assertTrue(result["is_conclusive"])
        self.assertEqual(result["predicted_class"], "Tomato___Early_blight")
        self.assertAlmostEqual(result["top_score"], 0.85)
        self.assertAlmostEqual(result["margin"], 0.75)
        self.assertEqual(result["status_label"], "Tomato___Early_blight")
        self.assertEqual(len(result["reasons"]), 0)

    def test_inconclusive_due_to_low_score(self) -> None:
        """Top score below 0.60 must flag inconclusive even if margin seems moderate."""
        probs = np.array([0.55, 0.25, 0.20], dtype=np.float64)
        result = evaluate_uncertainty(probs, self.classes)

        self.assertFalse(result["is_conclusive"])
        self.assertEqual(result["status_label"], "Inconclusive — please try another image")
        self.assertTrue(any("below provisional threshold" in r for r in result["reasons"]))

    def test_inconclusive_due_to_narrow_margin(self) -> None:
        """Narrow difference between top 2 classes must flag inconclusive."""
        probs = np.array([0.48, 0.45, 0.07], dtype=np.float64)
        result = evaluate_uncertainty(probs, self.classes)

        self.assertFalse(result["is_conclusive"])
        self.assertEqual(result["status_label"], "Inconclusive — please try another image")
        self.assertAlmostEqual(result["margin"], 0.03)
        # Margin reason must be present
        self.assertTrue(any("margin" in r.lower() for r in result["reasons"]))

    def test_configurable_thresholds(self) -> None:
        """Thresholds must be configurable and alter classification status accordingly."""
        # Score = 0.65, Margin = 0.10
        probs = np.array([0.65, 0.55 / 2 + 0.10 / 2, 0.25], dtype=np.float64)
        # Probs: [0.65, 0.325, 0.025] -> Margin = 0.325
        borderline_probs = np.array([0.65, 0.55, 0.00], dtype=np.float64)
        borderline_probs = borderline_probs / np.sum(borderline_probs)  # normalize to 1.0
        # Margin: 0.65/1.2 - 0.55/1.2 = 0.10 / 1.2 = ~0.0833

        # Default margin threshold is 0.15 -> should be inconclusive
        default_res = evaluate_uncertainty(borderline_probs, self.classes)
        self.assertFalse(default_res["is_conclusive"])

        # Relaxed margin threshold of 0.05 -> should become conclusive
        relaxed_res = evaluate_uncertainty(
            borderline_probs,
            self.classes,
            min_score_threshold=0.50,
            min_margin_threshold=0.05,
        )
        self.assertTrue(relaxed_res["is_conclusive"])


class TestModelCheckpointIntegration(unittest.TestCase):
    """Integration test verifying primary baseline checkpoint structure and forward pass."""

    @classmethod
    def setUpClass(cls) -> None:
        import tensorflow as tf
        cls.tf = tf
        cls.model_path = DEFAULT_MODEL_PATH
        cls.assertTrue(cls.model_path.is_file(), f"Model checkpoint must exist at {cls.model_path}")
        cls.model = tf.keras.models.load_model(cls.model_path)

    def test_model_input_and_output_shapes(self) -> None:
        """Baseline model must have (None, 224, 224, 3) input and (None, 3) output."""
        self.assertEqual(self.model.input_shape, (None, 224, 224, 3))
        self.assertEqual(self.model.output_shape, (None, 3))

    def test_output_layer_softmax_activation(self) -> None:
        """Output layer activation must be softmax."""
        last_layer = self.model.layers[-1]
        self.assertEqual(last_layer.activation.__name__, "softmax")

    def test_forward_pass_produces_valid_probabilities(self) -> None:
        """Synthetic forward pass must produce finite probabilities summing to 1.0."""
        synthetic_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        raw_pred = self.model.predict(synthetic_input, verbose=0)[0]

        self.assertEqual(raw_pred.shape, (3,))
        validate_probabilities(np.asarray(raw_pred, dtype=np.float64), 3)


if __name__ == "__main__":
    unittest.main()
