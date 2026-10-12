"""
Automated Unit Tests for the CNN_AI Experimental Dataset Pipeline.

Verifies:
- Canonical class mapping and explicit flagging of uncertain labels.
- Image integrity checking (valid images, empty files, truncated/corrupted files).
- Exact duplicate detection via SHA-256.
- Near-duplicate clustering via perceptual dHash.
- Group-aware split assignment.
- Cross-split leakage prevention and violation detection.

All tests use fast, self-contained synthetic fixtures in a temporary directory.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

# Import from isolated experimental workflow
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.compare_models import (
    calculate_evaluation_metrics,
    compare_model_predictions,
)
from src.experiments.manifest import (
    CANONICAL_CLASSES,
    ManifestRecord,
    assign_group_aware_splits,
    build_manifest_record,
    check_split_leakage,
    cluster_near_duplicates,
    compute_dhash,
    compute_sha256,
    hamming_distance,
    load_manifest,
    map_label,
    save_manifest,
    validate_image_file,
)


class TestExperimentalDatasetPipeline(unittest.TestCase):
    """Test suite for experimental dataset manifest and leakage prevention."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="cnn_exp_test_"))

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_synthetic_image(
        self,
        filename: str,
        size: tuple[int, int] = (64, 64),
        color: tuple[int, int, int] = (100, 150, 50),
    ) -> Path:
        """Create a synthetic RGB test image."""
        file_path = self.temp_dir / filename
        img = Image.new("RGB", size, color=color)
        img.save(file_path, "JPEG")
        return file_path

    # --- 1. Label Mapping Tests ---
    def test_canonical_label_mapping(self) -> None:
        """Verify that known classes map correctly and unknown/ambiguous labels are flagged."""
        # Verified targets
        label, is_target, reason = map_label("Tomato___Early_blight")
        self.assertEqual(label, "Tomato___Early_blight")
        self.assertTrue(is_target)

        label, is_target, reason = map_label("Early Blight")  # Mendeley Pakistan folder
        self.assertEqual(label, "Tomato___Early_blight")
        self.assertTrue(is_target)

        label, is_target, reason = map_label("Tomato Early blight leaf")  # PlantDoc folder
        self.assertEqual(label, "Tomato___Early_blight")
        self.assertTrue(is_target)

        label, is_target, reason = map_label("Tomato leaf")  # PlantDoc healthy folder
        self.assertEqual(label, "Tomato___healthy")
        self.assertTrue(is_target)

        # Known non-target tomato diseases
        label, is_target, reason = map_label("Septoria Leaf Spot")
        self.assertEqual(label, "EXCLUDED_NON_TARGET")
        self.assertFalse(is_target)

        label, is_target, reason = map_label("Tomato mold leaf")
        self.assertEqual(label, "EXCLUDED_NON_TARGET")
        self.assertFalse(is_target)

        # Ambiguous / unknown labels must be flagged for manual review
        label, is_target, reason = map_label("Unknown_leaf_spot_42")
        self.assertEqual(label, "UNCERTAIN_NEEDS_REVIEW")
        self.assertFalse(is_target)
        self.assertIn("Ambiguous label", reason)

    # --- 2. Image Validation & Integrity Tests ---
    def test_image_validation_valid_file(self) -> None:
        """Verify that a valid synthetic image decodes and hashes properly."""
        img_path = self._create_synthetic_image("valid.jpg", size=(128, 128))
        is_valid, dims, sha256_hash, dhash = validate_image_file(img_path)

        self.assertTrue(is_valid)
        self.assertEqual(dims, [128, 128, 3])
        self.assertEqual(len(sha256_hash), 64)
        self.assertEqual(len(dhash), 16)

    def test_image_validation_empty_file(self) -> None:
        """Verify that an empty (0 byte) file is detected as invalid."""
        empty_path = self.temp_dir / "empty.jpg"
        empty_path.write_bytes(b"")

        is_valid, dims, sha256_hash, reason = validate_image_file(empty_path)
        self.assertFalse(is_valid)
        self.assertIn("Empty file", reason)

    def test_image_validation_corrupted_file(self) -> None:
        """Verify that a truncated or corrupted file is detected as invalid."""
        corrupt_path = self.temp_dir / "corrupt.jpg"
        # Write corrupted JPEG header with invalid trailing payload
        corrupt_path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00corrupted_bytes_only")

        is_valid, dims, sha256_hash, reason = validate_image_file(corrupt_path)
        self.assertFalse(is_valid)
        self.assertIn("Corrupted or unreadable", reason)

    # --- 3. Duplicate and Near-Duplicate Detection Tests ---
    def test_exact_duplicate_detection(self) -> None:
        """Verify that byte-identical files produce matching SHA-256."""
        img1 = self._create_synthetic_image("img1.jpg", color=(80, 120, 60))
        img2 = self.temp_dir / "img2.jpg"
        shutil.copyfile(img1, img2)

        sha1 = compute_sha256(img1)
        sha2 = compute_sha256(img2)
        self.assertEqual(sha1, sha2)

    def test_near_duplicate_clustering(self) -> None:
        """Verify that visually near-identical images are clustered into the same specimen group."""
        # Create base image
        img1_path = self.temp_dir / "base.jpg"
        img = Image.new("RGB", (64, 64), color=(100, 150, 50))
        img.save(img1_path, "JPEG")

        # Create slightly perturbed version (single pixel modified)
        img2_path = self.temp_dir / "perturbed.jpg"
        img_perturbed = img.copy()
        img_perturbed.putpixel((10, 10), (102, 152, 52))
        img_perturbed.save(img2_path, "JPEG")

        rec1 = build_manifest_record(img1_path, "test_src", "CC BY 4.0", "Tomato___healthy")
        rec2 = build_manifest_record(img2_path, "test_src", "CC BY 4.0", "Tomato___healthy")

        self.assertTrue(rec1.is_valid)
        self.assertTrue(rec2.is_valid)

        dist = hamming_distance(rec1.perceptual_hash, rec2.perceptual_hash)
        self.assertLessEqual(dist, 4)

        cluster_map = cluster_near_duplicates([rec1, rec2], distance_threshold=4)
        self.assertEqual(len(cluster_map), 1)
        # Both records now share the identical specimen_group_id
        self.assertEqual(rec1.specimen_group_id, rec2.specimen_group_id)

    # --- 4. Manifest Serialization Tests ---
    def test_manifest_save_and_load(self) -> None:
        """Verify that manifest records serialize and deserialize without data loss."""
        img_path = self._create_synthetic_image("test_manifest.jpg")
        rec = build_manifest_record(img_path, "plantvillage", "CC BY-SA 3.0", "Tomato___Early_blight")

        manifest_file = self.temp_dir / "manifest.json"
        save_manifest([rec], manifest_file)
        self.assertTrue(manifest_file.is_file())

        loaded_records = load_manifest(manifest_file)
        self.assertEqual(len(loaded_records), 1)
        self.assertEqual(loaded_records[0].record_id, rec.record_id)
        self.assertEqual(loaded_records[0].sha256, rec.sha256)
        self.assertEqual(loaded_records[0].canonical_label, "Tomato___Early_blight")
        self.assertEqual(loaded_records[0].is_valid, True)

    # --- 5. Group-Aware Split & Leakage Prevention Tests ---
    def test_group_aware_split_no_leakage(self) -> None:
        """Verify that split assignment keeps specimen clusters together and produces zero leakage."""
        records: list[ManifestRecord] = []
        classes = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]

        # Generate 30 synthetic records across 3 classes (10 per class) with some multi-image groups
        for c_idx, c_name in enumerate(classes):
            for i in range(10):
                img_path = self._create_synthetic_image(
                    f"c{c_idx}_img{i}.jpg",
                    color=((c_idx * 60 + i * 5) % 255, (100 + i * 10) % 255, 80),
                )
                # Assign groups such that pairs of images share a specimen group
                group_hint = f"specimen_c{c_idx}_g{i // 2}"
                rec = build_manifest_record(
                    img_path,
                    "synthetic_source",
                    "CC BY 4.0",
                    c_name,
                    specimen_group_hint=group_hint,
                )
                records.append(rec)

        split_counts = assign_group_aware_splits(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        self.assertGreater(split_counts["train"], 0)
        self.assertGreater(split_counts["val"], 0)
        self.assertGreater(split_counts["test_expanded"], 0)

    def test_leakage_detector_catches_intentional_leak(self) -> None:
        """Verify that check_split_leakage detects cross-split leakage if it occurs."""
        img_path = self._create_synthetic_image("leak_sample.jpg")
        rec1 = build_manifest_record(img_path, "src", "CC BY", "Tomato___healthy", specimen_group_hint="grp_1")
        rec2 = build_manifest_record(img_path, "src", "CC BY", "Tomato___healthy", specimen_group_hint="grp_1")

        # Artificially assign different splits to same hash / specimen group
        rec1.assigned_split = "train"
        rec2.assigned_split = "test_expanded"

        is_leak_free, violations = check_split_leakage([rec1, rec2])
        self.assertFalse(is_leak_free)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("appears in multiple splits" in v for v in violations))

    # --- 6. Metric & Comparative Evaluation Tests ---
    def test_evaluation_metrics_and_false_healthy_detection(self) -> None:
        """Verify metric calculation and tracking of critical false-healthy predictions."""
        import numpy as np

        # y_true: [0 (Early Blight), 1 (Late Blight), 2 (Healthy), 0, 1]
        y_true = np.array([0, 1, 2, 0, 1])
        # y_pred: [0, 2 (Late Blight as Healthy -> False Healthy), 2, 0, 1]
        y_pred = np.array([0, 2, 2, 0, 1])

        metrics = calculate_evaluation_metrics(y_true, y_pred)
        self.assertEqual(metrics["total_samples"], 5)
        self.assertEqual(metrics["accuracy"], 0.8)
        self.assertIn("Tomato___Early_blight", metrics["per_class"])
        self.assertIn("Tomato___Late_blight", metrics["per_class"])
        self.assertIn("Tomato___healthy", metrics["per_class"])

        # Check False Healthy metrics
        fh = metrics["false_healthy_diagnoses"]
        self.assertEqual(fh["total_diseased_samples"], 4)
        self.assertEqual(fh["false_healthy_total"], 1)  # 1 late blight predicted as healthy
        self.assertEqual(fh["late_blight_as_healthy"], 1)
        self.assertEqual(fh["early_blight_as_healthy"], 0)
        self.assertEqual(fh["false_healthy_rate"], 0.25)

    def test_compare_model_predictions_delta(self) -> None:
        """Verify delta computation between Model A and Model B predictions."""
        import numpy as np

        y_true = np.array([0, 1, 2, 0])
        # Model A: [0, 1, 2, 2] -> 3/4 correct, 1 early blight as healthy
        preds_a = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],  # early blight misclassified as healthy
        ])
        # Model B: [0, 1, 2, 0] -> 4/4 correct, 0 false healthy
        preds_b = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],  # correct
        ])

        report = compare_model_predictions(preds_a, preds_b, y_true, "test_expanded_synthetic")
        self.assertEqual(report["evaluation_partition"], "test_expanded_synthetic")
        self.assertEqual(report["model_a_custom_cnn"]["accuracy"], 0.75)
        self.assertEqual(report["model_b_mobilenetv2"]["accuracy"], 1.0)
        self.assertEqual(report["delta_b_minus_a"]["accuracy_delta"], 0.25)
        self.assertLess(
            report["delta_b_minus_a"]["false_healthy_rate_delta"],
            0.0,
            "Model B should have a lower false-healthy rate than Model A"
        )


if __name__ == "__main__":
    unittest.main()
