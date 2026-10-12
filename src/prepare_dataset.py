"""Validate, deduplicate, and reproducibly split PlantVillage tomato images.

The script uses only the Python standard library and Pillow. It never changes
files in the raw source directory. Split outputs are copied into data/train,
data/validation, and data/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError

CLASSES = (
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ImageRecord:
    """A validated, unique image and its class label."""

    path: Path
    class_name: str
    sha256: str


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset directory (default: data/raw).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Directory receiving train/validation/test folders (default: data).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Reproducible shuffle seed.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the three output split directories if they already exist.",
    )
    return parser.parse_args()


def find_class_directory(source_root: Path, class_name: str) -> Path | None:
    """Find a class folder directly or under common PlantVillage layouts."""
    candidates = [
        source_root / class_name,
        source_root / "raw" / "color" / class_name,
        source_root / "color" / class_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    matches = [path for path in source_root.rglob(class_name) if path.is_dir()]
    if len(matches) > 1:
        raise ValueError(f"Found multiple directories for {class_name}: {matches}")
    return matches[0] if matches else None


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_collect(source_root: Path) -> tuple[dict[str, list[ImageRecord]], dict[str, object]]:
    """Inspect required class folders, skip bad files, and group exact duplicates."""
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {source_root}. Obtain PlantVillage "
            "and place the three required class folders under this directory."
        )

    class_dirs: dict[str, Path] = {}
    missing_classes: list[str] = []
    for class_name in CLASSES:
        class_dir = find_class_directory(source_root, class_name)
        if class_dir is None:
            missing_classes.append(class_name)
        else:
            class_dirs[class_name] = class_dir
    if missing_classes:
        raise FileNotFoundError(
            "Required class folder(s) missing under "
            f"{source_root}: {', '.join(missing_classes)}"
        )

    candidates: list[tuple[Path, str]] = []
    unsupported_files = 0
    raw_counts: Counter[str] = Counter()
    for class_name, class_dir in class_dirs.items():
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                unsupported_files += 1
                continue
            raw_counts[class_name] += 1
            candidates.append((path, class_name))

    records_by_hash: dict[str, ImageRecord] = {}
    corrupt_files: list[str] = []
    duplicate_filenames: Counter[str] = Counter()
    filenames_seen: set[tuple[str, str]] = set()
    duplicate_images = 0

    for path, class_name in candidates:
        filename_key = (class_name, path.name.casefold())
        if filename_key in filenames_seen:
            duplicate_filenames[f"{class_name}/{path.name}"] += 1
        filenames_seen.add(filename_key)

        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            corrupt_files.append(str(path))
            continue

        digest = sha256_file(path)
        existing = records_by_hash.get(digest)
        if existing is not None:
            if existing.class_name != class_name:
                raise ValueError(
                    "The same image bytes have conflicting class labels: "
                    f"{existing.path} ({existing.class_name}) and {path} ({class_name})"
                )
            duplicate_images += 1
            continue
        records_by_hash[digest] = ImageRecord(path=path, class_name=class_name, sha256=digest)

    records_by_class: dict[str, list[ImageRecord]] = {name: [] for name in CLASSES}
    for record in records_by_hash.values():
        records_by_class[record.class_name].append(record)
    for records in records_by_class.values():
        records.sort(key=lambda record: str(record.path).casefold())

    diagnostics: dict[str, object] = {
        "raw_supported_image_files_per_class": dict(raw_counts),
        "unique_valid_images_per_class": {
            name: len(records) for name, records in records_by_class.items()
        },
        "corrupted_or_unreadable_files": corrupt_files,
        "unsupported_files_skipped": unsupported_files,
        "exact_duplicate_files_skipped": duplicate_images,
        "duplicate_filename_occurrences": sum(duplicate_filenames.values()),
        "duplicate_filenames": dict(duplicate_filenames),
    }
    return records_by_class, diagnostics


def split_class_records(
    records: Sequence[ImageRecord], seed: int, class_name: str
) -> dict[str, list[ImageRecord]]:
    """Shuffle one class deterministically and allocate nearest 70/15/15 counts."""
    shuffled = list(records)
    class_seed = seed ^ int(hashlib.sha256(class_name.encode("utf-8")).hexdigest()[:8], 16)
    random.Random(class_seed).shuffle(shuffled)

    exact = {name: len(shuffled) * ratio for name, ratio in SPLITS.items()}
    counts = {name: int(exact[name]) for name in SPLITS}
    remaining = len(shuffled) - sum(counts.values())
    remainders = sorted(SPLITS, key=lambda name: (exact[name] - counts[name], name), reverse=True)
    for name in remainders[:remaining]:
        counts[name] += 1
    if any(counts[name] == 0 for name in SPLITS):
        raise ValueError(
            f"Class {class_name} has too few unique images ({len(shuffled)}) to populate all splits."
        )

    result: dict[str, list[ImageRecord]] = {}
    offset = 0
    for split_name in SPLITS:
        result[split_name] = shuffled[offset : offset + counts[split_name]]
        offset += counts[split_name]
    return result


def prepare_output_directories(output_root: Path, force: bool) -> None:
    """Create empty split targets, refusing to overwrite unless requested."""
    split_roots = [output_root / split_name for split_name in SPLITS]
    existing = [path for path in split_roots if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Split output already exists: "
            + ", ".join(str(path) for path in existing)
            + ". Review it or rerun with --force to replace these split directories."
        )
    if force:
        for path in existing:
            shutil.rmtree(path)
    for path in split_roots:
        path.mkdir(parents=True, exist_ok=True)
        for class_name in CLASSES:
            (path / class_name).mkdir(parents=True, exist_ok=True)


def copy_splits(
    records_by_class: dict[str, list[ImageRecord]], output_root: Path, seed: int, force: bool
) -> dict[str, dict[str, int]]:
    """Create the deterministic splits and copy each unique image exactly once."""
    split_records: dict[str, dict[str, list[ImageRecord]]] = {
        split_name: {class_name: [] for class_name in CLASSES} for split_name in SPLITS
    }
    for class_name, records in records_by_class.items():
        class_splits = split_class_records(records, seed, class_name)
        for split_name, class_records in class_splits.items():
            split_records[split_name][class_name] = class_records

    prepare_output_directories(output_root, force)
    used_names: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split_name, class_mapping in split_records.items():
        for class_name, records in class_mapping.items():
            destination_dir = output_root / split_name / class_name
            for record in records:
                output_name = record.path.name
                if output_name.casefold() in used_names[(split_name, class_name)]:
                    output_name = f"{record.path.stem}_{record.sha256[:10]}{record.path.suffix}"
                used_names[(split_name, class_name)].add(output_name.casefold())
                shutil.copy2(record.path, destination_dir / output_name)

    counts = {
        split_name: {class_name: len(records) for class_name, records in class_mapping.items()}
        for split_name, class_mapping in split_records.items()
    }
    return counts


def verify_no_hash_leakage(output_root: Path) -> dict[str, int]:
    """Check output splits for byte-identical files shared between splits."""
    hashes_by_split: dict[str, set[str]] = {}
    for split_name in SPLITS:
        hashes: set[str] = set()
        for path in (output_root / split_name).rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                hashes.add(sha256_file(path))
        hashes_by_split[split_name] = hashes

    names = list(SPLITS)
    intersections: dict[str, int] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            intersections[f"{first}_and_{second}"] = len(
                hashes_by_split[first].intersection(hashes_by_split[second])
            )
    if any(intersections.values()):
        raise RuntimeError(f"Exact-image data leakage detected between splits: {intersections}")
    return intersections


def main() -> None:
    """Run inspection, reproducible splitting, and verification."""
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output_root.resolve()
    records_by_class, diagnostics = validate_and_collect(source_root)
    split_counts = copy_splits(records_by_class, output_root, args.seed, args.force)
    leakage_check = verify_no_hash_leakage(output_root)

    summary = {
        "dataset": "PlantVillage (color tomato classes)",
        "source": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "split_ratios": SPLITS,
        "classes": list(CLASSES),
        "total_unique_valid_images": sum(map(len, records_by_class.values())),
        "diagnostics": diagnostics,
        "counts_by_split": split_counts,
        "exact_duplicate_leakage_check": leakage_check,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
