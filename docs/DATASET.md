# Dataset

## Dataset name and source

This project uses the **PlantVillage** tomato leaf image classes. The dataset is publicly described by Mohanty, Hughes, and Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*, [Frontiers in Plant Science](https://doi.org/10.3389/fpls.2016.01419). The source repository is [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset); an additional dataset card and distribution is available at [mohanty/PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage).

Only the following RGB classes are required:

- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___healthy`

The source repository organizes its color images below `raw/color/<class-name>/`. Place those three folders under this project's `data/raw/` folder. Do not commit the image data to GitHub; `data/` is ignored.

## Verified local dataset

The three requested RGB class folders were selectively retrieved from the official GitHub repository into `data/raw/`. The counts below were measured from those local files by `src/prepare_dataset.py`; upstream catalog values are not substituted for local checks.

### Raw files by class and format

| Class | Local image files | Formats |
|---|---:|---|
| Tomato___Early_blight | 1,000 | `.jpg`: 1,000 |
| Tomato___Late_blight | 1,909 | `.jpg`: 1,908; `.jpeg`: 1 |
| Tomato___healthy | 1,591 | `.jpg`: 1,591 |
| **Total raw supported files** | **4,500** | **`.jpg`: 4,499; `.jpeg`: 1** |

All 4,500 local files were readable; **0 corrupted/unreadable** and **0 unsupported** files were found. There were **14 byte-identical duplicates** within classes (0 Early blight, 8 Late blight, 6 healthy); the preparation script excluded duplicate copies before splitting. It found **0 repeated filenames within any class**.

### Valid unique images and split counts

| Class | Unique valid | Train (70%) | Validation (15%) | Test (15%) |
|---|---:|---:|---:|---:|
| Tomato___Early_blight | 1,000 | 700 | 150 | 150 |
| Tomato___Late_blight | 1,901 | 1,331 | 285 | 285 |
| Tomato___healthy | 1,585 | 1,109 | 238 | 238 |
| **Total** | **4,486** | **3,140** | **673** | **673** |

Split seed: **42**. Splitting was performed separately within each class. The preparation script reported zero exact SHA-256 overlaps between train/validation, train/test, and validation/test. Raw files were copied unchanged; no augmentation occurs during splitting.

To recreate the split, run `python src/prepare_dataset.py --seed 42`. Existing output split folders are protected from overwrite; use `--force` only when intentionally rebuilding them.

## Split and cleaning policy

- Target split: 70% train, 15% validation, 15% test, stratified separately within each class.
- Random seed: 42 (change only with an explicitly recorded experiment).
- Images are decoded/verified with Pillow; unreadable images are reported and excluded. Files with unsupported extensions are counted and skipped.
- Byte-identical files are detected by SHA-256 and duplicate copies in the same class are excluded before splitting. Identical bytes assigned conflicting labels stop preparation with an error.
- Split outputs are copied, leaving the raw source untouched. The script hashes output files to ensure no exact-image duplicates cross split boundaries.
- Augmentation is not part of dataset preparation. Model augmentation layers activate only when the model is called in training mode; validation/inference calls are deterministic.

## Leakage and limitations

The preparation script checks exact byte-identical duplicates and reports duplicate basenames. This is a practical check, not proof that near-duplicate photographs or images of the same physical leaf cannot cross splits. The downloaded image folders do not include verified leaf-group identifiers. A strict leaf-level split would require retrieving and mapping the separate leaf-group metadata before splitting. PlantVillage images are controlled, often uniformly staged leaf photographs and may not represent field conditions, camera variation, or disease prevalence. Results from this dataset should not be treated as a field-ready diagnosis.

## License and citation

The Hugging Face dataset card identifies the dataset as CC BY-SA 3.0. Review the source terms and attribution requirements before redistribution. Dataset citation:

> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419
