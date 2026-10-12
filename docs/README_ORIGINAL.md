# AgriAI: Tomato Leaf Disease Classification Using CNN

## 1. Project Overview

**AgriAI** is an end-to-end, reproducible deep learning system for automated identification and classification of tomato leaf diseases from digital RGB images. Built with TensorFlow and Keras, the project provides a verified convolutional neural network (CNN) pipeline that classifies tomato foliage into three agronomic categories: Early Blight, Late Blight, and Healthy.

The repository includes reproducible dataset preparation, baseline model architecture, validation-only hyperparameter tuning experiments, final held-out test evaluation, single-image prediction tooling, and comprehensive technical documentation.

---

## 2. Problem Statement

Tomato (*Solanum lycopersicum*) is one of the world's most widely cultivated vegetable crops, but foliar diseases pose severe threats to crop yields and food security. Fungal and oomycete pathogens—primarily **Early Blight** (*Alternaria solani*) and **Late Blight** (*Phytophthora infestans*)—can cause rapid defoliation and devastating yield losses if not detected and managed promptly.

Traditional disease diagnosis relies on manual visual inspection by agricultural experts, which is often slow, labor-intensive, and inaccessible to smallholder farmers in remote regions. Automated image-based classification using deep learning provides rapid, accessible, and scalable diagnostic assistance, enabling targeted interventions and reducing unnecessary broad-spectrum chemical applications.

---

## 3. Objectives

The primary objectives of this project are:
- **Automated Classification**: Build a CNN capable of classifying RGB tomato leaf images into three target classes (*Early Blight*, *Late Blight*, and *Healthy*).
- **Early Blight Detection**: Accurately detect concentric ring lesions and necrosis characteristic of *Alternaria solani*.
- **Late Blight Detection**: Accurately identify water-soaked lesions and pale halos characteristic of *Phytophthora infestans*.
- **Healthy Leaf Identification**: Differentiate disease-free foliage with high precision to prevent false alarms.
- **Methodological Rigor**: Enforce strict data isolation between training, validation, and test partitions to ensure credible benchmarks.
- **Architectural Exploration**: Benchmark the baseline CNN against deeper architectures and hyperparameter variations using validation data only.

---

## 4. Dataset

The project uses the RGB tomato leaf subset from the public **PlantVillage** dataset:
- **Classes**:
  1. `Tomato___Early_blight`
  2. `Tomato___Late_blight`
  3. `Tomato___healthy`
- **Total Verified Images**: **4,486 unique, valid images** (from 4,500 raw files; 14 exact byte-identical duplicates within classes were excluded prior to splitting).
- **Per-Class Breakdown**:
  - `Tomato___Early_blight`: 1,000 images
  - `Tomato___Late_blight`: 1,901 images
  - `Tomato___healthy`: 1,585 images

### Partitioning & Split Policy
The dataset is partitioned deterministically using random seed **42**:
- **Training Set (70%)**: 3,140 images
- **Validation Set (15%)**: 673 images
- **Test Set (15%)**: 673 images (strictly held out until final evaluation)

All splits maintain identical class proportions (stratification). SHA-256 hash verification confirms zero cross-split image overlap.

---

## 5. Preprocessing & Data Augmentation

- **RGB Color Space**: All images are loaded and decoded as 3-channel RGB.
- **Resolution**: Resized to **224 × 224** pixels using bilinear interpolation.
- **Internal Normalization**: The model incorporates an initial `Rescaling(1.0 / 255.0)` layer, mapping raw `[0, 255]` pixel intensities into `[0.0, 1.0]`.
- **Data Augmentation** (active during training mode only):
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.05)` (±5% rotational perturbation)
  - `RandomZoom(0.10)` (±10% scale perturbation)
- **Evaluation / Inference**: Augmentation layers automatically deactivate during validation, testing, and standalone inference (`training=False`), ensuring fully deterministic predictions.

---

## 6. CNN Architecture

The finalized model is a sequential Convolutional Neural Network designed to retain fine-grained spatial lesion patterns:

```text
Input (224 × 224 × 3)
   │
   ▼
Rescaling (1/255) & Training Augmentation
   │
   ▼
Conv2D (32 filters, 3×3, ReLU, padding='same') ──► MaxPooling2D (2×2)
   │
   ▼
Conv2D (64 filters, 3×3, ReLU, padding='same') ──► MaxPooling2D (2×2)
   │
   ▼
Conv2D (128 filters, 3×3, ReLU, padding='same') ──► MaxPooling2D (2×2)
   │
   ▼
Flatten (73,728 units)
   │
   ▼
Dense (128 units, ReLU)
   │
   ▼
Dropout (rate = 0.5)
   │
   ▼
Dense (3 units, Softmax) ──► Class Probabilities
```

- **Total Parameter Count**: **12,938,819 parameters** (100% trainable, 0 non-trainable).
- **Optimizer**: Adam (initial learning rate = 0.001).
- **Loss Function**: Sparse Categorical Cross-Entropy.
- **Training Callbacks**:
  - `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)`
  - `ModelCheckpoint(monitor='val_accuracy', mode='max', save_best_only=True)`

The baseline trained for 19 epochs (max 30); EarlyStopping restored the optimal weights from **epoch 14**.

---

## 7. Model Development & Phase 7 Experiments

During Phase 7, five candidate models were trained and evaluated exclusively on training and validation splits to test whether deeper networks (4 convolutional blocks with Batch Normalization and Global Average Pooling) or alternate hyperparameters could outperform the baseline:

| Experiment | Architecture / Configuration | Best Validation Accuracy | Best Validation Loss |
|---|---|---:|---:|
| **Baseline** | **Conv 32/64/128 + Flatten + Dense 128 (lr=0.001, bs=32, drop=0.5)** | **96.58%** | **0.086766** |
| Experiment 1 (`p7_proposal_lr1e-3_bs32_drop50`) | 4-block Conv-BN + GAP + Dense 128 (lr=0.001, bs=32, drop=0.5) | 91.08% | 0.266655 |
| Experiment 2 (`p7_proposal_lr3e-4_bs32_drop50`) | 4-block Conv-BN + GAP + Dense 128 (lr=0.0003, bs=32, drop=0.5) | 94.21% | 0.138650 |
| Experiment 3 (`p7_proposal_lr1e-3_bs32_drop30`) | 4-block Conv-BN + GAP + Dense 128 (lr=0.001, bs=32, drop=0.3) | 94.95% | 0.181050 |
| Experiment 4 (`p7_proposal_lr1e-3_bs64_drop50`) | 4-block Conv-BN + GAP + Dense 128 (lr=0.001, bs=64, drop=0.5) | 70.88% | 0.996171 |
| Experiment 5 (`p7_compact_lr1e-3_bs32_drop50`) | 3-block Conv-BN + GAP + Dense 128 (lr=0.001, bs=32, drop=0.5) | 87.22% | 0.350240 |

**Model Selection Decision**:
> The baseline remained the best-performing model based on validation accuracy. None of the five Phase 7 experiments exceeded the baseline validation accuracy (96.58%), so the baseline was selected as the final production model.

---

## 8. Final Results

Following model selection, the finalized model ([models/final_crop_disease_cnn.keras](models/final_crop_disease_cnn.keras)) was evaluated on the held-out **673-image test split**:

| Metric | Measured Value |
|---|---:|
| **Test Accuracy** | **94.80%** (638 / 673 correct) |
| **Test Loss** | **0.122676** |
| **Weighted Precision** | **94.83%** |
| **Weighted Recall** | **94.80%** |
| **Weighted F1-Score** | **94.81%** |
| **Macro Precision** | **94.05%** |
| **Macro Recall** | **94.30%** |
| **Macro F1-Score** | **94.17%** |

The difference between best validation accuracy (96.58%) and test accuracy (94.80%) is −1.78 percentage points, demonstrating strong generalization without severe overfitting.

---

## 9. Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Tomato___Early_blight` | **88.24%** | **90.00%** | **89.11%** | 150 |
| `Tomato___Late_blight` | **94.33%** | **93.33%** | **93.83%** | 285 |
| `Tomato___healthy` | **99.58%** | **99.58%** | **99.58%** | 238 |

- **Healthy leaves** achieved near-perfect identification (99.58% precision and recall).
- **Early and Late Blight** exhibited balanced recall (90.00% and 93.33% respectively).

---

## 10. Confusion Matrix

The confusion matrix on the 673 test images (rows = actual ground truth, columns = model prediction) is saved at [results/final_confusion_matrix.png](results/final_confusion_matrix.png):

```text
Actual \ Predicted       Early Blight    Late Blight      Healthy      Total
Tomato___Early_blight        135             15              0          150
Tomato___Late_blight          18            266              1          285
Tomato___healthy               0              1            237          238
Total Predicted              153            282            238          673
```

### Key Diagnostic Patterns:
- **Healthy vs. Diseased Discrimination**: Clean separation. Only 1 healthy leaf was misclassified as late blight, and only 1 late blight leaf was misclassified as healthy.
- **Blight Overlap**: The principal source of error occurs between Early Blight and Late Blight (15 Early Blight images predicted as Late Blight; 18 Late Blight images predicted as Early Blight). This corresponds to real-world visual similarity in necrotic foliar lesions at early developmental stages.

## Running the AgriAI Web App

To run the interactive AgriAI web interface locally:

1. Ensure the existing trained model file is present at `models/final_crop_disease_cnn.keras`.
2. Open a terminal in the project root directory.
3. Install dependencies and launch Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`), allowing you to upload tomato leaf photographs in JPG, JPEG, or PNG format and view instant diagnostic classifications with confidence metrics and probability breakdowns.

---

## 11. Prediction & Web Interface

### A. Streamlit Web Application (Interactive GUI)
Launch the interactive local web dashboard for point-and-click image uploads and diagnostic predictions:

```bash
streamlit run app.py
```

The web app:
- Provides a drag-and-drop file uploader for JPG, JPEG, and PNG leaf photos.
- Displays the uploaded photograph with resolution metadata.
- Preprocesses and runs inference with the final CNN model (`models/final_crop_disease_cnn.keras`).
- Displays the predicted disease classification with confidence percentage.
- Renders full 3-class probability progress bars and agronomic pathology profiles.
- Includes essential agricultural and technical disclaimers.

### B. Command-Line Interface (CLI)
A standalone CLI script is also available for batch or terminal predictions:

```powershell
.venv\Scripts\python.exe src\predict.py --image "path\to\leaf_image.jpg"
```

The script:
1. Loads the finalized CNN checkpoint (`models/crop_disease_cnn.keras`).
2. Converts the input image to 3-channel RGB.
3. Resizes the image to 224 × 224 pixels.
4. Feeds raw pixel intensities to the model (normalization is handled internally by the Rescaling layer).
5. Returns the predicted class label, confidence percentage, and full 3-class probability distribution.

### Sample Prediction Output:
```text
Predicted class: Tomato___healthy
Confidence: 99.98%
Class probabilities:
  Tomato___Early_blight: 0.00%
  Tomato___Late_blight: 0.02%
  Tomato___healthy: 99.98%
```

---

## 12. Installation & Setup

### Requirements:
- Windows OS (or Linux/macOS)
- Python 3.12+

### Step-by-Step Setup:

1. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit web interface**:
   ```powershell
   streamlit run app.py
   ```

4. **Run a CLI test prediction**:
   ```powershell
   python src\predict.py --image "data\validation\Tomato___healthy\000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG"
   ```

5. **(Optional) Re-run final test evaluation**:
   ```powershell
   python src\evaluate_final.py
   ```

---

## 13. Project Structure

```text
CNN_AI/
├── app.py                                # Streamlit interactive web application
├── data/                                 # Stratified local dataset (excluded from Git)
│   ├── train/                            # 3,140 training images (70%)
│   ├── validation/                       # 673 validation images (15%)
│   ├── test/                             # 673 test images (15%)
│   └── raw/                              # Original source images
├── docs/                                 # Technical documentation
│   ├── DATASET.md                        # Dataset preparation & audit
│   ├── MODEL.md                          # Architecture specification
│   ├── TRAINING.md                       # Baseline training records
│   ├── TESTING.md                        # Phase 6 test evaluation
│   ├── MODEL_IMPROVEMENT.md              # Phase 7 tuning experiments
│   ├── FINAL_MODEL_SELECTION.md          # Phase 8 selection audit
│   ├── FINAL_EVALUATION.md               # Phase 8 test evaluation report
│   ├── PRESENTATION_OUTLINE.md           # 12-slide presentation outline
│   ├── FINAL_STATUS.md                   # Formal project completion status
│   └── PROJECT_COMPLETION_REPORT.md      # Comprehensive wrap-up report
├── models/                               # Keras model checkpoints
│   ├── crop_disease_cnn.keras            # Baseline model checkpoint
│   ├── final_crop_disease_cnn.keras      # Final model checkpoint (copy of baseline)
│   └── experiments/                      # Phase 7 experiment checkpoints
├── results/                              # Evaluation outputs and metrics
│   ├── final_results_summary.csv         # Overall model results summary
│   ├── hyperparameter_experiments.csv    # Phase 7 experiment comparison table
│   ├── phase7_histories/                 # Phase 7 per-epoch JSON logs
│   ├── final_test_results.json           # Phase 8 final test metrics
│   ├── final_classification_report.txt   # Phase 8 formatted report
│   ├── final_confusion_matrix.png        # Phase 8 confusion matrix figure
│   ├── model_architecture.png            # Visual architecture diagram
│   ├── baseline_vs_final.csv             # Baseline vs final comparison
│   ├── training_accuracy.png             # Baseline accuracy curve
│   └── training_loss.png                 # Baseline loss curve
├── src/                                  # Source code
│   ├── prepare_dataset.py                # Dataset deduplication & splitting
│   ├── model.py                          # Baseline model definition
│   ├── train.py                          # Baseline training runner
│   ├── evaluate.py                       # Phase 6 test evaluator
│   ├── tune.py                           # Phase 7 hyperparameter experiment runner
│   ├── summarize_phase7.py               # Phase 7 artifact validator
│   ├── select_final_model.py             # Phase 8 validation-only model selector
│   ├── evaluate_final.py                 # Phase 8 test evaluation runner
│   └── predict.py                        # Single-image inference CLI
├── .gitignore                            # Excludes dataset, venv, cache, and binaries
├── README.md                             # Main GitHub README
└── requirements.txt                      # Pinned Python package dependencies
```

---

## 14. Limitations

1. **Controlled Photography Domain**: PlantVillage images consist of leaves photographed against uniform laboratory backgrounds. Performance may degrade on complex field imagery featuring soil, shadows, overlapping leaves, and direct sunlight.
2. **External Real-World Imagery**: Real-world external images may yield lower confidence or misclassifications due to domain shift. High model confidence does not guarantee clinical agricultural correctness.
3. **Three-Class Scope**: The system supports only Early Blight, Late Blight, and Healthy leaves. Other tomato pathologies (e.g., *Septoria* leaf spot, Bacterial spot, Tomato Yellow Leaf Curl Virus, iron chlorosis) will be forced into one of the three supported classes.
4. **Near-Duplicate Imagery**: Exact SHA-256 duplicate removal eliminates byte-identical files, but subtle near-duplicate photographs of the same physical leaf taken from slightly different angles may remain across split partitions.

---

## 15. Future Improvements

1. **In-the-Wild Dataset Expansion**: Fine-tune on heterogeneous outdoor farm datasets (e.g., PlantDoc) to improve robustness under complex farm illumination and backgrounds.
2. **Multi-Class Disease Expansion**: Extend coverage to the full spectrum of 10+ tomato pathologies and nutrient deficiencies.
3. **Transfer Learning**: Benchmark pre-trained vision backbones (e.g., MobileNetV3, EfficientNetV2, ConvNeXt) to compare against custom CNN representations.
4. **Model Explainability**: Integrate Grad-CAM (Gradient-weighted Class Activation Mapping) to visually highlight foliar regions driving predictions.
5. **Edge & Mobile Deployment**: Quantize the model using TensorFlow Lite (INT8/FP16) for real-time, offline mobile deployment on edge devices.
