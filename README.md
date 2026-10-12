# AgriAI: Tomato Leaf Disease Classification Using CNN

---

## 1. Assignment Executive Summary & Core Deliverables

### Project Goal
**AgriAI** is an end-to-end deep learning system built with **TensorFlow and Keras** for automated identification and classification of tomato foliar diseases from digital RGB leaf photographs.

### Target Classes
The model is trained on three agronomic categories:
1. `0`: `Tomato___Early_blight` (*Alternaria solani*)
2. `1`: `Tomato___Late_blight` (*Phytophthora infestans*)
3. `2`: `Tomato___healthy` (Disease-free foliage)

---

### Dataset & Partitioning
- **Data Source**: Public PlantVillage tomato leaf RGB dataset.
- **Dataset Audit**: 4,486 authentic unique images verified across the 3 classes (14 byte-identical duplicate files excluded prior to partitioning).
- **Partition Policy**: Deterministic stratified split using random seed **42** ensuring identical class proportions across splits with **zero SHA-256 hash leakage**:
  - **Training Set (70%)**: **3,140** images (`Tomato___Early_blight`: 700, `Tomato___Late_blight`: 1,331, `Tomato___healthy`: 1,109)
  - **Validation Set (15%)**: **673** images (`Tomato___Early_blight`: 150, `Tomato___Late_blight`: 285, `Tomato___healthy`: 238)
  - **Held-Out Test Set (15%)**: **673** images (`Tomato___Early_blight`: 150, `Tomato___Late_blight`: 285, `Tomato___healthy`: 238)

---

### Image Preprocessing Pipeline
- **Resolution**: $224 \times 224$ pixels, 3 channels (RGB).
- **Color Format**: 3-channel RGB. Grayscale or alpha channels are converted to standard 3-channel RGB.
- **Normalization Range**: $[0.0, 1.0]$.
  > **Internal Rescaling Layer Invariant**: The model embeds an initial `tf.keras.layers.Rescaling(scale=1.0/255.0)` layer. Inputs must be fed as **raw pixel intensities in $[0, 255]$ (`float32`)**. Pre-normalizing images prior to inference will cause double-scaling and corrupt model outputs.
- **Data Type**: `float32` tensors with shape `(batch_size, 224, 224, 3)`.
- **Training Augmentation** (active during training mode only):
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.05)` (±5% rotational perturbation)
  - `RandomZoom(0.10)` (±10% zoom perturbation)

---

### CNN Architecture Specification
The finalized model is a 3-block convolutional neural network designed to capture spatial lesion features while avoiding parameter explosion:

| Layer # | Layer Name | Layer Type | Input Shape | Output Shape | Parameters | Operational Details |
|:---:|---|---|---|---|---:|---|
| 1 | `rescaling` | `Rescaling` | `(None, 224, 224, 3)` | `(None, 224, 224, 3)` | 0 | $\text{scale}=1/255.0, \text{offset}=0.0$ |
| 2 | `random_horizontal_flip` | `RandomFlip` | `(None, 224, 224, 3)` | `(None, 224, 224, 3)` | 0 | `mode="horizontal"` (training only) |
| 3 | `random_rotation` | `RandomRotation` | `(None, 224, 224, 3)` | `(None, 224, 224, 3)` | 0 | `factor=0.05` (training only) |
| 4 | `random_zoom` | `RandomZoom` | `(None, 224, 224, 3)` | `(None, 224, 224, 3)` | 0 | `factor=0.10` (training only) |
| 5 | `conv2d_32` | `Conv2D` | `(None, 224, 224, 3)` | `(None, 224, 224, 32)` | 896 | 32 filters, $3\times 3$ kernel, stride $(1,1)$, padding `'same'`, activation `'relu'` |
| 6 | `max_pooling_2d_1` | `MaxPooling2D` | `(None, 224, 224, 32)` | `(None, 112, 112, 32)` | 0 | pool $2\times 2$, stride $2\times 2$ |
| 7 | `conv2d_64` | `Conv2D` | `(None, 112, 112, 32)` | `(None, 112, 112, 64)` | 18,496 | 64 filters, $3\times 3$ kernel, stride $(1,1)$, padding `'same'`, activation `'relu'` |
| 8 | `max_pooling_2d_2` | `MaxPooling2D` | `(None, 112, 112, 64)` | `(None, 56, 56, 64)` | 0 | pool $2\times 2$, stride $2\times 2$ |
| 9 | `conv2d_128` | `Conv2D` | `(None, 56, 56, 64)` | `(None, 56, 56, 128)` | 73,856 | 128 filters, $3\times 3$ kernel, stride $(1,1)$, padding `'same'`, activation `'relu'` |
| 10 | `max_pooling_2d_3` | `MaxPooling2D` | `(None, 56, 56, 128)` | `(None, 28, 28, 128)` | 0 | pool $2\times 2$, stride $2\times 2$ |
| 11 | `flatten` | `Flatten` | `(None, 28, 28, 128)` | `(None, 100352)` | 0 | Flatten $28 \times 28 \times 128 = 100,352$ |
| 12 | `dense_128` | `Dense` | `(None, 100352)` | `(None, 128)` | 12,845,184 | 128 units, activation `'relu'` |
| 13 | `dropout_50_percent` | `Dropout` | `(None, 128)` | `(None, 128)` | 0 | dropout rate $0.50$ (training only) |
| 14 | `class_probabilities` | `Dense` | `(None, 128)` | `(None, 3)` | 387 | 3 units, activation `'softmax'` |

- **Trainable Parameters**: **12,938,819** (49.36 MB)
- **Non-trainable Parameters**: **0**

---

### Training Configuration & Hyperparameters
- **Loss Function**: `SparseCategoricalCrossentropy(from_logits=False)`
- **Optimizer**: Adam ($\text{learning\_rate} = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$)
- **Batch Size**: 32
- **Maximum Epochs**: 30 (halted at Epoch 19 via `EarlyStopping`, restoring best weights from Epoch 14)
- **Callbacks**:
  - `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)`
  - `ModelCheckpoint(monitor='val_accuracy', mode='max', save_best_only=True)`

---

### Per-Epoch Training Progression & Diagnosis
Source file: [`results/training_history.csv`](results/training_history.csv)

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Learning Rate | Status / Event |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 0.9508 | 55.45% | 0.8308 | 62.85% | 0.001 | Initial convergence |
| 5 | 0.2278 | 91.15% | 0.2732 | 87.82% | 0.001 | Rapid feature extraction |
| 10 | 0.1917 | 92.96% | 0.1127 | 96.43% | 0.001 | High validation precision |
| **14** | **0.1502** | **93.66%** | **0.0868** | **96.58%** | **0.001** | **Optimal Checkpoint (Restored)** |
| 17 | 0.2405 | 90.67% | 0.1814 | 92.57% | 0.001 | Fluctuation observed |
| 19 | 0.1399 | 94.97% | 0.3081 | 88.11% | 0.001 | EarlyStopping triggered (patience=5) |

**Diagnosis (Underfitting vs. Overfitting):**
- **Underfitting**: Absent. The model rapidly reached high training and validation accuracy ($>93\%$) with low loss ($<0.15$), confirming that model capacity is fully sufficient for the 3 target classes.
- **Overfitting**: Mild divergence observed after Epoch 14 (where validation loss increased from 0.0868 to 0.3081). This was cleanly managed by `EarlyStopping`, which restored the exact weights from **Epoch 14**, preventing overparameterized memorization. Throughout earlier epochs, validation loss was slightly lower than training loss because data augmentation and 50% dropout were active exclusively during training.

---

### Separate Held-Out Test Set Performance
Evaluated on the completely isolated `data/test/` partition (673 images, 0 training/validation overlap):

- **Test Accuracy**: **94.80%** (638 / 673 images correctly classified)
- **Test Loss**: **0.1227**
- **Macro Precision**: 94.05% | **Macro Recall**: 94.30% | **Macro F1-Score**: **94.17%**
- **Weighted Precision**: 94.83% | **Weighted Recall**: 94.80% | **Weighted F1-Score**: **94.81%**

#### Per-Class Performance Table

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `Tomato___Early_blight` | 0.8824 | 0.9000 | **0.8911** | 150 |
| `Tomato___Late_blight` | 0.9433 | 0.9333 | **0.9383** | 285 |
| `Tomato___healthy` | 0.9958 | 0.9958 | **0.9958** | 238 |

#### Confusion Matrix (Held-Out Test Set)

```
                            Predicted Class
Actual Class         Early Blight    Late Blight    Healthy    Total Actual
Early Blight             135             15            0           150
Late Blight               18            266            1           285
Healthy                    0              1          237           238
Total Predicted          153            282          238           673
```

#### Early Blight Performance Analysis
The test confusion matrix reveals why `Tomato___Early_blight` has the lowest F1-score (89.11%):
- **Confusions**: Early Blight is confused exclusively with `Tomato___Late_blight`:
  - 15 actual Early Blight images were misclassified as Late Blight (false negatives).
  - 18 actual Late Blight images were misclassified as Early Blight (false positives).
  - 0 Early Blight images were confused with Healthy leaves.
- Healthy leaves are identified with near-perfect separation (99.58% precision and recall).

---

### Written Conclusion & Recommended Next Steps
1. **Generalization**: The model generalizes effectively to unseen held-out leaf imagery, attaining 94.80% test accuracy matching its 96.58% validation baseline.
2. **Weakest Class**: `Tomato___Early_blight` has the lowest F1-score (89.11%), driven entirely by inter-class confusion with `Tomato___Late_blight`.
3. **Suggested Fine-Tuning Interventions**:
   - Apply class weights during training to compensate for the smaller Early Blight training support (700 images vs. 1,331 Late Blight images).
   - Train with a reduced learning rate ($3 \times 10^{-4}$) to stabilize dense layer feature representations.
   - Fine-tune a lightweight pre-trained transfer learning backbone (such as MobileNetV2) for richer multi-scale foliar lesion representations.

---

### Scope and Limitations
> [!IMPORTANT]
> - **Domain Scope**: The model is trained exclusively on PlantVillage tomato leaf photographs taken under controlled conditions against uniform backgrounds. Accuracy on field or real-world photos has not yet been measured.
> - **Closed-World Output**: The network supports only the 3 trained classes (`Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___healthy`). It **always outputs one of these three classes**, even when presented with other plant crops, other tomato diseases, or arbitrary non-leaf objects.
> - **Empirical OOD Risk**: As documented in Section 3, empirical auditing demonstrates that unsupported tomato diseases produce $>92\%$ false acceptance rates at high confidence under standard closed-world softmax. The model must not be used as an unguided clinical diagnostic tool in open-field farming without expert human verification.

---

### Environment Requirements
- **Python Version**: `Python 3.12.10`
- **Framework**: `tensorflow==2.21.0` (all dependencies pinned in `requirements.txt`)

---

### Copy-Paste Prediction Snippet
To classify a single image using the verified lightweight inference checkpoint:

```python
import numpy as np
import tensorflow as tf

# 1. Load deployable model checkpoint (<50 MB; raw pixel [0, 255] input)
model = tf.keras.models.load_model("models/final_crop_disease_cnn_deploy.keras", compile=False)

# 2. Canonical class order
classes = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]

# 3. Load and resize using TensorFlow bilinear interpolation (identical to training loader)
raw_bytes = tf.io.read_file("path/to/leaf_image.jpg")
img = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
resized = tf.image.resize(img, (224, 224), method="bilinear")
# Feed raw [0, 255] float32 pixels — internal Rescaling layer normalizes to [0, 1]
tensor = tf.expand_dims(tf.cast(resized, tf.float32), axis=0)

# 4. Predict
probs = model(tensor, training=False).numpy()[0]
pred_idx = int(np.argmax(probs))
print(f"Predicted Class: {classes[pred_idx]} ({probs[pred_idx]:.2%})")
```

---

## 2. Interactive Application & CLI Usage

### Streamlit Web Interface
Launch the interactive diagnostic web application:
```powershell
streamlit run app.py
```
- Includes early file-size validation ($\le 15$ MB) and minimum dimension validation ($\ge 64\times 64$ px).
- Provides uncertainty warning banners when top scores or class margins fall below provisional thresholds.

### CLI Single-Image Prediction & Evaluation
Run single-image prediction via CLI:
```powershell
python src/predict.py --image "data/validation/Tomato___healthy/000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG"
```

Re-run test set evaluation across a folder:
```powershell
python src/predict.py --eval-dir "data/test"
```

---

## 3. Reliability & Out-of-Distribution (OOD) Research

An empirical reliability benchmark was conducted in [`results/reliability_audit/benchmark_report.md`](results/reliability_audit/benchmark_report.md) to evaluate model behavior under open-world inputs:

### Benchmark Findings Across 4 Evaluation Groups

| Evaluation Group | Sample Count | False Acceptance Rate | Mean Confidence | Key Finding |
|---|:---:|:---:|:---:|---|
| **Group A: Supported Tomato Leaves** (`data/validation/`) | 673 | 2.08% (FRR) | 96.10% | High accuracy (96.73%) and calibrated predictions on in-distribution leaves. |
| **Group B: Unsupported Tomato Diseases** (Quarantine) | 350 | **92.86% (FAR)** | 88.91% | Closed-world softmax forces unsupported diseases into the 3 trained classes with high confidence. |
| **Group C: Non-Tomato Crop Leaves** (PlantVillage) | 250 | **92.00% (FAR)** | 87.00% | Potato, pepper, apple, corn, and grape leaves are falsely classified as tomato diseases with $>90\%$ confidence. |
| **Group D: Non-Leaf Objects** | N/A | Excluded | N/A | Excluded from evaluation due to lack of a verified, labeled non-leaf dataset. |

### Experimental Filter Findings
1. **Excess Green Index (ExG $\ge 0.10$)**:
   - Exhibits **0.0% False Rejection Rate** on genuine tomato leaves.
   - **Scope**: Serves strictly as a non-vegetative filter (catching tools, soil, clothing). It **cannot** distinguish tomato leaves from other crops (mean ExG is 0.40–0.55 across all plant species).
2. **Laplacian Blur Screening ($\ge 50$)**:
   - Rejects ungrounded, severely blurred photos with **0.0% FRR** on in-focus tomato leaves.
3. **Pre-Softmax Free Energy Scoring ($E(\mathbf{x}) = -\log \sum e^{z_c}$)**:
   - In-distribution leaves produce lower energy (-11.90) than unsupported diseases (-8.27). However, overlapping distributions mean any threshold rejecting $>50\%$ of unsupported diseases also falsely rejects $14\%\text{--}35\%$ of genuine tomato leaves.

---

## 4. Repository Structure & Key Files

```text
CNN_AI/
├── app.py                                     # Streamlit interactive diagnostic web application
├── models/
│   ├── final_crop_disease_cnn_deploy.keras    # Verified inference-only model (<50 MB; Git tracked)
│   └── model_config.json                      # Formal model metadata and architecture specification
├── results/                                   # Empirical evaluation metrics, confusion matrices, and logs
│   ├── final_classification_report.txt        # Full held-out test classification metrics
│   ├── final_confusion_matrix.png             # Test set confusion matrix figure
│   ├── final_test_results.json                # Structured test metrics JSON
│   ├── training_history.csv                   # Per-epoch training/validation loss & accuracy log
│   ├── training_loss.png                      # Per-epoch loss curve
│   ├── training_accuracy.png                  # Per-epoch accuracy curve
│   └── reliability_audit/                     # Phase 8 empirical reliability audit reports
├── src/                                       # Core production source code
│   ├── model.py                               # CNN architecture definition
│   ├── train.py                               # Training runner with early stopping & callbacks
│   ├── predict.py                             # Single-image inference and evaluation CLI
│   └── prepare_dataset.py                     # Stratified dataset preparation & hash deduplication
├── tests/                                     # Automated test suite (39 tests)
│   ├── test_reliability.py                    # Input validation, preprocessing, and uncertainty tests
│   └── test_experimental_pipeline.py          # Leakage prevention and dataset integrity tests
├── docs/                                      # Extended documentation and audit reports
├── .gitignore                                 # Excludes datasets, venv, and large checkpoints
├── README.md                                  # Main repository documentation (this file)
└── requirements.txt                           # Pinned package dependencies
```
