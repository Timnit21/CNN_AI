# Phase 8 — Final Model Selection and Verification

## 1. Objective

The objective of Phase 8 is to perform rigorous model selection and final held-out test evaluation for **AgriAI: Tomato Leaf Disease Classification Using Convolutional Neural Networks**. The project classifies tomato leaf images into three distinct agronomic categories: *Early blight* (*Alternaria solani*), *Late blight* (*Phytophthora infestans*), and *healthy*.

Model selection is strictly conducted using validation evidence accumulated through Phase 5 (baseline training) and Phase 7 (systematic architecture and hyperparameter tuning). Following model selection, the chosen final model is evaluated exactly once on the held-out test split.

---

## 2. Dataset Summary

The dataset consists of the three RGB tomato leaf classes from the public **PlantVillage** benchmark repository:
- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___healthy`

A total of 4,500 raw image files were verified locally. After removing 14 exact same-class duplicate files, 4,486 unique, valid images remained:
- **Tomato___Early_blight**: 1,000 images
- **Tomato___Late_blight**: 1,901 images
- **Tomato___healthy**: 1,585 images

---

## 3. Train / Validation / Test Split

The dataset was partitioned using a deterministic, stratified split with random seed **42**:
- **Training Set (70%)**: 3,140 images (700 Early blight, 1,331 Late blight, 1,109 healthy)
- **Validation Set (15%)**: 673 images (150 Early blight, 285 Late blight, 238 healthy)
- **Test Set (15%)**: 673 images (150 Early blight, 285 Late blight, 238 healthy)

There is zero SHA-256 hash overlap across train, validation, and test subsets. The test set was strictly isolated and never accessed during baseline training or Phase 7 tuning.

---

## 4. Baseline Architecture

The baseline model is a sequential Convolutional Neural Network designed for 224 × 224 RGB inputs:
- **Data Preprocessing & Augmentation** (active during training only):
  - `Rescaling(1.0 / 255.0)`
  - `RandomFlip('horizontal')`
  - `RandomRotation(0.05)`
  - `RandomZoom(0.10)`
- **Feature Extraction**:
  - `Conv2D(32, (3, 3), padding='same', activation='relu')` → `MaxPooling2D((2, 2))`
  - `Conv2D(64, (3, 3), padding='same', activation='relu')` → `MaxPooling2D((2, 2))`
  - `Conv2D(128, (3, 3), padding='same', activation='relu')` → `MaxPooling2D((2, 2))`
- **Classification Head**:
  - `Flatten()`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(3, activation='softmax')`
- **Total Parameters**: 12,938,819 (12,938,819 trainable, 0 non-trainable).
- **Optimization**: Adam optimizer (learning rate = 0.001) with sparse categorical cross-entropy loss.

During Phase 5 training (batch size 32, max 30 epochs), EarlyStopping restored the best weights at **epoch 14**, achieving:
- **Best Validation Accuracy**: **96.58%** (`0.965825`)
- **Best Validation Loss**: **0.086766**

---

## 5. Phase 7 Experiment Comparison

Phase 7 evaluated 5 planned candidate models using only training and validation data:

| Model / Experiment | Architecture Description | LR | Batch | Dropout | Params | Epochs | Best Val Accuracy | Best Val Loss | Early Stopping |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Baseline (Phase 5)** | Conv 32/64/128 + Flatten + Dense 128 | 0.001 | 32 | 0.5 | 12,938,819 | 19 / 30 | **96.58%** (`0.965825`) | **0.086766** | Yes (Epoch 14) |
| **Exp 1** (`p7_proposal_lr1e-3_bs32_drop50`) | Proposal: Conv-BN 32/64/128/256 + GAP + Dense 128 | 0.001 | 32 | 0.5 | 423,619 | 11 / 20 | **91.08%** (`0.910847`) | **0.266655** | Yes (Epoch 10) |
| **Exp 2** (`p7_proposal_lr3e-4_bs32_drop50`) | Proposal: Conv-BN 32/64/128/256 + GAP + Dense 128 | 0.0003 | 32 | 0.5 | 423,619 | 12 / 20 | **94.21%** (`0.942051`) | **0.138650** | Yes (Epoch 7) |
| **Exp 3** (`p7_proposal_lr1e-3_bs32_drop30`) | Proposal: Conv-BN 32/64/128/256 + GAP + Dense 128 | 0.001 | 32 | 0.3 | 423,619 | 20 / 20 | **94.95%** (`0.949480`) | **0.181050** | Yes (Epoch 15) |
| **Exp 4** (`p7_proposal_lr1e-3_bs64_drop50`) | Proposal: Conv-BN 32/64/128/256 + GAP + Dense 128 | 0.001 | 64 | 0.5 | 423,619 | 8 / 20 | **70.88%** (`0.708767`) | **0.996171** | Yes (Epoch 8) |
| **Exp 5** (`p7_compact_lr1e-3_bs32_drop50`) | Compact: Conv-BN 32/64/128 + GAP + Dense 128 | 0.001 | 32 | 0.5 | 111,043 | 12 / 20 | **87.22%** (`0.872214`) | **0.350240** | Yes (Epoch 7) |

---

## 6. Why the Baseline Was Selected

Model selection is strictly governed by validation metrics:
1. Primary criterion: Highest best validation accuracy.
2. Tie-breaker: Lowest best validation loss.

**Selection Statement**:
> The baseline CNN achieved the strongest validation performance at 96.58% and was therefore selected as the final model. None of the five Phase 7 experiments exceeded the baseline validation accuracy.

The best-performing Phase 7 candidate (`p7_proposal_lr1e-3_bs32_drop30`) reached 94.95% validation accuracy, which remains **1.63 percentage points below the baseline**. Furthermore, the baseline achieved a validation loss of 0.086766, significantly lower than the best candidate validation loss (0.138650). 

Consequently, the verified baseline checkpoint `models/crop_disease_cnn.keras` was copied directly to `models/final_crop_disease_cnn.keras` (SHA-256: `FE22B7CEE1D0A14B2818941A2BA881EF169907D0B0D94AAE390DCA4624A9409C`).

---

## 7. Final Test Methodology

The selected final model was evaluated on the held-out test split under standard conditions:
- **Test Dataset**: `data/test/` containing 673 images.
- **Image Resolution**: 224 × 224 RGB.
- **Batch Size**: 32, deterministic order (no shuffling).
- **Inference Mode**: Augmentation layers deactivated (`training=False`).
- **Pass Count**: Exactly 1 full pass over all 673 test images.

---

## 8. Final Test Metrics

The final evaluation yielded the following metrics on the held-out test set:

| Metric | Measured Value |
|---|---:|
| **Test Loss** | **0.122676** |
| **Test Accuracy** | **94.80%** (`0.947994`) |
| **Weighted Precision** | **94.83%** (`0.948266`) |
| **Weighted Recall** | **94.80%** (`0.947994`) |
| **Weighted F1-Score** | **94.81%** (`0.948099`) |
| **Macro Precision** | **94.05%** (`0.940471`) |
| **Macro Recall** | **94.30%** (`0.943044`) |
| **Macro F1-Score** | **94.17%** (`0.941720`) |

The difference between best validation accuracy (96.58%) and test accuracy (94.80%) is **1.78 percentage points**, reflecting a normal generalization gap on unseen test data.

---

## 9. Confusion Matrix Interpretation

Below is the confusion matrix for the 673 test samples (rows represent actual ground truth, columns represent model predictions):

| Actual \\ Predicted | Tomato___Early_blight | Tomato___Late_blight | Tomato___healthy | Support |
|---|---:|---:|---:|---:|
| **Tomato___Early_blight** | **135** | 15 | 0 | 150 |
| **Tomato___Late_blight** | 18 | **266** | 1 | 285 |
| **Tomato___healthy** | 0 | 1 | **237** | 238 |
| **Total Predicted** | **153** | **282** | **238** | **673** |

### Key Diagnostic Observations:
1. **Healthy Leaf Discrimination**: Outstanding performance. 237 of 238 healthy leaves were correctly classified (99.58% precision and recall). Healthy leaves rarely confuse the model with diseased leaves.
2. **Early vs. Late Blight Overlap**: The primary source of error is discrimination between Early blight and Late blight (15 Early blight leaves misclassified as Late blight; 18 Late blight leaves misclassified as Early blight). This aligns with pathological reality, where early necrotic lesions of both blights exhibit visual similarities in leaf texture and coloration.

---

## 10. Per-Class Performance

| Class Name | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Tomato___Early_blight` | 0.8824 (88.24%) | 0.9000 (90.00%) | 0.8911 (89.11%) | 150 |
| `Tomato___Late_blight` | 0.9433 (94.33%) | 0.9333 (93.33%) | 0.9383 (93.83%) | 285 |
| `Tomato___healthy` | 0.9958 (99.58%) | 0.9958 (99.58%) | 0.9958 (99.58%) | 238 |

---

## 11. Limitations

While the model demonstrates high classification accuracy on the benchmark test split, several technical and operational limitations must be noted:
1. **Controlled Laboratory Conditions**: The PlantVillage dataset features leaves photographed on uniform backgrounds under controlled lighting. The model has not been evaluated against messy farm backgrounds, complex field illumination, shadows, or dew.
2. **Co-occurring Pathologies**: The current model assumes single-label classification across three categories. It cannot identify multiple simultaneous diseases, nutrient deficiencies, or pest damage.
3. **Severity Assessment**: The classifier provides discrete category predictions and class probabilities, not quantitative disease severity (e.g., percentage leaf area affected).
4. **Hardware and Environment**: Native Windows execution runs on CPU; inference latency is approximately 1.5–2.0 seconds per batch on CPU.

---

## 12. Final Conclusion

The model selection process was executed transparently using validation evidence exclusively. The baseline 3-block CNN outperformed all tested deeper and regularized architectural variations, demonstrating that spatial feature preservation via flattening provided superior discriminative power on this 3-class dataset. 

The finalized model achieved **94.80% accuracy** and **94.81% weighted F1-score** on the untouched 673-image test split, establishing a robust, reproducible benchmark for tomato leaf disease identification.
