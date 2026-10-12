# AgriAI: Project Completion Report

**Project Title**: AgriAI: Tomato Leaf Disease Classification Using Convolutional Neural Networks  
**Workspace**: `C:\Users\timnit\Desktop\CNN_AI`  
**Execution Phase**: Completed through Phase 8 (Final Model Selection & Test Evaluation)  

---

## 1. Project Objective

The AgriAI project develops a machine learning system to classify tomato leaf photographs into three categories:
1. `Tomato___Early_blight` (*Alternaria solani*)
2. `Tomato___Late_blight` (*Phytophthora infestans*)
3. `Tomato___healthy`

The project adheres to scientific data isolation rules: model development, architectural experimentation, and hyperparameter tuning were conducted exclusively using training and validation splits. The held-out test split was reserved strictly for final evaluation after model selection was locked.

---

## 2. Dataset Preparation and Verification

- **Source**: Public PlantVillage tomato RGB subset.
- **Verification**: 4,500 total files inspected locally with zero corrupted files. 14 exact duplicate files within classes were removed, yielding **4,486 unique valid images**.
- **Class Breakdown**:
  - `Tomato___Early_blight`: 1,000 images
  - `Tomato___Late_blight`: 1,901 images
  - `Tomato___healthy`: 1,585 images
- **Stratified Partitioning (Random Seed 42)**:
  - **Train (70%)**: 3,140 images
  - **Validation (15%)**: 673 images
  - **Test (15%)**: 673 images
- **Data Integrity**: Verified 0 SHA-256 hash overlaps across train, validation, and test partitions.

---

## 3. Baseline Model & Training Results

- **Architecture**: 3-block Convolutional Neural Network:
  - Input: (224, 224, 3)
  - Preprocessing & Augmentation: `Rescaling(1/255)`, `RandomFlip('horizontal')`, `RandomRotation(0.05)`, `RandomZoom(0.10)`
  - Blocks: `Conv2D(32, 3x3, ReLU)` → `MaxPool`, `Conv2D(64, 3x3, ReLU)` → `MaxPool`, `Conv2D(128, 3x3, ReLU)` → `MaxPool`
  - Head: `Flatten` → `Dense(128, ReLU)` → `Dropout(0.5)` → `Dense(3, Softmax)`
  - Parameters: 12,938,819 (100% trainable)
- **Training Setup**: Adam optimizer (learning rate = 0.001), sparse categorical cross-entropy loss, batch size 32, up to 30 epochs.
- **Baseline Validation Performance**:
  - Best Checkpoint: Epoch 14 (restored via EarlyStopping after epoch 19)
  - **Best Validation Accuracy**: **96.58%** (`0.965825`)
  - **Best Validation Loss**: **0.086766**

---

## 4. Phase 7 Hyperparameter & Architecture Experiments

Five planned experimental models were trained and benchmarked strictly on the validation set:

| Experiment Name | Architecture | LR | Batch | Dropout | Params | Epochs | Best Val Accuracy | Best Val Loss | Early Stopping |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Baseline (Phase 5)** | Conv 32/64/128 + Flatten + Dense 128 | 0.001 | 32 | 0.5 | 12,938,819 | 19 / 30 | **96.58%** (`0.965825`) | **0.086766** | Yes (Epoch 14) |
| `p7_proposal_lr1e-3_bs32_drop30` | 4-block Conv-BN + GAP + Dense 128 | 0.001 | 32 | 0.3 | 423,619 | 20 / 20 | 94.95% (`0.949480`) | 0.181050 | Yes (Epoch 15) |
| `p7_proposal_lr3e-4_bs32_drop50` | 4-block Conv-BN + GAP + Dense 128 | 0.0003 | 32 | 0.5 | 423,619 | 12 / 20 | 94.21% (`0.942051`) | 0.138650 | Yes (Epoch 7) |
| `p7_proposal_lr1e-3_bs32_drop50` | 4-block Conv-BN + GAP + Dense 128 | 0.001 | 32 | 0.5 | 423,619 | 11 / 20 | 91.08% (`0.910847`) | 0.266655 | Yes (Epoch 10) |
| `p7_compact_lr1e-3_bs32_drop50` | 3-block Conv-BN + GAP + Dense 128 | 0.001 | 32 | 0.5 | 111,043 | 12 / 20 | 87.22% (`0.872214`) | 0.350240 | Yes (Epoch 7) |
| `p7_proposal_lr1e-3_bs64_drop50` | 4-block Conv-BN + GAP + Dense 128 | 0.001 | 64 | 0.5 | 423,619 | 8 / 20 | 70.88% (`0.708767`) | 0.996171 | Yes (Epoch 8) |

All 5 experiment checkpoints, full JSON epoch histories, and summary CSV records were verified and persisted.

---

## 5. Final Model Selection

Model selection was governed entirely by validation evidence:
- Primary Metric: Highest validation accuracy.
- Tie-Breaker: Lowest validation loss.

**Selection Finding**:
> The baseline CNN achieved the strongest validation performance at 96.58% and was therefore selected as the final model. None of the five Phase 7 experiments exceeded the baseline validation accuracy.

The verified baseline checkpoint `models/crop_disease_cnn.keras` was copied to `models/final_crop_disease_cnn.keras` (SHA-256: `FE22B7CEE1D0A14B2818941A2BA881EF169907D0B0D94AAE390DCA4624A9409C`).

---

## 6. Final Test Evaluation

The selected final model was evaluated once on the untouched 673-image test split:

| Metric | Result |
|---|---:|
| **Test Accuracy** | **94.80%** (`0.947994`) |
| **Test Loss** | **0.122676** |
| **Weighted Precision** | **94.83%** (`0.948266`) |
| **Weighted Recall** | **94.80%** (`0.947994`) |
| **Weighted F1-Score** | **94.81%** (`0.948099`) |
| **Macro F1-Score** | **94.17%** (`0.941720`) |

### Per-Class Test Performance:
- **Tomato___Early_blight**: Precision 88.24%, Recall 90.00%, F1 89.11% (150 test images)
- **Tomato___Late_blight**: Precision 94.33%, Recall 93.33%, F1 93.83% (285 test images)
- **Tomato___healthy**: Precision 99.58%, Recall 99.58%, F1 99.58% (238 test images)

---

## 7. Prediction Capability & Smoke Testing

The inference pipeline `src/predict.py` was tested on sample validation images from all three classes:
- **Healthy Leaf**: Predicted `Tomato___healthy` with **99.98% confidence**.
- **Early Blight Leaf**: Predicted `Tomato___Early_blight` with **76.18% confidence**.
- **Late Blight Leaf**: Predicted `Tomato___Late_blight` with **100.00% confidence**.

All predictions verified:
- Correct RGB loading and resizing to (224, 224, 3)
- No double scaling (raw pixels fed to model containing `Rescaling(1/255)`)
- Output probabilities sum to 100.00% across all 3 classes.

---

## 8. Limitations

1. **Controlled Photography Bias**: Benchmark images feature single leaves staged on uniform backgrounds; real-world field lighting, complex canopies, and weed backgrounds were not in the training distribution.
2. **Three-Class Closed World**: Any other tomato disease or abiotic stress will be forced into one of the three trained classes.
3. **CPU Execution Speed**: Inference takes ~1.5 seconds per image on native Windows CPU.

---

## 9. Future Work

1. **Field Dataset Expansion**: Incorporate in-the-wild field datasets (e.g., PlantDoc) to improve robustness to outdoor farm environments.
2. **Multi-Class Expansion**: Add classes for Septoria, Bacterial spot, and Tomato Yellow Leaf Curl.
3. **Quantization & Mobile Porting**: Export to TFLite (INT8 quantized) for offline smartphone deployment.

---

## 10. Final Project Status

**Phase 8 Status**: **COMPLETE**.  
All project deliverables, model checkpoints, validation records, test evaluations, and documentation files are fully verified and synchronized. No further phases or tuning will be executed without explicit instruction.
