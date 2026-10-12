# Phase 7 — Model Improvement and Hyperparameter Tuning

## 1. Objective and Motivation

The primary goal of Phase 7 is to evaluate whether architectural enhancements (such as deeper convolutional hierarchies, Batch Normalization, and Global Average Pooling) and hyperparameter variations (learning rate, dropout rate, batch size, and network depth) can improve generalization performance and reduce overfitting relative to the baseline CNN model established in Phase 5.

### Critical Data Policy
- **Validation-Only Tuning**: All Phase 7 tuning and comparison strictly utilize only the training split (`data/train`, 3,140 images) and validation split (`data/validation`, 673 images).
- **Test Set Isolation**: The held-out test split (`data/test`, 673 images) was **never loaded, scored, or used** during Phase 7 to prevent data leakage.
- **Baseline Preservation**: The baseline checkpoint (`models/crop_disease_cnn.keras`) and baseline history records (`results/training_history.json`) were preserved untouched.

## 2. Experimental Design and Modifications

### Architectural Candidates
1. **Proposal Architecture (4 Convolutional Blocks)**:
   - **Preprocessing & Augmentation**: Input (224 × 224 × 3), `RandomFlip('horizontal')`, `RandomRotation(0.1)`, `RandomZoom(0.1)`, `RandomContrast(0.1)`, `Rescaling(1/255)`.
   - **Feature Extractor**: 4 sequential blocks of `Conv2D(3x3, padding='same', activation='relu')` → `BatchNormalization()` → `MaxPooling2D((2,2))` with filter counts: 32 → 64 → 128 → 256.
   - **Classification Head**: `GlobalAveragePooling2D()` → `Dense(128, activation='relu')` → `Dropout(rate)` → `Dense(3, activation='softmax')`.
   - **Parameter Count**: 423,619 parameters.
2. **Compact Architecture (3 Convolutional Blocks)**:
   - Identical preprocessing and head structure, but with 3 convolutional blocks (filters: 32 → 64 → 128).
   - Designed to isolate whether the 4th convolutional block (256 filters) provides net validation benefit.
   - **Parameter Count**: 111,043 parameters.
3. **Baseline Reference (Phase 5 Architecture)**:
   - 3 Conv2D blocks (32 → 64 → 128) without Batch Normalization, followed by `Flatten()` → `Dense(128)` → `Dropout(0.5)` → `Dense(3, softmax)`.
   - **Parameter Count**: 12,938,819 parameters (due to large flattened dense transition).

### Controlled Hyperparameter Variations
Five predetermined experiments systematically investigated specific hyperparameter and structural axes:
- **Experiment 1 (`p7_proposal_lr1e-3_bs32_drop50`)**: Proposal baseline configuration (lr = 0.001, batch size = 32, dropout = 0.5).
- **Experiment 2 (`p7_proposal_lr3e-4_bs32_drop50`)**: Reduced learning rate (lr = 0.0003, batch size = 32, dropout = 0.5) to test finer gradient convergence.
- **Experiment 3 (`p7_proposal_lr1e-3_bs32_drop30`)**: Reduced dropout regularization (dropout = 0.3, lr = 0.001, batch size = 32) to evaluate regularization penalty.
- **Experiment 4 (`p7_proposal_lr1e-3_bs64_drop50`)**: Increased batch size (batch size = 64, lr = 0.001, dropout = 0.5) to observe batch normalization stability and gradient smoothing.
- **Experiment 5 (`p7_compact_lr1e-3_bs32_drop50`)**: Compact 3-block architecture (lr = 0.001, batch size = 32, dropout = 0.5) to assess depth necessity.

### Common Training Configuration
- **Optimizer**: Adam with sparse categorical cross-entropy loss.
- **Epochs**: Maximum 20 epochs.
- **Callbacks**: `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)`, `ModelCheckpoint(monitor='val_accuracy', mode='max', save_best_only=True)`.
- **Random Seed**: Fixed at 42 across all runs.

## 3. Validation Results and Model Comparison

All metrics below are strictly derived from actual completed Keras validation histories. No test data was evaluated.

| Model / Experiment | Architecture | Optimizer | Learning Rate | Batch Size | Dropout | Parameters | Epochs Trained | Best Val Accuracy | Best Val Loss | Early Stopping | Training Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **Baseline (Phase 5)** | Conv 32/64/128 + Flatten + Dense128 | Adam | 0.001 | 32 | 0.5 | 12,938,819 | 19 | **96.58%** (0.965825) | **0.086766** | Yes | — |
| `p7_proposal_lr1e-3_bs32_drop50` | Proposal: Conv-BN-MaxPool blocks 32/64/128/256, GAP, Dense128 | Adam | 0.001 | 32 | 0.5 | 423,619 | 11 | 91.08% (0.910847) | 0.266655 | Yes | 2260.9 s |
| `p7_proposal_lr3e-4_bs32_drop50` | Proposal: Conv-BN-MaxPool blocks 32/64/128/256, GAP, Dense128 | Adam | 0.0003 | 32 | 0.5 | 423,619 | 12 | 94.21% (0.942051) | 0.138650 | Yes | 2316.0 s |
| `p7_proposal_lr1e-3_bs32_drop30` | Proposal: Conv-BN-MaxPool blocks 32/64/128/256, GAP, Dense128 | Adam | 0.001 | 32 | 0.3 | 423,619 | 20 | 94.95% (0.949480) | 0.181050 | Yes | 6816.5 s |
| `p7_proposal_lr1e-3_bs64_drop50` | Proposal: Conv-BN-MaxPool blocks 32/64/128/256, GAP, Dense128 | Adam | 0.001 | 64 | 0.5 | 423,619 | 8 | 70.88% (0.708767) | 0.996171 | Yes | 1653.0 s |
| `p7_compact_lr1e-3_bs32_drop50` | Compact: Conv-BN-MaxPool blocks 32/64/128, GAP, Dense128 | Adam | 0.001 | 32 | 0.5 | 111,043 | 12 | 87.22% (0.872214) | 0.350240 | Yes | 1762.1 s |

## 4. Key Findings and Comparison Analysis

### Candidate Ranking (Validation Performance Only)
1. **Best Phase 7 Accuracy Candidate**: `p7_proposal_lr1e-3_bs32_drop30` achieved **94.95%** validation accuracy (best val loss: 0.181050) over 20 epochs.
2. **Lowest Phase 7 Loss Candidate**: `p7_proposal_lr3e-4_bs32_drop50` achieved the lowest validation loss of **0.138650** (best val accuracy: 94.21%) over 12 epochs.
3. **Overall Best Candidate**: `p7_proposal_lr1e-3_bs32_drop30` (highest validation accuracy; validation loss as tie-breaker).

### Baseline Comparison
- **Baseline Validation Accuracy**: **96.58%** (0.965825)
- **Baseline Best Validation Loss**: **0.086766**
- **Top Candidate Validation Accuracy**: **94.95%** (0.949480)
- **Top Candidate Best Validation Loss**: **0.181050**

> **Outcome: Baseline Remains Superior**. None of the Phase 7 experimental models surpassed the baseline validation accuracy (96.58%) or baseline validation loss (0.086766). The Phase 5 baseline remains the top-performing model across all validation metrics.

### Architectural and Hyperparameter Insights
- **Global Average Pooling vs. Flattening**: While GAP drastically reduced parameter count (from ~12.9M to ~423K), the spatial feature compression resulted in lower peak validation accuracy on this 3-class classification task.
- **Dropout Regularization**: Reducing dropout from 0.5 to 0.3 in the proposal architecture improved validation accuracy from 91.08% to 94.95%, indicating that 0.5 dropout combined with GAP was over-regularizing the dense classification head.
- **Learning Rate Tuning**: Lowering learning rate to 0.0003 produced the lowest validation loss among candidates (0.138650) and strong validation accuracy (94.21%), demonstrating more stable convergence.
- **Batch Size Sensitivity**: Increasing batch size to 64 caused severe performance degradation (70.88% validation accuracy, validation loss 0.996171), primarily due to reduced update frequency over the 3,140-sample training dataset.
- **Network Depth (4 blocks vs. 3 blocks)**: The 4-block proposal (94.95% top accuracy) substantially outperformed the 3-block compact architecture (87.22% top accuracy), validating the benefit of hierarchical feature depth when using GAP.

## 5. Limitations of the Experiments

- **Discrete Search Grid**: The exploration was restricted to 5 manual configurations due to native CPU computation constraints; continuous Bayesian optimization or random search over wider parameter ranges was not feasible.
- **Epoch Ceiling**: Maximum epochs were capped at 20; some models (such as `p7_proposal_lr1e-3_bs32_drop30`) trained through all 20 epochs and might have benefited from additional training steps.
- **Single Random Seed**: All experiments were run with random seed 42. While ensuring deterministic comparisons, seed variance across multiple initializations was not quantified.
- **Strict Validation Boundary**: Model decisions are based exclusively on validation metrics; generalizability across unseen field distributions cannot be inferred solely from validation performance.

## 6. Artifact Verification and Reproducibility

- **Summary Table**: `results/hyperparameter_experiments.csv` (contains all 5 completed experiment records with full parameter and outcome fields).
- **Per-Epoch Histories**: `results/phase7_histories/` (contains JSON files for each completed run including epoch-by-epoch loss, accuracy, and callback outcomes).
- **Model Checkpoints**: `models/experiments/` (contains `.keras` checkpoints saved at the best validation accuracy epoch for each experiment).
- **Runner Script**: `src/tune.py` (idempotent, validation-only runner that checks existing runs and safely preserves artifacts).

---
*Note: Test split data was strictly isolated throughout Phase 7. Final test evaluation and selection occur exclusively in Phase 8.*
