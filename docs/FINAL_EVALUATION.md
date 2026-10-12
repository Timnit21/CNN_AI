# Phase 8 — Final Evaluation and Benchmark Results

## 1. Executive Summary

Phase 8 completes the model selection and final test evaluation for **AgriAI: Tomato Leaf Disease Classification Using Convolutional Neural Networks**. Model selection was determined strictly based on validation results across the baseline and all 5 Phase 7 hyperparameter experiments. 

Following model selection, the chosen final model ([models/final_crop_disease_cnn.keras](file:///C:/Users/timnit/Desktop/CNN_AI/models/final_crop_disease_cnn.keras)) was evaluated once on the strictly held-out test split of 673 images.

---

## 2. Test Set Isolation Policy

- **Zero Test Contamination**: The test dataset (`data/test`, 673 images) was strictly isolated throughout Phase 1 through Phase 7. It was never loaded during training, validation, hyperparameter tuning, or model selection.
- **Single Final Evaluation**: Test evaluation was conducted only after model selection was completed and locked.
- **Stratified Partitioning**: The 673 test images preserve the exact class distribution (150 Early blight, 285 Late blight, 238 healthy) with zero hash overlap with training or validation splits.

---

## 3. Final Model Selection Summary

Model selection compared the Phase 5 baseline against all five completed Phase 7 experiments using validation accuracy (primary) and validation loss (secondary):

| Model / Experiment | Best Validation Accuracy | Best Validation Loss | Selection Outcome |
|---|---:|---:|---|
| **Baseline (Phase 5)** | **96.58%** (`0.965825`) | **0.086766** | **Selected as Final Model** |
| `p7_proposal_lr1e-3_bs32_drop30` | 94.95% (`0.949480`) | 0.181050 | Not selected (Validation acc < Baseline) |
| `p7_proposal_lr3e-4_bs32_drop50` | 94.21% (`0.942051`) | 0.138650 | Not selected (Validation acc < Baseline) |
| `p7_proposal_lr1e-3_bs32_drop50` | 91.08% (`0.910847`) | 0.266655 | Not selected (Validation acc < Baseline) |
| `p7_compact_lr1e-3_bs32_drop50` | 87.22% (`0.872214`) | 0.350240 | Not selected (Validation acc < Baseline) |
| `p7_proposal_lr1e-3_bs64_drop50` | 70.88% (`0.708767`) | 0.996171 | Not selected (Validation acc < Baseline) |

**Conclusion**: The baseline model achieved the highest validation accuracy (96.58%) and lowest validation loss (0.086766) among all candidates. It was staged to `models/final_crop_disease_cnn.keras`.

---

## 4. Final Test Performance

Evaluation was executed with `src/evaluate_final.py` on 673 test images:

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

---

## 5. Validation vs. Test Accuracy Analysis

- **Best Validation Accuracy**: **96.58%** (`0.965825`)
- **Final Test Accuracy**: **94.80%** (`0.947994`)
- **Generalization Difference**: **-1.78 percentage points**

### Why There is a Difference:
1. **Checkpoint Selection Bias**: The checkpoint was saved at the epoch with the highest validation accuracy (epoch 14 of training). Validation metrics during training reflect peak performance across epochs on the validation sample.
2. **Unseen Data Distribution**: Although both validation and test sets were drawn from the same stratified pool, subtle variations in leaf orientation, necrosis severity, and lighting in the test set introduce slight distribution variance.
3. **Generalization Gap**: A drop of only 1.78 percentage points indicates solid generalization with minimal overfitting.

---

## 6. Per-Class Test Performance

| Class Name | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Tomato___Early_blight` | 0.8824 (88.24%) | 0.9000 (90.00%) | 0.8911 (89.11%) | 150 |
| `Tomato___Late_blight` | 0.9433 (94.33%) | 0.9333 (93.33%) | 0.9383 (93.83%) | 285 |
| `Tomato___healthy` | 0.9958 (99.58%) | 0.9958 (99.58%) | 0.9958 (99.58%) | 238 |

---

## 7. Confusion Matrix

The confusion matrix on the 673 held-out test images is shown below:

```text
Actual \ Predicted       Early Blight    Late Blight      Healthy      Total
Tomato___Early_blight        135             15              0          150
Tomato___Late_blight          18            266              1          285
Tomato___healthy               0              1            237          238
Total Predicted              153            282            238          673
```

- **Healthy class**: Precision 99.58%, Recall 99.58%. Only 1 healthy leaf was misclassified as late blight.
- **Blight confusion**: 15 Early blight images were predicted as Late blight; 18 Late blight images were predicted as Early blight.

---

## 8. Artifact Locations

- Checkpoint: `models/final_crop_disease_cnn.keras`
- Test metrics: `results/final_test_results.json`
- Classification report: `results/final_classification_report.txt`
- Confusion matrix plot: `results/final_confusion_matrix.png`
- Comparison table: `results/baseline_vs_final.csv`
