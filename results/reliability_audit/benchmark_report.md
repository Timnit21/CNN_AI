# AgriAI Prediction Reliability Empirical Benchmark Report

- **Date:** 2026-10-12T08:32:38Z
- **Evaluated Checkpoint:** `models\final_crop_disease_cnn.keras`
- **Execution Mode:** Read-Only Inference (0 training, 0 file alterations)
- **Provisional Thresholds:** Min Score $\ge 0.6$, Min Margin $\ge 0.15$

---

## 1. Executive Summary & Core Reliability Findings

> [!WARNING]
> **Key Finding: Softmax Overconfidence on Out-of-Distribution Inputs is Severe.**
> The empirical benchmark demonstrates that the provisional thresholds (0.60 score, 0.15 margin) **fail completely** at detecting unsupported tomato diseases and non-tomato leaves:
> - **False Acceptance Rate on Unsupported Tomato Diseases (Group B):** **92.86%** (325 / 350) were falsely accepted as conclusive.
> - **False Acceptance Rate on Non-Tomato Crop Leaves (Group C):** **92.0%** (230 / 250) were falsely accepted as conclusive.
> - **Extreme Overconfidence ($\ge 99\%$ score):** **29.43%** of unsupported tomato diseases and **31.2%** of non-tomato leaves produced $\ge 99\%$ top-1 softmax scores!

---

## 2. Benchmark Metrics & Group Comparison

| Metric / Attribute | Group A: Supported Tomato (Validation) | Group B: Unsupported Tomato Diseases | Group C: Non-Tomato Crop Leaves | Group D: Non-Leaf Objects |
|---|:---:|:---:|:---:|:---:|
| **Sample Count** | **673** | **350** | **250** | *Excluded (None available)* |
| **Accepted (Conclusive)** | 659 (97.92%) | **325 (92.86%) [FAR]** | **230 (92.0%) [FAR]** | N/A |
| **Rejected (Inconclusive)** | 14 (2.08%) [FRR] | 25 (7.14%) | 20 (8.0%) | N/A |
| **Mean Top Score** | **0.9610** | **0.8891** | **0.8700** | N/A |
| **Median Top Score (P50)** | **0.9995** | **0.9559** | **0.9430** | N/A |
| **Scores $\ge 90\%$** | 86.92% | **64.86%** | **55.6%** | N/A |
| **Scores $\ge 99\%$** | 74.74% | **29.43%** | **31.2%** | N/A |
| **Mean Margin** | 0.9229 | **0.7818** | **0.7467** | N/A |

### Metric Definitions and Denominators
1. **False Acceptance Rate (FAR) for Unsupported Inputs (Groups B & C):**
   $$\text{FAR} = \frac{\text{Unsupported Samples Accepted as Conclusive}}{\text{Total Unsupported Samples in Group}} \times 100\%$$
   - *Denominator:* Total unsupported images tested (350 for Group B; 250 for Group C).
   - *Interpretation:* The catastrophic failure rate where an unsupported leaf is confidently diagnosed with treatment advice.
2. **False Rejection Rate (FRR) for Supported Inputs (Group A):**
   $$\text{FRR} = \frac{\text{Supported Tomato Samples Flagged Inconclusive}}{\text{Total Supported Tomato Samples in Group A}} \times 100\%$$
   - *Denominator:* Total supported validation images tested (673).
   - *Value:* **2.08%**.
3. **Rejection Rate:**
   $$\text{Rejection Rate} = 100\% - \text{Acceptance Rate}$$

---

## 3. Group A: Supported Tomato In-Distribution Performance

- **Evaluated Dataset:** `data/validation/` (673 images, 0 test collisions).
- **Raw Multi-Class Accuracy:** **96.73%** (651 / 673).
- **Conclusive Subset Accuracy:** **97.42%** (evaluated on the 659 accepted images).

### Confusion Matrix (Rows = Actual, Columns = Predicted)
```
                          Predicted
Actual            Early Blight    Late Blight    Healthy
Early Blight           137            13             0             
Late Blight            8              276            1             
Healthy                0              0              238           
```

---

## 4. Forced Output Distributions on Unsupported Inputs

When presented with unsupported classes, the closed-world softmax layer forced them into the 3 trained classes as follows:

### Group B (Unsupported Tomato Diseases — 350 samples across 7 diseases):
- **Tomato___Early_blight:** 226 (64.6%)
- **Tomato___Late_blight:**  103 (29.4%)
- **Tomato___healthy:**      21 (6.0%)

### Group C (Non-Tomato Crop Leaves — 250 samples: Potato, Pepper, Apple, Corn, Grape):
- **Tomato___Early_blight:** 95 (38.0%)
- **Tomato___Late_blight:**  132 (52.8%)
- **Tomato___healthy:**      23 (9.2%)

---

## 5. Group D Status: Non-Leaf Objects
- **Status:** **EXCLUDED & DOCUMENTED.**
- **Reason:** No legitimate, appropriately labeled non-leaf dataset exists in the CNN_AI project. To strictly adhere to anti-fabrication standards, Group D was not populated with arbitrary or unverified files.
