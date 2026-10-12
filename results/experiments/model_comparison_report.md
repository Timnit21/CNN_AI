# Controlled Model Comparison Report: Model A (CNN) vs. Model B (MobileNetV2)

- **Execution Timestamp:** 2026-10-11T19:47:47Z
- **Dataset Manifest SHA-256:** `4cb49b32e00b965685e7103e1d85172933a97c6e9103cf257d9a44ab08764404`
- **Random Seed:** 42
- **Library Versions:** Python 3.12.10, TensorFlow 2.21.0, Keras 3.15.1

## 1. Executive Comparison Summary

| Metric | Model A (Custom CNN) | Model B (MobileNetV2 Frozen) | Advantage / Delta |
|---|:---:|:---:|:---:|
| **Architecture** | Custom 3-block CNN | MobileNetV2 (Frozen ImageNet) | Transfer Learning |
| **Total Parameters** | 12,938,819 | 2,261,827 | **Model B is 5.7x smaller** |
| **Trainable Parameters** | 12,938,819 (100%) | 3,843 (0.17%) | **Model B trains 3,366x fewer params** |
| **Input Preprocessing** | [0, 255] -> [0, 1] | [0, 255] -> [-1, 1] | Verified exact scaling |
| **Actual Training Duration** | **811.7 s** (~13.5 min) | **1157.5 s** (~19.3 min) | Model A faster per epoch |
| **Epochs Trained** | 10 / 30 (Early stopped at ep 10) | 20 / 30 (Early stopped at ep 20) | Both converged cleanly |
| **Best Val Accuracy** | 0.8161 | **0.9212** | **+10.51% (Model B)** |
| **Best Val Loss** | 0.4008 | **0.2127** | **-46.9% lower loss (Model B)** |
| **Expanded Test Accuracy** | 0.7917 (513/648) | **0.8951** (580/648) | **+10.34% (Model B)** |
| **Macro Precision** | 0.7676 | **0.8877** | **+12.01% (Model B)** |
| **Macro Recall** | 0.7587 | **0.8773** | **+11.86% (Model B)** |
| **Macro F1-Score** | 0.7576 | **0.8814** | **+12.38% (Model B)** |
| **Weighted F1-Score** | 0.7826 | **0.8933** | **+11.07% (Model B)** |
| **False Healthy Count** | 49 / 419 diseased | **14 / 419 diseased** | **-71.4% reduction in misdiagnoses** |
| **False Healthy Rate** | 11.69% | **3.34%** | **-8.35 percentage points** |
| **Inference Latency (p50 CPU)** | **19.42 ms** | 148.19 ms | **Model A is 7.6x faster on CPU** |
| **Mean Latency (CPU)** | **20.51 ms** | 149.42 ms | Model A faster inference |

## 2. Per-Class Performance Breakdown (Expanded Test Set: 648 Images)

### Model A (Custom 3-block CNN)
| Canonical Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Tomato___Early_blight** | 0.6638 | 0.5033 | 0.5725 | 153 |
| **Tomato___Late_blight** | 0.8213 | 0.8120 | 0.8166 | 266 |
| **Tomato___healthy** | 0.8178 | 0.9607 | 0.8835 | 229 |

### Model B (MobileNetV2 Frozen)
| Canonical Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Tomato___Early_blight** | 0.8382 | 0.7451 | 0.7889 | 153 |
| **Tomato___Late_blight** | 0.8841 | 0.9173 | 0.9004 | 266 |
| **Tomato___healthy** | 0.9407 | 0.9694 | 0.9548 | 229 |

## 3. Confusion Matrices (Expanded Test Set)

```
Model A (Custom CNN) Confusion Matrix:
                     Pred Early Blight   Pred Late Blight   Pred Healthy
Actual Early Blight:       77                  42                 34          
Actual Late Blight:        35                  216                15          
Actual Healthy:            4                   5                  220         

Model B (MobileNetV2 Frozen) Confusion Matrix:
                     Pred Early Blight   Pred Late Blight   Pred Healthy
Actual Early Blight:       114                 29                 10          
Actual Late Blight:        18                  244                4           
Actual Healthy:            4                   3                  222         
```

## 4. Honest Assessment of Dataset Size & Target Fulfillment

- **Target:** Provisional target of ~10,000 unique usable images.
- **Current Reality:** Authentic pool is **4,325 unique images** (PlantVillage: 3,813, PlantDoc: 262, Mendeley Pakistan: 250).
- **Why the 10,000 Target Cannot Be Authentically Claimed:**
  1. The external Mendeley Pakistan claim of 7,200 images consisted of synthetic augmentations (`_aug712aug274.jpg`), which were strictly excluded per the requirement not to count augmentations as new images.
  2. 119 Mendeley raw images were exact bit-for-bit SHA-256 duplicates of PlantDoc images.
  3. Early Blight is capped globally across all authentic sources at **1,022 unique images**.
  4. We refused to fabricate artificial synthetic copies or inflate counts to superficially hit 10,000.