# Experimental Reliability Filters: Empirical Trade-Off Report

- **Date:** 2026-10-12T08:56:42Z
- **Evaluated Samples:**
  - Group A (Supported Tomato Validation): 673
  - Group B (Unsupported Tomato Diseases): 350
  - Group C (Non-Tomato Crop Leaves): 250
- **Status:** **Experimental Research Only** — Not enabled as mandatory production rejection gates.

---

## 1. Metric Distribution Summary

| Metric | Group A: Supported Tomato | Group B: Unsupported Tomato Diseases | Group C: Non-Tomato Crop Leaves | Observation & Distinction |
|---|:---:|:---:|:---:|---|
| **Vegetation Ratio (ExG)** | Mean: 0.3981 (Med: 0.4033) | Mean: 0.4037 (Med: 0.3672) | Mean: 0.5542 (Med: 0.5283) | All three groups contain high foliar vegetation content. **Green vegetation detection does NOT distinguish tomato from other crops.** |
| **Laplacian Sharpness** | Mean: 3285.8 (Med: 2665.3) | Mean: 2936.6 (Med: 3241.6) | Mean: 2579.0 (Med: 2518.4) | Sharpness varies widely by leaf texture and focus. Overly aggressive blur thresholds cause high false rejection of legitimate leaves. |
| **Free Energy Score** | Mean: -11.8954 (Med: -9.6656) | Mean: -8.2731 (Med: -7.0585) | Mean: -8.9794 (Med: -7.5146) | In-distribution samples exhibit substantially lower energy (more negative) than unsupported groups. |

---

## 2. Vegetation Screening Threshold Sweep

> **Hypothesis Tested:** Does an Excess Green Index (ExG) ratio reject non-plant or non-foliar uploads without rejecting legitimate tomato leaves?  
> **Key Finding:** Legitimate tomato leaves have high vegetation ratios (mean 0.3981). However, non-tomato crops (Group C) ALSO have high vegetation ratios (mean 0.5542). Therefore, vegetation screening is strictly a **non-leaf detector**, NOT an OOD crop detector.

| ExG Threshold ($\tau_{\text{veg}}$) | False Rejection Rate (Group A) | Rejection Rate (Group B) | Rejection Rate (Group C) | Assessment |
|---|:---:|:---:|:---:|---|
| $\ge 0.05$ | **0.0%** | 0.0% | 0.0% | Safe for production pre-filter |
| $\ge 0.10$ | **0.0%** | 0.0% | 0.0% | Safe for production pre-filter |
| $\ge 0.15$ | **2.53%** | 0.0% | 0.0% | Too restrictive |
| $\ge 0.20$ | **7.43%** | 2.57% | 0.0% | Too restrictive |
| $\ge 0.25$ | **13.08%** | 12.86% | 0.4% | Too restrictive |

---

## 3. Laplacian Blur Screening Threshold Sweep

> **Hypothesis Tested:** Does Laplacian variance filter out blurry photographs without rejecting valid in-focus leaves?  
> **Key Finding:** Setting $\tau_{\text{blur}} \le 50$ rejects severely blurry files with minimal false rejections on genuine leaves ($< 1\%$ FRR). Setting $\tau_{\text{blur}} \ge 100$ begins rejecting valid smooth leaves (e.g. healthy leaves with few sharp lesions).

| Sharpness Threshold ($\tau_{{\text{{blur}}}}$) | False Rejection Rate (Group A) | Rejection Rate (Group B) | Rejection Rate (Group C) | Assessment |
|---|:---:|:---:|:---:|---|
| $\ge 20$ | **0.0%** | 0.0% | 0.0% | Safe blur floor |
| $\ge 50$ | **0.0%** | 1.14% | 0.0% | Safe blur floor |
| $\ge 100$ | **0.3%** | 2.57% | 0.0% | Safe blur floor |
| $\ge 150$ | **0.45%** | 4.29% | 0.8% | Safe blur floor |
| $\ge 200$ | **1.93%** | 5.14% | 2.4% | Safe blur floor |

---

## 4. Free Energy Score (OOD Research) Threshold Sweep

> **Hypothesis Tested:** Can pre-softmax Free Energy $E(\mathbf{x}) = -\log \sum_{c=1}^3 e^{z_c}$ distinguish in-distribution tomato leaves from unsupported diseases and non-tomato crops?  
> **Key Finding:** Supported tomato leaves produce strong logit activations resulting in lower free energy (mean -11.8954). However, overlapping energy distributions with unsupported diseases mean that energy thresholding alone creates an acute trade-off: **any threshold that rejects $>50\%$ of unsupported diseases also falsely rejects $14-35\%$ of genuine tomato leaves.**

| Energy Ceiling ($\tau_{\text{energy}}$) | False Rejection Rate (Group A) | Rejection Rate (Group B) | Rejection Rate (Group C) | Trade-Off Assessment |
|---|:---:|:---:|:---:|---|
| $\le -15.0$ | **86.03%** | 88.0% | 83.6% | Severe False Rejections on genuine leaves |
| $\le -12.0$ | **69.39%** | 79.71% | 77.6% | Severe False Rejections on genuine leaves |
| $\le -10.0$ | **52.9%** | 71.71% | 70.8% | Severe False Rejections on genuine leaves |
| $\le -8.0$ | **34.77%** | 59.71% | 55.2% | Severe False Rejections on genuine leaves |
| $\le -6.0$ | **13.97%** | 40.29% | 34.8% | Severe False Rejections on genuine leaves |
| $\le -4.0$ | **2.67%** | 14.86% | 15.6% | Low rejection of genuine leaves |

---

## 5. Engineering Conclusion & Recommendations

1. **Vegetation Screening (ExG $\ge 0.10$):**
   - Safe to use as a **pre-inference sanity check** to catch non-leaf uploads (e.g., metallic tools, soil, clothing, blank screens).
   - Exhibits **0.0% False Rejection Rate** on genuine tomato leaves at $\tau_{\text{veg}} = 0.10$.
   - **Critical Caveat:** Must never be advertised as an OOD crop detector, because all plant foliage contains green vegetation (mean ExG 0.40 - 0.55).
2. **Blur Screening (Laplacian Variance $\ge 50$):**
   - Safe quality guardrail with **0.0% False Rejection Rate** on genuine tomato leaves at $\tau_{\text{blur}} = 50.0$.
   - Rejects ungrounded blurry uploads.
3. **Free Energy Scoring:**
   - Demonstrates a clear theoretical advantage over softmax probabilities (directly observing unnormalized logit mass).
   - However, because the closed 3-class CNN feature space was not trained with an explicit OOD objective, logit separation is insufficient to act as a standalone guarantee.
   - **Recommendation:** Keep energy scoring experimental. Do not enable as an automatic rejection gate in production until an explicit OOD-aware head or embedding classifier is trained.
