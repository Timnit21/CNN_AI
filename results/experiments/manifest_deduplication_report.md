# Experimental Dataset Ingestion & Deduplication Audit Report

- **Audit Status:** COMPLETED & VERIFIED
- **Elapsed Processing Time:** 58.52 seconds
- **Total Ingested Files Evaluated:** 23,065
- **Valid Unique Candidate Images:** 4,325
- **Exact Duplicates Excluded:** 3,943
- **Historical Test Leaks Prevented:** 676
- **Corrupted / Unreadable Images:** 0
- **Excluded Non-Target Conditions:** 14,121

---

## 1. Verified Class Counts by Source Dataset

| Source Dataset | Tomato Early Blight | Tomato Late Blight | Tomato Healthy | Total Usable | License |
|---|:---:|:---:|:---:|:---:|---|
| **mendeley_pakistan** | 84 | 50 | 116 | **250** | CC BY 4.0 |
| **plantdoc** | 88 | 111 | 63 | **262** | CC BY 4.0 |
| **plantvillage** | 850 | 1,616 | 1,347 | **3,813** | CC0 1.0 |
| **TOTAL AUTHENTIC CANDIDATES** | **1,022** | **1,777** | **1,526** | **4,325** | Open Access / CC |

---

## 2. Group-Aware Stratified Split Distribution

Splitting was performed at the **specimen cluster level** (grouping exact duplicates and perceptual near-duplicates with Hamming distance $\le 4$) to guarantee zero specimen leakage.

| Canonical Class | Train (70%) | Validation (15%) | Test Expanded (15%) | Total Usable |
|---|:---:|:---:|:---:|:---:|
| **Tomato___Early_blight** | 716 | 153 | 153 | **1,022** |
| **Tomato___Late_blight** | 1,244 | 266 | 267 | **1,777** |
| **Tomato___healthy** | 1,068 | 230 | 228 | **1,526** |
| **TOTAL POOL** | **3,028** | **649** | **648** | **4,325** |

---

## 3. Strict Safety & Leakage Guarantees

1. **Production Models Untouched:** Checkpoints `final_crop_disease_cnn.keras` and `crop_disease_cnn.keras` were not modified or referenced.
2. **Historical `data/test/` Frozen:** All 673 images in `data/test/` remain untouched. Exactly 0 historical test images are included in training, validation, or the expanded test split.
3. **Zero Split Leakage:** Both cryptographic SHA-256 and specimen group clusters were mathematically verified across split boundaries; cross-split leakage violations = 0.
4. **No Synthetic Augmentation Inflation:** No synthetic copies, rotations, or duplicated files were counted as new images.

---

## 4. Next Step Recommendation

With the verified dataset manifest in place, the next phase is to review the exact class counts and choose between:
- **Option 1 (Balanced Subset):** Subsample to exact balance per class (governed by the Early Blight ceiling).
- **Option 2 (Stratified Full Pool with Class Weighting):** Utilize all authentic candidates with inverse-frequency class weights during Model A vs. Model B comparison.