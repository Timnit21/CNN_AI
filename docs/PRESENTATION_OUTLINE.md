# Presentation Outline: AgriAI

**Project Title**: AgriAI: Tomato Leaf Disease Classification Using Convolutional Neural Networks  
**Target Format**: 12-Slide Technical & Executive Presentation  

---

### Slide 1 — Title
- **Title**: AgriAI: Tomato Leaf Disease Classification Using Convolutional Neural Networks
- **Subtitle**: Automated Foliar Disease Diagnosis with Deep Learning
- **Author/Team**: AgriAI Engineering Team
- **Key Focus**: Deep learning computer vision pipeline for agricultural crop health monitoring
- **Recommended Visual**: High-resolution banner image of healthy vs. diseased tomato leaves with project logo.

---

### Slide 2 — Problem Statement
- **Title**: Problem Statement: Tomato Crop Disease Threats
- **Foliar Pathogens**: Early Blight (*Alternaria solani*) and Late Blight (*Phytophthora infestans*) cause devastating yield losses in global tomato production.
- **Manual Diagnosis Challenges**: Expert scouting is labor-intensive, slow, subjective, and inaccessible in remote farming regions.
- **Economic & Environmental Impact**: Delayed detection leads to total crop failure; misdiagnosis causes unnecessary broad-spectrum chemical application.
- **AI Opportunity**: Automated computer vision on mobile/edge devices provides rapid, accessible diagnostic assistance.
- **Recommended Visual**: Side-by-side comparison of Early Blight concentric lesions vs. Late Blight necrotic water-soaked patches.

---

### Slide 3 — Objectives
- **Title**: Project Objectives
- **Target Disease Classification**: Accurately classify tomato foliage into 3 categories: *Early Blight*, *Late Blight*, and *Healthy*.
- **CNN Development**: Design, train, and optimize a lightweight custom Convolutional Neural Network.
- **Experimental Exploration**: Benchmark the baseline against deeper architectures, Batch Normalization, and hyperparameter tuning.
- **Methodological Integrity**: Enforce strict data leakage prevention by isolating test data until post-selection.
- **Deployment-Ready Tooling**: Provide standalone CLI inference tooling with full class probability distributions.
- **Recommended Visual**: High-level system diagram from leaf photo input to predicted probability outputs.

---

### Slide 4 — Dataset
- **Title**: Dataset Summary & Partitioning
- **Data Source**: Verified PlantVillage RGB tomato subset.
- **Total Valid Dataset**: 4,486 unique images after deduplicating 14 byte-identical raw files.
- **Class Distribution**: 1,000 Early Blight (22.3%), 1,901 Late Blight (42.4%), 1,585 Healthy (35.3%).
- **Stratified Split (Seed 42)**: 70% Train (3,140 images) | 15% Validation (673 images) | 15% Held-Out Test (673 images).
- **Zero Leakage**: Hash verification confirms 0 cross-split SHA-256 overlap.
- **Recommended Visual**: Pie chart of class distribution and bar chart of stratified train/validation/test split counts.

---

### Slide 5 — Data Preprocessing
- **Title**: Image Preprocessing & Augmentation Pipeline
- **Image Standardization**: All images converted to 3-channel RGB and resized to 224 × 224 pixels.
- **In-Model Normalization**: `Rescaling(1/255.0)` layer maps pixel intensities to `[0.0, 1.0]`.
- **Training-Only Augmentation**: Random horizontal flip, rotation (±5%), and zoom (±10%) to prevent overfitting.
- **Deterministic Inference**: Augmentation layers automatically deactivate during validation, test, and inference modes.
- **Recommended Visual**: Visual comparison showing an original leaf image alongside augmented variations.

---

### Slide 6 — CNN Architecture
- **Title**: Baseline CNN Architecture Specification
- **Feature Extractor**: 3 sequential convolutional blocks with 32, 64, and 128 filters (3×3 kernel, ReLU, 'same' padding) each followed by MaxPooling (2×2).
- **Spatial Flattening**: Feature maps flattened to a 73,728-unit dense representation preserving spatial pattern distribution.
- **Regularized Head**: Dense layer (128 units, ReLU) with Dropout (0.5) to mitigate co-adaptation.
- **Output Layer**: 3-unit Softmax dense layer producing normalized class probabilities.
- **Total Parameters**: 12,938,819 parameters (100% trainable).
- **Recommended Visual**: Layer-by-layer architectural flow diagram ([results/model_architecture.png](file:///C:/Users/timnit/Desktop/CNN_AI/results/model_architecture.png)).

---

### Slide 7 — Training Methodology
- **Title**: Training Dynamics & Optimization
- **Optimizer & Loss**: Adam optimizer (learning rate = 0.001) with Sparse Categorical Cross-Entropy.
- **Batching**: Batch size 32 over maximum 30 epochs on training split (3,140 images).
- **Adaptive Callbacks**: `EarlyStopping` (patience = 5 on validation loss) and `ReduceLROnPlateau` (factor = 0.5, patience = 2).
- **Optimal Checkpoint**: Restored best weights from Epoch 14 (Validation Accuracy: 96.58%, Validation Loss: 0.086766).
- **Recommended Visual**: Dual line plot of training vs. validation accuracy and loss curves across epochs.

---

### Slide 8 — Model Experiments
- **Title**: Phase 7 Hyperparameter & Architecture Exploration
- **Experimental Scope**: Tested 5 candidate variations using validation data only (no test set access).
- **Architectural Variations**: Evaluated 4-block Conv-BN hierarchy with Global Average Pooling (GAP) and a 3-block compact model.
- **Hyperparameter Variations**: Evaluated learning rates (0.001 vs. 0.0003), dropout rates (0.5 vs. 0.3), and batch sizes (32 vs. 64).
- **Key Finding**: Global Average Pooling compressed parameters (~423K) but suffered lower peak validation accuracy (best candidate: 94.95%) compared to the baseline.
- **Recommended Visual**: Summary table comparing validation accuracy, loss, parameter counts, and training time across all 5 experiments.

---

### Slide 9 — Final Model Selection
- **Title**: Model Selection: Baseline Proven Superior
- **Selection Criterion**: Governed strictly by highest validation accuracy, with validation loss as tie-breaker.
- **Validation Comparison**:
  - Baseline CNN: **96.58%** Val Accuracy | **0.086766** Val Loss
  - Best Phase 7 Candidate (`drop=0.3`): **94.95%** Val Accuracy | **0.181050** Val Loss
- **Selection Decision**: Baseline CNN selected as final model (+1.63% validation accuracy advantage over best candidate).
- **Integrity Statement**: Zero test data was used in selecting the final model.
- **Recommended Visual**: Bar graph comparing baseline validation accuracy against all 5 candidate experiments.

---

### Slide 10 — Final Test Results
- **Title**: Final Test Evaluation on Held-Out Split
- **Held-Out Test Set**: 673 unseen images evaluated once after final model selection.
- **Overall Benchmark Metrics**:
  - **Accuracy**: **94.80%** (638 / 673 images correctly identified)
  - **Weighted Precision / Recall / F1**: **94.83%** / **94.80%** / **94.81%**
  - **Macro F1-Score**: **94.17%**
- **Per-Class Breakdown**:
  - Healthy: **99.58% F1** (237 / 238 correct)
  - Late Blight: **93.83% F1** (266 / 285 correct)
  - Early Blight: **89.11% F1** (135 / 150 correct)
- **Generalization Gap**: Only −1.78% difference between validation and test accuracy.
- **Recommended Visual**: Test confusion matrix heatmap ([results/final_confusion_matrix.png](file:///C:/Users/timnit/Desktop/CNN_AI/results/final_confusion_matrix.png)).

---

### Slide 11 — Limitations and Future Work
- **Title**: Practical Limitations & Roadmap
- **Dataset Constraints**: PlantVillage images are lab-staged on uniform backdrops; outdoor field lighting and complex backgrounds may cause domain shift.
- **Diagnostic Scope**: Closed 3-class classifier cannot diagnose other tomato diseases or abiotic nutrient deficiencies.
- **Roadmap 1 — Field Dataset Expansion**: Integrate in-the-wild agricultural datasets (e.g., PlantDoc) for multi-condition robustness.
- **Roadmap 2 — Explainability**: Implement Grad-CAM heatmap overlays to visualize foliar lesion focus areas.
- **Roadmap 3 — Edge Deployment**: Export to TensorFlow Lite (TFLite INT8) for sub-100ms smartphone app execution.
- **Recommended Visual**: Mockup of mobile application UI displaying leaf camera preview with Grad-CAM heatmap overlay.

---

### Slide 12 — Conclusion
- **Title**: Conclusion & Summary
- **Validated Performance**: High-accuracy (94.80% test accuracy, 94.81% weighted F1) tomato leaf disease classifier.
- **Methodological Excellence**: Fully reproducible pipeline with strict data isolation and transparent validation-based selection.
- **Reliable Foliar Screening**: Near-perfect healthy leaf screening (99.58%) and strong blight detection (89.11%–93.83% F1).
- **Ready for Review**: Open-source, clean codebase with comprehensive documentation and verified CLI inference.
- **Recommended Visual**: Summary scorecard highlighting 94.80% Accuracy, 94.81% F1, and 673 Test Images.
