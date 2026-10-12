# Baseline training — Phase 5

## Dataset and procedure

The experiment used the locally prepared PlantVillage three-class subset documented in [DATASET.md](DATASET.md): 3,140 training images and 673 validation images at 224 × 224 RGB. The held-out test set was not loaded, accessed, or evaluated by the training runner.

The existing baseline CNN was used without architecture changes. Training configuration:

| Setting | Value |
|---|---|
| Batch size | 32 |
| Maximum epochs | 30 |
| Epochs completed | 19 |
| Random seed | 42 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Sparse categorical cross-entropy |
| Metric | Accuracy |
| EarlyStopping | Monitor `val_loss`; patience 5; restore best weights |
| ModelCheckpoint | Monitor `val_accuracy`; save best only to `models/crop_disease_cnn.keras` |
| Augmentation | Existing random horizontal flip, mild rotation, and mild zoom layers; active only for training calls |

Training ran on TensorFlow CPU on native Windows. It stopped at epoch 19 when EarlyStopping triggered and restored weights from epoch 14. The best validation-accuracy checkpoint was saved at epoch 14. In this run epoch 14 also had the best validation loss.

## Measured results

Values below are from the recorded Keras training history (`results/training_history.json`). They are training/validation metrics only, not test metrics.

| Measurement | Value |
|---|---:|
| Best checkpoint epoch (highest validation accuracy) | 14 |
| Best validation accuracy | 0.9658247 (96.58%) |
| Best validation loss | 0.0867663 |
| Best validation loss epoch | 14 |
| Final recorded training accuracy (epoch 19) | 0.9496815 (94.97%) |
| Final recorded validation accuracy (epoch 19) | 0.8811293 (88.11%) |
| Final recorded training loss (epoch 19) | 0.1398808 |
| Final recorded validation loss (epoch 19) | 0.3081443 |
| EarlyStopping triggered | Yes, at epoch 19 |

The checkpoint was subsequently loaded successfully. Its input shape is `(None, 224, 224, 3)` and output shape is `(None, 3)`.

## Learning curves and interpretation

- [Training and validation accuracy](../results/training_accuracy.png)
- [Training and validation loss](../results/training_loss.png)

Both curves improve substantially during the early epochs. Validation accuracy reaches its observed maximum at epoch 14. After that point training loss remains low and training accuracy is high, but validation metrics fluctuate and the final validation loss rises to 0.3081 from its minimum of 0.0868. The final epoch also has a wider training/validation accuracy gap than the selected epoch.

**Interpretation:** the baseline does not show persistent underfitting in these curves. Validation performance indicates reasonable generalization around the selected epoch 14 checkpoint, while the later divergence/fluctuation is evidence of overfitting after that point. Early stopping selected an earlier checkpoint accordingly. The separate held-out test set was subsequently evaluated in Phase 6; see [TESTING.md](TESTING.md) for that independent measured result.

## Artifacts

- `models/crop_disease_cnn.keras` — best-only checkpoint by validation accuracy.
- `results/training_accuracy.png` — training and validation accuracy by epoch.
- `results/training_loss.png` — training and validation loss by epoch.
- `results/training_history.json` — full measured Keras history and run configuration.

## Held-out test evaluation (Phase 6)

The Phase 5 training runner uses only `data/train` and `data/validation`; it does not load the test split. The saved epoch-14 checkpoint was subsequently evaluated once on the separate 673-image `data/test` split. The measured test accuracy was 94.80%, compared with the best validation accuracy of 96.58% (a difference of −1.78 percentage points). Detailed test metrics, per-class analysis, and the confusion matrix are documented in [TESTING.md](TESTING.md). The evaluation did not retrain or modify the model.
