# Baseline CNN architecture

This document describes the implemented baseline in `src/model.py`. The model is constructed and forward-pass tested; it has **not** been trained. No accuracy or evaluation results exist yet.

## Input and preprocessing

- Input tensor: `224 × 224 × 3` (RGB; batch shape `(None, 224, 224, 3)`).
- `Rescaling(1/255)` maps image pixel values from `[0, 255]` to `[0, 1]`.
- Training-time augmentation layers: random horizontal flip, mild random rotation (factor `0.05`), and mild random zoom (`0.10`). Keras disables random augmentation behavior for validation/inference calls made with `training=False`.

## Layer-by-layer shapes

| Stage | Output shape (one image) |
|---|---|
| Input / rescale / augmentation | `224 × 224 × 3` |
| Conv2D, 32 filters, `3 × 3`, same padding, ReLU | `224 × 224 × 32` |
| MaxPooling2D, `2 × 2` | `112 × 112 × 32` |
| Conv2D, 64 filters, `3 × 3`, same padding, ReLU | `112 × 112 × 64` |
| MaxPooling2D, `2 × 2` | `56 × 56 × 64` |
| Conv2D, 128 filters, `3 × 3`, same padding, ReLU | `56 × 56 × 128` |
| MaxPooling2D, `2 × 2` | `28 × 28 × 128` |
| Flatten | `100,352` |
| Dense, 128 units, ReLU | `128` |
| Dropout, rate `0.5` | `128` |
| Dense, 3 units, Softmax | `3` class probabilities |

The three pooling layers halve each spatial dimension: `224 → 112 → 56 → 28`. Same-padded convolutions retain the spatial dimensions. Flatten therefore receives `28 × 28 × 128 = 100,352` values.

## Compilation

- Optimizer: Adam, learning rate `0.001`.
- Loss: sparse categorical cross-entropy (integer class labels).
- Metric: accuracy.
- Hidden activation: ReLU on all three convolution layers and the 128-unit dense layer.
- Output activation: Softmax across exactly three classes.

## Actual model parameter counts

Obtained from TensorFlow's `model.summary()` and checked against the model's trainable-weight tensors:

- Trainable parameters: **12,938,819**
- Non-trainable parameters: **0**
- Total parameters: **12,938,819**

The large dense layer after `Flatten` accounts for most of the parameters. These are architecture counts, not measures of model performance.

## Verification performed

`python src/model.py` constructs the model, prints its summary and parameter counts, verifies input/output shapes and activation layers, and runs one all-zero dummy image through it with `training=False`. The output shape was `(1, 3)` and Softmax probabilities summed to 1. This is only a forward-pass smoke test. No dataset is loaded and no call to `fit()` or test evaluation is made.
