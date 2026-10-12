"""Convolutional Neural Network models for tomato leaf disease classification.

This module provides architecture definitions, parameter counting, and verification
routines for both the baseline 3-block CNN and the enhanced-augmentation CNN variant.
"""

from __future__ import annotations

import json
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "model_config.json"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 3
DEFAULT_LEARNING_RATE = 0.001


def load_model_config(config_path: Path = CONFIG_PATH) -> dict[str, object]:
	"""Load machine-readable model configuration if available."""
	if config_path.is_file():
		with config_path.open("r", encoding="utf-8") as f:
			return json.load(f)
	return {}


def build_model(
	num_classes: int = NUM_CLASSES,
	learning_rate: float = DEFAULT_LEARNING_RATE,
	use_enhanced_augmentation: bool = False,
	dropout_rate: float = 0.5,
) -> tf.keras.Model:
	"""Build and compile the sequential CNN for tomato leaf disease classification.

	Args:
		num_classes: Number of output disease classes (default: 3).
		learning_rate: Initial learning rate for Adam optimizer (default: 0.001).
		use_enhanced_augmentation: If True, incorporates additional lighting, contrast,
			translation, and vertical flip augmentations. If False, uses baseline augmentation.
		dropout_rate: Dropout rate before the final classification head (default: 0.5).

	Returns:
		Compiled tf.keras.Model ready for training or inference.
	"""
	if num_classes != NUM_CLASSES:
		raise ValueError(f"This project requires exactly {NUM_CLASSES} output classes.")

	layers: list[tf.keras.layers.Layer] = [
		tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), name="image"),
		tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling"),
	]

	if use_enhanced_augmentation:
		# Enhanced data augmentation pipeline for improved background & lighting robustness
		layers.extend(
			[
				tf.keras.layers.RandomFlip("horizontal_and_vertical", name="random_flip_hv"),
				tf.keras.layers.RandomRotation(0.10, name="random_rotation"),
				tf.keras.layers.RandomZoom(0.15, name="random_zoom"),
				tf.keras.layers.RandomTranslation(0.08, 0.08, name="random_translation"),
				tf.keras.layers.RandomContrast(0.15, value_range=(0.0, 1.0), name="random_contrast"),
				tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0), name="random_brightness"),
			]
		)
	else:
		# Baseline data augmentation pipeline
		layers.extend(
			[
				tf.keras.layers.RandomFlip("horizontal", name="random_horizontal_flip"),
				tf.keras.layers.RandomRotation(0.05, name="random_rotation"),
				tf.keras.layers.RandomZoom(0.10, name="random_zoom"),
			]
		)

	# 3-Block Convolutional Feature Extractor
	layers.extend(
		[
			tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_32"),
			tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling_2d_1"),
			tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_64"),
			tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling_2d_2"),
			tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2d_128"),
			tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling_2d_3"),
			tf.keras.layers.Flatten(name="flatten"),
			tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
			tf.keras.layers.Dropout(dropout_rate, name="dropout_50_percent"),
			tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities"),
		]
	)

	model = tf.keras.Sequential(
		layers,
		name="tomato_disease_cnn_enhanced" if use_enhanced_augmentation else "tomato_disease_cnn",
	)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def count_parameters(model: tf.keras.Model) -> tuple[int, int]:
	"""Return actual trainable and non-trainable parameter counts."""
	trainable = sum(int(tf.keras.backend.count_params(weight)) for weight in model.trainable_weights)
	non_trainable = sum(
		int(tf.keras.backend.count_params(weight)) for weight in model.non_trainable_weights
	)
	return trainable, non_trainable


def verify_model(model: tf.keras.Model, print_summary: bool = True) -> dict[str, object]:
	"""Assert required architecture properties and run a single forward pass."""
	expected_input_shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
	expected_output_shape = (None, NUM_CLASSES)
	if model.input_shape != expected_input_shape:
		raise AssertionError(f"Expected input {expected_input_shape}, got {model.input_shape}.")
	if model.output_shape != expected_output_shape:
		raise AssertionError(f"Expected output {expected_output_shape}, got {model.output_shape}.")
	if model.layers[-1].activation.__name__ != "softmax":
		raise AssertionError("The output layer must use softmax activation.")

	relu_layers = [
		layer.name
		for layer in model.layers
		if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))
		and getattr(layer, "activation", None) is not None
		and layer.activation.__name__ == "relu"
	]
	if len(relu_layers) != 4:
		raise AssertionError(f"Expected ReLU on 3 convolution layers and 1 dense layer, got {relu_layers}.")

	expected_convolutions = [(32, "same"), (64, "same"), (128, "same")]
	actual_convolutions = [
		(layer.filters, layer.padding)
		for layer in model.layers
		if isinstance(layer, tf.keras.layers.Conv2D)
	]
	if actual_convolutions != expected_convolutions:
		raise AssertionError(f"Unexpected convolution configuration: {actual_convolutions}.")

	dummy_batch = tf.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=tf.float32)
	probabilities = model(dummy_batch, training=False)
	if tuple(probabilities.shape) != (1, NUM_CLASSES):
		raise AssertionError(f"Unexpected forward-pass output shape: {probabilities.shape}.")
	tf.debugging.assert_near(
		tf.reduce_sum(probabilities, axis=1),
		tf.ones((1,), dtype=tf.float32),
		message="Softmax output probabilities should sum to 1.",
	)

	trainable_params, non_trainable_params = count_parameters(model)
	if trainable_params + non_trainable_params != model.count_params():
		raise AssertionError("Parameter totals do not match model.count_params().")
	if print_summary:
		model.summary()
		print(f"Input shape: {model.input_shape}")
		print(f"Output shape: {model.output_shape}")
		print(f"Trainable parameters: {trainable_params:,}")
		print(f"Non-trainable parameters: {non_trainable_params:,}")
		print(f"Total parameters: {model.count_params():,}")
		print(f"Forward-pass output: {probabilities.numpy()[0]}")

	return {
		"input_shape": model.input_shape,
		"output_shape": model.output_shape,
		"trainable_parameters": trainable_params,
		"non_trainable_parameters": non_trainable_params,
		"total_parameters": model.count_params(),
		"forward_pass_shape": tuple(probabilities.shape),
		"softmax_probability_sum": float(tf.reduce_sum(probabilities[0]).numpy()),
		"activation": model.layers[-1].activation.__name__,
		"relu_layers": relu_layers,
	}


if __name__ == "__main__":
	baseline_model = build_model()
	verification_results = verify_model(baseline_model)
	print("Architecture verification: PASS")
