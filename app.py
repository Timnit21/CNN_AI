"""AgriAI: Streamlit Web Interface for Tomato Leaf Disease Classification.

This application provides a web interface for uploading tomato leaf photographs,
running inference with the trained Convolutional Neural Network (CNN),
and displaying predicted disease classes, confidence scores, and probability breakdowns.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

# ==============================================================================
# Configuration and Constants
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (  # noqa: E402
    DEFAULT_MIN_MARGIN_THRESHOLD,
    DEFAULT_MIN_SCORE_THRESHOLD,
    evaluate_uncertainty,
    validate_probabilities,
)

CONFIG_PATH = PROJECT_ROOT / "model_config.json"
MODEL_PATH = PROJECT_ROOT / "models" / "final_crop_disease_cnn.keras"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "crop_disease_cnn.keras"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "results" / "final_confusion_matrix.png"


def load_app_config() -> dict[str, Any]:
    """Load configuration from model_config.json if available."""
    if CONFIG_PATH.is_file():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


APP_CONFIG = load_app_config()
IMAGE_SIZE = (
    APP_CONFIG.get("input_specification", {}).get("image_height", 224),
    APP_CONFIG.get("input_specification", {}).get("image_width", 224),
)
MAX_UPLOAD_SIZE_MB = float(APP_CONFIG.get("input_specification", {}).get("max_upload_size_mb", 15.0))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_SIZE_MB * 1024 * 1024)
MIN_IMAGE_DIMENSION = int(APP_CONFIG.get("input_specification", {}).get("min_image_dimension", 64))

# Exact class order required by model output layer
CLASS_NAMES = APP_CONFIG.get("classes", {}).get("class_names", [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
])

# Readable labels for user display
READABLE_LABELS = APP_CONFIG.get("classes", {}).get("display_names", {
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___healthy": "Healthy",
})

# Scientific names and agronomic context
CLASS_DETAILS = {
    "Tomato___Early_blight": {
        "scientific_name": "Alternaria solani",
        "type": "Fungal Pathogen",
        "description": (
            "Early Blight is caused by the fungal pathogen *Alternaria solani*. It produces "
            "characteristic dark brown to black spots with concentric rings ('target-board' pattern) "
            "primarily on older foliage, often surrounded by a yellow chlorotic halo. If unmanaged, "
            "it leads to substantial defoliation and yield reduction."
        ),
        "management": (
            "Remove and destroy infected lower foliage; avoid overhead irrigation to reduce canopy wetness; "
            "practice crop rotation; apply recommended preventative fungicides when conditions favor disease spread."
        ),
    },
    "Tomato___Late_blight": {
        "scientific_name": "Phytophthora infestans",
        "type": "Oomycete Pathogen",
        "description": (
            "Late Blight is caused by the destructive oomycete *Phytophthora infestans*. It manifests "
            "as rapidly expanding, irregular water-soaked lesions that turn dark brown or purplish-black. "
            "Under humid conditions, delicate white fungal sporulation may appear on the undersides of leaves."
        ),
        "management": (
            "Monitor fields closely during cool, wet weather; destroy severely infected plants immediately; "
            "ensure adequate plant spacing for airflow; apply targeted oomycete-specific protective fungicides."
        ),
    },
    "Tomato___healthy": {
        "scientific_name": "Solanum lycopersicum",
        "type": "Disease-Free Foliage",
        "description": (
            "The leaf exhibits uniform green coloration, intact cellular structure, and no visible "
            "necrotic lesions, water-soaked margins, or fungal sporulation."
        ),
        "management": (
            "Maintain balanced fertilization, consistent drip irrigation, and routine scouting to preserve plant vigor."
        ),
    },
}

DISCLAIMER_TEXT = (
    "Predictions are estimates produced by a machine-learning model. Performance on real-world "
    "images may differ from the training dataset. This tool is an educational prototype and is "
    "not a substitute for expert agricultural diagnosis."
)


# ==============================================================================
# Streamlit Page Setup & Styling
# ==============================================================================
def setup_page() -> None:
    """Configure Streamlit page layout and custom CSS styling."""
    st.set_page_config(
        page_title="AgriAI — Tomato Leaf Disease Classification",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.3rem;
            font-weight: 800;
            color: #1b4332;
            margin-bottom: 0.25rem;
            letter-spacing: -0.5px;
        }
        .main-subtitle {
            font-size: 1.05rem;
            color: #4a5568;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }
        .result-card {
            padding: 1.3rem;
            border-radius: 0.75rem;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 1.25rem;
        }
        .confidence-tag {
            display: inline-block;
            font-size: 1.15rem;
            font-weight: 700;
            padding: 0.35rem 0.85rem;
            border-radius: 0.5rem;
            margin-top: 0.4rem;
        }
        .tag-healthy {
            color: #166534;
            background-color: #dcfce7;
            border: 1px solid #86efac;
        }
        .tag-disease {
            color: #991b1b;
            background-color: #fee2e2;
            border: 1px solid #fca5a5;
        }
        .tag-inconclusive {
            color: #92400e;
            background-color: #fef3c7;
            border: 1px solid #fcd34d;
        }
        .tag-margin {
            color: #1e40af;
            background-color: #dbeafe;
            border: 1px solid #93c5fd;
        }
        .disclaimer-card {
            font-size: 0.88rem;
            color: #475569;
            background-color: #f8fafc;
            border-left: 4px solid #0284c7;
            padding: 0.85rem 1.15rem;
            border-radius: 0.35rem;
            margin-top: 2rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Model Loading & Preprocessing
# ==============================================================================
@st.cache_resource(show_spinner="Loading deep learning model checkpoint...")
def load_classification_model() -> tf.keras.Model:
    """Load the trained Keras CNN model once and cache it in memory.

    Checks primary model path (final_crop_disease_cnn.keras) and falls back
    to baseline checkpoint (crop_disease_cnn.keras) if needed.
    """
    target_path = MODEL_PATH if MODEL_PATH.is_file() else FALLBACK_MODEL_PATH
    if not target_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found. Expected model file at: '{MODEL_PATH}' "
            f"or fallback at: '{FALLBACK_MODEL_PATH}'."
        )

    model = tf.keras.models.load_model(target_path)

    # Validate model input/output dimensions
    expected_input_shape = (None, *IMAGE_SIZE, 3)
    if model.input_shape != expected_input_shape:
        raise ValueError(
            f"Model input shape mismatch: expected {expected_input_shape}, got {model.input_shape}."
        )
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model output mismatch: expected {len(CLASS_NAMES)} classes, got {model.output_shape[-1]}."
        )

    return model


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image into a 4D batch tensor of shape (1, 224, 224, 3).

    Note: The CNN model contains an internal Rescaling(1/255.0) layer,
    so raw pixel values in [0, 255] are preserved without external normalization.
    """
    # Ensure full pixel data is decoded into memory
    pil_image.load()

    # Ensure RGB color mode (drops alpha channel if PNG, converts grayscale if 1-channel)
    rgb_image = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image

    # Resize to exact expected input dimensions
    resized_image = rgb_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    # Convert to float32 NumPy array with shape (224, 224, 3)
    image_array = np.array(resized_image, dtype=np.float32)

    if image_array.shape != (*IMAGE_SIZE, 3):
        raise ValueError(
            f"Unexpected preprocessed shape {image_array.shape}; expected {(*IMAGE_SIZE, 3)}."
        )

    # Expand batch dimension -> (1, 224, 224, 3)
    return np.expand_dims(image_array, axis=0)


def predict(
    model: tf.keras.Model,
    input_batch: np.ndarray,
    min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
    min_margin_threshold: float = DEFAULT_MIN_MARGIN_THRESHOLD,
) -> tuple[str, float, dict[str, float], dict[str, Any]]:
    """Execute model prediction and return predicted class, top score, probability breakdown, and uncertainty analysis."""
    raw_predictions = model.predict(input_batch, verbose=0)[0]
    probabilities = np.asarray(raw_predictions, dtype=np.float64)

    uncertainty = evaluate_uncertainty(
        probabilities,
        CLASS_NAMES,
        min_score_threshold=min_score_threshold,
        min_margin_threshold=min_margin_threshold,
    )

    class_probs = {
        CLASS_NAMES[idx]: float(probabilities[idx])
        for idx in range(len(CLASS_NAMES))
    }

    predicted_class = uncertainty["predicted_class"]
    top_score = uncertainty["top_score"]

    return predicted_class, top_score, class_probs, uncertainty


# ==============================================================================
# Main Streamlit Application
# ==============================================================================
def main() -> None:
    """Render the AgriAI Streamlit application interface."""
    setup_page()

    # Sidebar: Model and Project Overview
    with st.sidebar:
        st.markdown("## 🌿 AgriAI Overview")
        st.markdown(
            "Automated tomato foliar disease diagnostic system powered by a custom "
            "Convolutional Neural Network (CNN)."
        )

        if CONFUSION_MATRIX_PATH.is_file():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Test Confusion Matrix (673 Held-Out Images)")

        st.markdown("### 📊 Benchmark Metrics")
        st.markdown(
            """
            - **Model Architecture**: 3-Block CNN (Conv 32/64/128 + Dense 128)
            - **Parameters**: 12,938,819 (100% Trainable)
            - **Validation Accuracy**: **96.58%**
            - **Test Accuracy**: **94.80%** (638 / 673 images)
            - **Target Input**: 224 × 224 × 3 RGB
            """
        )

        st.markdown("---")
        st.markdown("### 🔍 Target Classes")
        st.markdown(
            """
            1. **Early Blight** (*Alternaria solani*)
            2. **Late Blight** (*Phytophthora infestans*)
            3. **Healthy** (Disease-free foliage)
            """
        )

        st.markdown("---")
        st.markdown("### ⚙️ Diagnostic Guardrails")
        st.caption(
            "Provisional thresholds used to flag borderline or ambiguous classifications. "
            "These thresholds do not guarantee detection of non-tomato images."
        )
        min_score_thresh = st.slider(
            "Min Model Output Score",
            min_value=0.34,
            max_value=0.95,
            value=float(DEFAULT_MIN_SCORE_THRESHOLD),
            step=0.05,
            help="Classifications with top score below this threshold are marked Inconclusive.",
        )
        min_margin_thresh = st.slider(
            "Min Class Margin",
            min_value=0.00,
            max_value=0.50,
            value=float(DEFAULT_MIN_MARGIN_THRESHOLD),
            step=0.05,
            help="Minimum difference between top-1 and top-2 class scores needed for a conclusive result.",
        )

    # Main Header
    st.markdown('<div class="main-header">🌿 AgriAI: Tomato Leaf Disease Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">'
        'Upload a photograph of a tomato leaf in <strong>JPG, JPEG, or PNG</strong> format '
        'to identify potential diseases and view relative class score distributions.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load model with caching
    try:
        model = load_classification_model()
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"❌ Failed to load model: {exc}")
        st.stop()

    # Image Upload Section
    uploaded_file = st.file_uploader(
        "Upload a tomato leaf image (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Select a clear photograph of a single tomato leaf.",
    )

    if uploaded_file is not None:
        if uploaded_file.size == 0:
            st.error("❌ The uploaded file is empty (0 bytes). Please upload a valid image file.")
            return

        if uploaded_file.size > MAX_UPLOAD_BYTES:
            st.error(
                f"❌ The uploaded file ({uploaded_file.size / (1024 * 1024):.1f} MB) exceeds the maximum allowed size of {MAX_UPLOAD_SIZE_MB:.0f} MB. "
                "Please upload a standard mobile or compressed photograph."
            )
            return

        try:
            image = Image.open(uploaded_file)
            image.load()  # Force decode stream to catch truncated or corrupted files immediately
        except UnidentifiedImageError:
            st.error("❌ Unsupported or unreadable image file. Please upload a valid JPG, JPEG, or PNG photograph.")
            return
        except Image.DecompressionBombError:
            st.error("❌ Image dimensions exceed safe memory processing limits. Please upload a standard photograph.")
            return
        except (OSError, SyntaxError, ValueError):
            st.error("❌ The uploaded image file appears corrupted or truncated. Please try another photograph.")
            return
        except Exception:
            st.error("❌ Could not process uploaded image. Please ensure it is a valid photograph.")
            return

        if image.width < MIN_IMAGE_DIMENSION or image.height < MIN_IMAGE_DIMENSION:
            st.error(
                f"❌ Image resolution is too small ({image.width}×{image.height} px). "
                f"Minimum required resolution is {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} pixels for reliable diagnostic analysis."
            )
            return

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### 📷 Uploaded Leaf Image")
            st.image(
                image,
                use_container_width=True,
                caption=f"Image details: {image.size[0]}×{image.size[1]} px | Format: {image.format or 'RGB'}",
            )

        with col2:
            st.markdown("### 🔬 Diagnostic Results")
            with st.spinner("Analyzing foliar pattern with Convolutional Neural Network..."):
                try:
                    input_batch = preprocess_image(image)
                    predicted_class, top_score, class_probs, uncertainty = predict(
                        model,
                        input_batch,
                        min_score_threshold=min_score_thresh,
                        min_margin_threshold=min_margin_thresh,
                    )
                except Exception:
                    st.error("❌ An unexpected error occurred during model analysis. Please verify the image and try again.")
                    return

            is_conclusive = uncertainty["is_conclusive"]
            is_healthy = predicted_class == "Tomato___healthy"
            readable_label = READABLE_LABELS[predicted_class]
            details = CLASS_DETAILS[predicted_class]
            margin = uncertainty["margin"]

            if is_conclusive:
                tag_class = "tag-healthy" if is_healthy else "tag-disease"
                status_header = "Predicted Classification"
                display_title = readable_label
                subtitle = f"{details['scientific_name']} &bull; {details['type']}"
                border_color = "#e2e8f0"
            else:
                tag_class = "tag-inconclusive"
                status_header = "Diagnostic Status"
                display_title = "Inconclusive — Please try another image"
                subtitle = f"Tentative Lead: {readable_label} (Unverified &bull; Low Confidence)"
                border_color = "#f59e0b"

            # Result Summary Card
            st.markdown(
                f"""
                <div class="result-card" style="border-left: 5px solid {border_color};">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">{status_header}</div>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #0f172a; margin-top: 0.2rem;">{display_title}</div>
                    <div style="font-size: 0.95rem; color: #64748b; font-style: italic; margin-bottom: 0.4rem;">{subtitle}</div>
                    <div>
                        <span class="confidence-tag {tag_class}">Model Output Score: {top_score * 100:.2f}%</span>
                        <span class="confidence-tag tag-margin">Margin: {margin * 100:.2f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not is_conclusive:
                st.warning(
                    "⚠️ **Borderline / Low-Certainty Classification:**\n\n"
                    + "\n".join(f"• {r}" for r in uncertainty["reasons"])
                )

            # Relative Model Output Score Distribution
            st.markdown("#### Relative Model Output Score Distribution")
            st.caption("Scores represent uncalibrated relative outputs across the three supported classes, not guaranteed real-world probabilities.")
            for cls_key in CLASS_NAMES:
                prob = class_probs[cls_key]
                label_text = READABLE_LABELS[cls_key]
                st.write(f"**{label_text}**: `{prob * 100:.2f}%`")
                st.progress(float(min(max(prob, 0.0), 1.0)))

            # Agronomic Profile & Management
            if is_conclusive:
                st.markdown("#### Agronomic Profile & Recommendation")
                st.info(f"{details['description']}\n\n**Recommended Action:** {details['management']}")
            else:
                st.markdown("#### Diagnostic Advisory")
                st.warning(
                    "**Treatment advice withheld:** Definitive disease treatments and chemical sprays "
                    "should **NOT** be applied based on an inconclusive classification.\n\n"
                    "**Recommended Actions:**\n"
                    "• Capture a clearer, well-lit, close-up photograph of a single affected leaflet.\n"
                    "• Ensure the leaf is in focus and fills the frame.\n"
                    "• Verify that the plant is a tomato leaf showing characteristic foliar symptoms.\n"
                    "• If uncertain or symptoms continue spreading, consult a certified local agronomist or extension service."
                )

    else:
        st.info("👆 Please upload a tomato leaf image above to receive a diagnostic classification.")

    # Educational Disclaimer & Limitations Notice
    st.markdown(
        f"""
        <div class="disclaimer-card">
            <strong>⚠️ Agricultural & Technical Disclaimer:</strong><br>
            • <strong>Three-Class Closed Model:</strong> This system is trained exclusively to classify three tomato foliar categories: 
            <em>Early Blight</em>, <em>Late Blight</em>, and <em>Healthy</em>. It has <strong>no validated out-of-distribution detection mechanism</strong> 
            and cannot reliably identify non-tomato leaves, other plant diseases, or arbitrary images.<br>
            • <strong>Uncalibrated Scores:</strong> Model output scores indicate uncalibrated relative outputs among the three supported classes and are not guaranteed real-world probabilities.<br>
            • <strong>Educational Prototype:</strong> High model scores do not guarantee 100% diagnostic accuracy. Always verify symptoms with certified agronomists or local agricultural extension services before taking chemical or cultural action.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
