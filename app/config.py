"""
Central configuration for the Diabetic Retinopathy Grading application.

Keeping every path, hyperparameter and clinical-mapping constant in one file means the Streamlit UI,
the inference backend and the report generator all stay consistent with each other, and with the exact
settings used during training on Kaggle (Notebooks 2, 4 and 5). Nothing in this file changes model
weights or triggers any training -- it only configures how the already-trained model is loaded and used.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_FILE_PATH = os.path.join(
    APP_ROOT_DIRECTORY, "models", "diabetic_retinopathy_efficientnetb0_notebook5_final_model.keras"
)
SAMPLE_IMAGES_DIRECTORY = os.path.join(APP_ROOT_DIRECTORY, "sample_images")

# ---------------------------------------------------------------------------
# Model / preprocessing constants (must match Notebooks 2, 4 and 5 exactly)
# ---------------------------------------------------------------------------
TARGET_IMAGE_SIZE = (224, 224)                # Notebook 2 Cell 7 / Notebook 4 Cell 3
DARK_BORDER_CROP_THRESHOLD = 7                # Notebook 2 Cell 4
CLAHE_CLIP_LIMIT = 2.0                        # Notebook 2 Cell 5
CLAHE_TILE_GRID_SIZE = (8, 8)                 # Notebook 2 Cell 5
BEN_GRAHAM_BLUR_SIGMA_FRACTION = 10           # Notebook 2 Cell 6
LAST_CONVOLUTIONAL_LAYER_NAME = "top_conv"    # Notebook 6 Cell 13 (EfficientNetB0 backbone)

DIABETIC_RETINOPATHY_CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# ---------------------------------------------------------------------------
# Monte Carlo Dropout uncertainty estimation (app-level innovation feature)
# ---------------------------------------------------------------------------
# See docs/innovation_design_notes.md for the full justification and citations
# (Siebert, Grasshoff and Rostalski, 2023; DRetNet, 2025).
MONTE_CARLO_DROPOUT_FORWARD_PASSES = 30
HIGH_UNCERTAINTY_STANDARD_DEVIATION_THRESHOLD = 0.12  # flag for manual review at/above this std

# ---------------------------------------------------------------------------
# Clinical referral-urgency mapping (app-level innovation feature)
# ---------------------------------------------------------------------------
# Aligned in spirit with NHS England's (2023) diabetic eye screening grading/referral pathway --
# see docs/innovation_design_notes.md for the citation and the full disclaimer. This is a simplified,
# illustrative mapping for a coursework prototype, NOT validated clinical guidance.
REFERRAL_URGENCY_BY_CLASS_NAME = {
    "No_DR": {
        "urgency_level": "Routine",
        "recommended_action": "Continue routine diabetic eye screening at the standard interval.",
    },
    "Mild": {
        "urgency_level": "Routine",
        "recommended_action": "Continue routine screening; rescreen within 12 months.",
    },
    "Moderate": {
        "urgency_level": "Routine referral",
        "recommended_action": "Refer to ophthalmology for routine review (non-urgent).",
    },
    "Severe": {
        "urgency_level": "Urgent",
        "recommended_action": (
            "Refer to ophthalmology urgently (within approximately 2-4 weeks) -- high risk of "
            "progression to proliferative disease."
        ),
    },
    "Proliferative_DR": {
        "urgency_level": "Emergency",
        "recommended_action": (
            "Refer to ophthalmology as an emergency (within approximately 1-2 weeks) -- sight-"
            "threatening; risk of vitreous haemorrhage or retinal detachment."
        ),
    },
}
