"""
Model performance facts, for the application's "Model Performance" page.

Every number in this module is a REAL measured figure from this project's own Kaggle notebooks --
nothing here is estimated, rounded for presentation, or invented:

  - Test-set metrics (accuracy, QWK, per-class precision/recall/F1/support) come from Notebook 6
    Cell 8, computed on the 550-image held-out test split that was untouched until that point.
  - The error-structure breakdown (adjacent-stage vs. far-miss) comes from Notebook 6 Cell 11.
  - Validation accuracy and the training configuration come from Notebook 5's metadata.
  - Dataset/class-distribution counts come from Notebook 1, and the augmentation counts from
    Notebook 3.

Showing honest performance data -- including the weak minority-class recall this project genuinely
has -- is a deliberate design decision, not an oversight. A screening-support tool that hides how
often it is wrong on the sight-threatening classes would be actively misleading to the clinician
using it. See docs/innovation_design_notes.md.

Where the corresponding artefact file (a CSV or figure saved by a notebook) is present under
kaggle_artifacts/, the loader functions below prefer reading it directly, so the page stays in sync
if a notebook is ever re-run with different results. The constants act as the documented fallback.
"""

import os

import pandas as pd

from config import APP_ROOT_DIRECTORY


# Repository root is one level above app/, so kaggle_artifacts/ is a sibling of app/.
REPOSITORY_ROOT_DIRECTORY = os.path.dirname(APP_ROOT_DIRECTORY)
KAGGLE_ARTIFACTS_DIRECTORY = os.path.join(REPOSITORY_ROOT_DIRECTORY, "kaggle_artifacts")

NOTEBOOK_6_ARTIFACTS_DIRECTORY = os.path.join(KAGGLE_ARTIFACTS_DIRECTORY, "notebook_6_evaluation_gradcam")
NOTEBOOK_5_ARTIFACTS_DIRECTORY = os.path.join(KAGGLE_ARTIFACTS_DIRECTORY, "notebook_5_training_strategy")
NOTEBOOK_3_ARTIFACTS_DIRECTORY = os.path.join(KAGGLE_ARTIFACTS_DIRECTORY, "notebook_3_augmentation")
NOTEBOOK_1_ARTIFACTS_DIRECTORY = os.path.join(KAGGLE_ARTIFACTS_DIRECTORY, "notebook_1_eda")


# ---------------------------------------------------------------------------
# Headline test-set metrics (Notebook 6, Cell 8)
# ---------------------------------------------------------------------------

OVERALL_TEST_ACCURACY = 0.7582
QUADRATIC_WEIGHTED_KAPPA = 0.8256
NOTEBOOK_5_BEST_VALIDATION_ACCURACY = 0.7927
VALIDATION_TO_TEST_ACCURACY_GAP = OVERALL_TEST_ACCURACY - NOTEBOOK_5_BEST_VALIDATION_ACCURACY
TEST_SET_IMAGE_COUNT = 550

PER_CLASS_TEST_METRICS = [
    {"class_name": "No_DR", "precision": 0.9457, "recall": 0.9631, "f1_score": 0.9543, "support": 271},
    {"class_name": "Mild", "precision": 0.4186, "recall": 0.3214, "f1_score": 0.3636, "support": 56},
    {"class_name": "Moderate", "precision": 0.6571, "recall": 0.7667, "f1_score": 0.7077, "support": 150},
    {"class_name": "Severe", "precision": 0.3333, "recall": 0.3793, "f1_score": 0.3548, "support": 29},
    {"class_name": "Proliferative_DR", "precision": 0.5217, "recall": 0.2727, "f1_score": 0.3582, "support": 44},
]

MACRO_AVERAGE_TEST_METRICS = {"precision": 0.58, "recall": 0.54, "f1_score": 0.55}
WEIGHTED_AVERAGE_TEST_METRICS = {"precision": 0.75, "recall": 0.76, "f1_score": 0.75}


# ---------------------------------------------------------------------------
# Error structure (Notebook 6, Cell 11)
# ---------------------------------------------------------------------------

TOTAL_MISCLASSIFICATION_COUNT = 133
ADJACENT_STAGE_ERROR_COUNT = 95          # stage distance = 1, clinically minor
FAR_MISS_ERROR_COUNT = 38                # stage distance >= 2, clinically serious
ADJACENT_STAGE_ERROR_SHARE = 0.714
FAR_MISS_ERROR_SHARE = 0.286


# ---------------------------------------------------------------------------
# Dataset composition (Notebook 1) and augmentation outcome (Notebook 3)
# ---------------------------------------------------------------------------

ORIGINAL_CLASS_IMAGE_COUNTS = {
    "No_DR": 1805,
    "Mild": 370,
    "Moderate": 999,
    "Severe": 193,
    "Proliferative_DR": 295,
}
ORIGINAL_TOTAL_IMAGE_COUNT = 3662
ORIGINAL_CLASS_IMBALANCE_RATIO = 9.35
REBALANCED_CLASS_IMBALANCE_RATIO = 2.34
AUGMENTED_TRAINING_SET_SIZE = 4926
SYNTHETIC_TRAINING_IMAGE_SHARE = 0.48     # explicitly declared, per the coursework brief


# ---------------------------------------------------------------------------
# Model / training configuration (Notebooks 4 and 5)
# ---------------------------------------------------------------------------

MODEL_CONFIGURATION_FACTS = [
    ("Backbone architecture", "EfficientNetB0 (ImageNet pre-trained)"),
    ("Transfer learning strategy", "Two-stage: frozen head-only training, then fine-tuning the last ~20 layers"),
    ("Selected configuration", "Stage 2, Configuration B (3-epoch learning-rate warm-up)"),
    ("Input resolution", "224 x 224, raw [0, 255] pixel range"),
    ("Classification head", "GlobalAveragePooling2D -> Dropout(0.3) -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(5, softmax)"),
    ("Total parameters", "4,214,184"),
    ("Total training epochs", "18 (8 Stage 1 + 10 Stage 2)"),
    ("Overfitting prevention", "EarlyStopping, ReduceLROnPlateau, L2 regularisation, dropout"),
    ("Class imbalance handling", "Capped per-class augmentation + computed class weights"),
]


# ---------------------------------------------------------------------------
# Known limitations -- surfaced in the app deliberately, not hidden
# ---------------------------------------------------------------------------

DOCUMENTED_MODEL_LIMITATIONS = [
    "Minority-class recall is weak: Mild (0.32), Severe (0.38) and Proliferative_DR (0.27). The model "
    "misses roughly two thirds to three quarters of these cases -- precisely the sight-threatening "
    "stages where a miss matters most. This is the project's clearest limitation.",

    "Minority-class figures are estimated from very few test images (29-56 each), so they carry wide "
    "uncertainty and should not be read as precise.",

    "Grad-CAM overlays are sometimes confounded by the optic disc's high visual salience -- a known "
    "limitation of gradient-based saliency methods on fundus imagery, not a pipeline defect.",

    "Test accuracy (0.7582) sits 3.45 points below validation accuracy (0.7927). With n=550 the standard "
    "error is about +/-1.8 points, so this is a real but modest gap, most plausibly explained by "
    "validation-driven model selection across Notebooks 4-5.",

    "The referral-urgency mapping in this application is a simplified, illustrative prototype mapping "
    "inspired by NHS England (2023) guidance -- it is NOT validated clinical guidance.",

    "This is a coursework prototype, not a certified diagnostic device. Every graded image requires "
    "review by a qualified ophthalmologist or optometrist.",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_per_class_metrics_dataframe():
    """Return the per-class metrics as a DataFrame.

    Prefers the real CSV saved by Notebook 6 if it is present in kaggle_artifacts/, so the page
    reflects an actual re-run rather than a stale constant; otherwise falls back to the documented
    constants above (which are themselves the figures that CSV contained).
    """
    metrics_csv_path = os.path.join(NOTEBOOK_6_ARTIFACTS_DIRECTORY, "notebook_6_test_set_metrics_table.csv")

    if os.path.exists(metrics_csv_path):
        try:
            loaded_dataframe = pd.read_csv(metrics_csv_path)
            if {"class_name", "precision", "recall", "f1_score"}.issubset(loaded_dataframe.columns):
                return loaded_dataframe, "kaggle_artifacts CSV (Notebook 6)"
        except Exception:
            # A malformed/unreadable CSV should never break the page -- fall through to constants.
            pass

    return pd.DataFrame(PER_CLASS_TEST_METRICS), "documented Notebook 6 figures"


def find_artifact_figure_path(artifacts_directory, figure_file_name):
    """Return the full path to a saved notebook figure if it exists, else None.

    Used so the performance page can display the project's real confusion matrix / training curve
    images when the artefacts have been copied into the repository, and simply omit them (with a
    short note) when they have not.
    """
    candidate_path = os.path.join(artifacts_directory, figure_file_name)
    return candidate_path if os.path.exists(candidate_path) else None


def get_class_distribution_dataframe():
    """Return the original (pre-augmentation) class distribution as a DataFrame, for charting."""
    return pd.DataFrame(
        {
            "class_name": list(ORIGINAL_CLASS_IMAGE_COUNTS.keys()),
            "image_count": list(ORIGINAL_CLASS_IMAGE_COUNTS.values()),
        }
    )
