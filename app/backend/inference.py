"""
Model loading and prediction, including Monte Carlo Dropout uncertainty estimation -- the first
pillar of this application's innovation feature.

No additional training happens anywhere in this module. MC Dropout works by keeping the model's
existing Dropout(0.3)/Dropout(0.2) head layers ACTIVE at inference time (instead of their normal
eval-mode no-op behaviour) and running the same preprocessed image through the network multiple
times. Because dropout randomly zeroes different units on each pass, the resulting spread of
predictions is a genuine (if approximate) measure of the model's own uncertainty about that specific
image. This exact technique -- reusing an already-trained model's dropout layers for uncertainty,
with no retraining -- has been applied specifically to diabetic retinopathy grading in recent work;
see docs/innovation_design_notes.md for the full justification and citations.
"""

import numpy as np
import tensorflow as tf

from config import (
    TRAINED_MODEL_FILE_PATH,
    DIABETIC_RETINOPATHY_CLASS_NAMES,
    MONTE_CARLO_DROPOUT_FORWARD_PASSES,
)


def load_trained_diabetic_retinopathy_model(model_file_path=TRAINED_MODEL_FILE_PATH):
    """Load the final trained Keras model produced by Notebook 5.

    This is the exact .keras file saved at the end of Notebook 5
    (diabetic_retinopathy_efficientnetb0_notebook5_final_model.keras) -- copy it into app/models/
    before running this application. No weights are modified, fine-tuned or retrained anywhere in
    this codebase; this function only deserialises the file.
    """
    loaded_model = tf.keras.models.load_model(model_file_path)
    print(
        f"[APP] Loaded trained model from: {model_file_path} "
        f"({loaded_model.count_params():,} total parameters)"
    )
    return loaded_model


def _prepare_single_image_batch(preprocessed_image_uint8):
    """Add the batch dimension EfficientNetB0 expects: shape (1, 224, 224, 3), dtype float32."""
    return np.expand_dims(preprocessed_image_uint8.astype(np.float32), axis=0)


def run_monte_carlo_dropout_prediction(
    trained_model,
    preprocessed_image_uint8,
    number_of_forward_passes=MONTE_CARLO_DROPOUT_FORWARD_PASSES,
):
    """Run repeated stochastic forward passes over one image and summarise the result.

    Returns a dictionary with:
        mean_class_probabilities      -- the averaged softmax output across all passes (the actual
                                          predicted-grade probabilities shown to the user)
        predicted_class_index/name    -- argmax of the mean
        predicted_class_confidence    -- mean probability of the predicted class
        per_class_standard_deviation  -- how much each class's probability varied across passes --
                                          the core uncertainty signal
        predicted_class_uncertainty_std -- that same std, for just the predicted class
        predictive_entropy            -- a single scalar summarising overall uncertainty (higher =
                                          the model is less sure, regardless of which class)
    """
    model_input_batch = _prepare_single_image_batch(preprocessed_image_uint8)

    # training=True keeps Dropout layers stochastic instead of their normal inference-time no-op
    # behaviour -- this single argument is the entire mechanism behind MC Dropout, and requires no
    # change to the model's saved weights or architecture.
    all_pass_probabilities = np.stack(
        [
            trained_model(model_input_batch, training=True).numpy()[0]
            for _ in range(number_of_forward_passes)
        ]
    )

    mean_class_probabilities = all_pass_probabilities.mean(axis=0)
    per_class_standard_deviation = all_pass_probabilities.std(axis=0)
    predicted_class_index = int(np.argmax(mean_class_probabilities))

    predictive_entropy = float(
        -np.sum(mean_class_probabilities * np.log(mean_class_probabilities + 1e-8))
    )

    result = {
        "mean_class_probabilities": mean_class_probabilities,
        "per_class_standard_deviation": per_class_standard_deviation,
        "predicted_class_index": predicted_class_index,
        "predicted_class_name": DIABETIC_RETINOPATHY_CLASS_NAMES[predicted_class_index],
        "predicted_class_confidence": float(mean_class_probabilities[predicted_class_index]),
        "predicted_class_uncertainty_std": float(per_class_standard_deviation[predicted_class_index]),
        "predictive_entropy": predictive_entropy,
        "number_of_forward_passes": number_of_forward_passes,
    }

    print(
        f"[APP] MC Dropout prediction complete ({number_of_forward_passes} passes): "
        f"{result['predicted_class_name']} "
        f"(confidence {result['predicted_class_confidence']:.3f} "
        f"+/- {result['predicted_class_uncertainty_std']:.3f})"
    )

    return result
