"""
Interprets the raw Monte Carlo Dropout statistics produced by
inference.run_monte_carlo_dropout_prediction() into a simple, three-level confidence flag a
non-technical user (or a busy clinician triaging many images) can act on at a glance. Pure
post-processing of numbers already computed -- no model changes.
"""

from config import HIGH_UNCERTAINTY_STANDARD_DEVIATION_THRESHOLD


def classify_prediction_confidence_level(mc_dropout_result):
    """Return one of 'High', 'Moderate', 'Low' plus a short plain-English reason.

    The threshold in config.py was chosen conservatively so that this flag errs on the side of
    asking for manual review -- appropriate for a screening-support tool, where a falsely
    "confident" reading on a genuinely ambiguous image (Notebook 6 found real examples of exactly
    this, e.g. around the Moderate/Severe boundary) is the costlier mistake to make silently.
    """
    predicted_class_uncertainty_std = mc_dropout_result["predicted_class_uncertainty_std"]
    predicted_class_confidence = mc_dropout_result["predicted_class_confidence"]

    if predicted_class_uncertainty_std >= HIGH_UNCERTAINTY_STANDARD_DEVIATION_THRESHOLD:
        confidence_level = "Low"
        reason = (
            f"The model's prediction varied substantially across "
            f"{mc_dropout_result['number_of_forward_passes']} repeated stochastic passes "
            f"(std={predicted_class_uncertainty_std:.3f}) -- treat this grade as provisional and "
            f"prioritise it for specialist review."
        )
    elif predicted_class_confidence < 0.5:
        confidence_level = "Moderate"
        reason = (
            f"The top predicted class only reached {predicted_class_confidence:.0%} mean "
            f"probability, meaning the model saw meaningful evidence for more than one grade."
        )
    else:
        confidence_level = "High"
        reason = (
            f"The prediction was stable across repeated stochastic passes "
            f"(std={predicted_class_uncertainty_std:.3f}) with {predicted_class_confidence:.0%} "
            f"mean confidence."
        )

    return {"confidence_level": confidence_level, "reason": reason}
