"""
Template-based, plain-language explanation generator.

This satisfies the coursework brief's third listed innovation example (a "chatbot/Q&A layer
explaining a prediction in plain language") without calling an external LLM API or training any new
component -- every sentence below is built from values already computed by inference.py,
uncertainty.py, gradcam.py and clinical_triage.py. Keeping the application fully self-contained and
offline-reproducible avoids any dependency that could fail during the live video demo.
"""

DIABETIC_RETINOPATHY_STAGE_PLAIN_LANGUAGE_DESCRIPTIONS = {
    "No_DR": "no visible signs of diabetic retinopathy",
    "Mild": (
        "early, mild changes -- a small number of microaneurysms (tiny bulges in the retina's "
        "blood vessels)"
    ),
    "Moderate": "moderate changes, including more numerous microaneurysms and early blood vessel damage",
    "Severe": "severe changes, with widespread blood vessel damage across multiple regions of the retina",
    "Proliferative_DR": (
        "the most advanced stage, where abnormal new blood vessels have started to grow -- the "
        "stage most associated with vision loss if untreated"
    ),
}


def generate_plain_language_explanation(
    predicted_class_name,
    mc_dropout_result,
    confidence_assessment,
    heatmap_region_description,
    referral_recommendation,
):
    """Compose a short, structured, plain-language paragraph explaining one prediction."""
    stage_description = DIABETIC_RETINOPATHY_STAGE_PLAIN_LANGUAGE_DESCRIPTIONS[predicted_class_name]

    explanation_lines = [
        f"This image was graded as **{predicted_class_name.replace('_', ' ')}**, consistent with "
        f"{stage_description}.",
        f"The model's confidence in this grade is **{confidence_assessment['confidence_level']}** "
        f"({mc_dropout_result['predicted_class_confidence']:.0%} mean probability across "
        f"{mc_dropout_result['number_of_forward_passes']} repeated internal checks). "
        f"{confidence_assessment['reason']}",
        f"The Grad-CAM overlay shows the model's attention was "
        f"**{heatmap_region_description['concentration_description']}**, centred in the "
        f"**{heatmap_region_description['region_description']}**.",
        f"Recommended next step: **{referral_recommendation['urgency_level']}** -- "
        f"{referral_recommendation['recommended_action']}",
    ]

    return "\n\n".join(explanation_lines)
