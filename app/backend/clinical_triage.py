"""
Maps a predicted DR grade, combined with the confidence flag from uncertainty.py, onto a
referral-urgency recommendation. This is the "practical impact" half of the application's
innovation feature: turning a bare class label into decision-support output closer to what a real
screening pipeline would need to produce.

IMPORTANT: REFERRAL_URGENCY_BY_CLASS_NAME in config.py is a simplified, illustrative mapping
inspired by NHS England's (2023) diabetic eye screening grading/referral pathway (see
docs/innovation_design_notes.md for the citation). It is NOT validated clinical guidance, and every
recommendation produced here carries an explicit disclaimer -- this tool is a screening-support
coursework prototype, not a diagnostic device.
"""

from config import REFERRAL_URGENCY_BY_CLASS_NAME


CLINICAL_SAFETY_DISCLAIMER = (
    "This is an automated screening-support prototype built for a computer vision coursework "
    "project. It is NOT a certified diagnostic device and must not be used to make real clinical "
    "decisions. Every graded image should be reviewed by a qualified ophthalmologist or optometrist."
)


def determine_referral_recommendation(predicted_class_name, confidence_level):
    """Combine the predicted grade with the confidence flag into one recommendation.

    A Low-confidence prediction always escalates the recommended action by at least one step (e.g.
    a Low-confidence 'Moderate' reading is treated as at least a routine referral, never silently
    left at 'continue routine screening') -- reflecting the principle that an uncertain automated
    grade should never result in LESS clinical attention than a confident one of the same predicted
    class.
    """
    base_recommendation = REFERRAL_URGENCY_BY_CLASS_NAME[predicted_class_name]

    escalation_note = ""
    if confidence_level == "Low" and base_recommendation["urgency_level"] == "Routine":
        escalation_note = (
            " Because this prediction has LOW confidence, treat it as at least a routine referral "
            "rather than routine screening alone, pending specialist review."
        )

    return {
        "predicted_class_name": predicted_class_name,
        "confidence_level": confidence_level,
        "urgency_level": base_recommendation["urgency_level"],
        "recommended_action": base_recommendation["recommended_action"] + escalation_note,
        "disclaimer": CLINICAL_SAFETY_DISCLAIMER,
    }
