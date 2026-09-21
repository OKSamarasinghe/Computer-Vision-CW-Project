"""
Streamlit UI for the Diabetic Retinopathy Grading application -- the application-layer innovation
deliverable for Criterion 9 (Innovation, Practical Impact & Critical Discussion).

Run locally with (from the app/ directory):

    streamlit run frontend/streamlit_app.py

Combines:
  1. The exact Notebook 2 preprocessing pipeline (backend/preprocessing.py).
  2. The trained Notebook 5 model, used purely for inference -- NO additional training happens
     anywhere in this file or the modules it imports.
  3. Monte Carlo Dropout uncertainty estimation (backend/inference.py, backend/uncertainty.py) --
     flags low-confidence grades for manual review.
  4. Grad-CAM explanation overlays, ported from Notebook 6 (backend/gradcam.py).
  5. A clinical referral-urgency mapping (backend/clinical_triage.py).
  6. A plain-language explanation panel (backend/explanation.py).
  7. A downloadable one-page PDF screening report (backend/report_generator.py).
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Make the app/ directory importable regardless of the working directory Streamlit was launched
# from, so `import config` and `from backend import ...` resolve whether this file is run via
# `streamlit run frontend/streamlit_app.py` from app/, or from the repository root.
APP_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIRECTORY not in sys.path:
    sys.path.insert(0, APP_DIRECTORY)

import config
from backend import preprocessing, inference, gradcam, uncertainty, clinical_triage, explanation, report_generator


@st.cache_resource
def get_cached_model_and_gradcam_gradient_model():
    """Load the trained model once per Streamlit session -- reloading it on every click would be
    slow and pointless, since the weights never change."""
    trained_model = inference.load_trained_diabetic_retinopathy_model()
    gradcam_gradient_model = gradcam.build_gradcam_gradient_model(trained_model)
    return trained_model, gradcam_gradient_model


def run_full_grading_pipeline(uploaded_image_pil, trained_model, gradcam_gradient_model):
    """Run preprocessing -> MC-Dropout prediction -> Grad-CAM -> triage -> explanation, in order."""
    raw_image_bgr = cv2.cvtColor(np.array(uploaded_image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    preprocessed_image_uint8, intermediate_steps = preprocessing.preprocess_fundus_image_for_inference(
        raw_image_bgr, return_intermediate_steps=True
    )

    mc_dropout_result = inference.run_monte_carlo_dropout_prediction(trained_model, preprocessed_image_uint8)
    confidence_assessment = uncertainty.classify_prediction_confidence_level(mc_dropout_result)

    model_input_batch = np.expand_dims(preprocessed_image_uint8.astype(np.float32), axis=0)
    heatmap = gradcam.generate_gradcam_heatmap(
        gradcam_gradient_model, model_input_batch, target_class_index=mc_dropout_result["predicted_class_index"]
    )
    overlay_image_bgr = gradcam.overlay_gradcam_heatmap_on_image(preprocessed_image_uint8.astype(np.uint8), heatmap)
    heatmap_region_description = gradcam.describe_heatmap_activation_region(heatmap)

    referral_recommendation = clinical_triage.determine_referral_recommendation(
        mc_dropout_result["predicted_class_name"], confidence_assessment["confidence_level"]
    )

    explanation_text = explanation.generate_plain_language_explanation(
        mc_dropout_result["predicted_class_name"],
        mc_dropout_result,
        confidence_assessment,
        heatmap_region_description,
        referral_recommendation,
    )

    return {
        "preprocessed_image_uint8": preprocessed_image_uint8,
        "intermediate_steps": intermediate_steps,
        "mc_dropout_result": mc_dropout_result,
        "confidence_assessment": confidence_assessment,
        "overlay_image_bgr": overlay_image_bgr,
        "heatmap_region_description": heatmap_region_description,
        "referral_recommendation": referral_recommendation,
        "explanation_text": explanation_text,
    }


def render_streamlit_interface():
    st.set_page_config(page_title="DR Grading Assistant", layout="wide")
    st.title("Diabetic Retinopathy Grading Assistant")
    st.caption(
        "Coursework prototype -- uploads a fundus image, grades it with the trained EfficientNetB0 "
        "model, and explains the prediction with confidence-aware, clinically-framed decision "
        "support. Not a certified diagnostic device."
    )

    trained_model, gradcam_gradient_model = get_cached_model_and_gradcam_gradient_model()

    uploaded_file = st.file_uploader("Upload a fundus photograph", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Upload a retinal fundus image to get a grade, an explanation, and a downloadable report.")
        with st.expander("Don't have a sample image handy?"):
            st.write(
                "Drop a few example fundus images (e.g. one per class from your Notebook 2 test "
                "split) into `app/sample_images/` for quick manual testing and for recording the "
                "video demo."
            )
        return

    uploaded_image_pil = Image.open(uploaded_file)
    with st.spinner("Preprocessing, grading, and generating the explanation..."):
        pipeline_result = run_full_grading_pipeline(uploaded_image_pil, trained_model, gradcam_gradient_model)

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(
            cv2.cvtColor(pipeline_result["preprocessed_image_uint8"], cv2.COLOR_BGR2RGB),
            caption="Preprocessed fundus image",
            use_container_width=True,
        )
    with right_column:
        st.image(
            cv2.cvtColor(pipeline_result["overlay_image_bgr"], cv2.COLOR_BGR2RGB),
            caption="Grad-CAM explanation overlay",
            use_container_width=True,
        )

    mc_dropout_result = pipeline_result["mc_dropout_result"]
    confidence_assessment = pipeline_result["confidence_assessment"]
    referral_recommendation = pipeline_result["referral_recommendation"]

    st.subheader(f"Predicted grade: {mc_dropout_result['predicted_class_name'].replace('_', ' ')}")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Confidence", f"{mc_dropout_result['predicted_class_confidence']:.0%}")
    metric_columns[1].metric("Uncertainty (std)", f"{mc_dropout_result['predicted_class_uncertainty_std']:.3f}")
    metric_columns[2].metric("Confidence level", confidence_assessment["confidence_level"])

    urgency_colour_by_level = {
        "Routine": "blue", "Routine referral": "orange", "Urgent": "red", "Emergency": "red",
    }
    st.markdown(
        f"**Recommended action:** "
        f":{urgency_colour_by_level.get(referral_recommendation['urgency_level'], 'grey')}"
        f"[{referral_recommendation['urgency_level']}] -- {referral_recommendation['recommended_action']}"
    )

    st.markdown("### Explanation")
    st.markdown(pipeline_result["explanation_text"])
    st.caption(referral_recommendation["disclaimer"])

    with st.expander("Show per-class probabilities and preprocessing steps"):
        st.write("**Per-class mean probability (+/- std across MC Dropout passes):**")
        for class_name, probability, std in zip(
            config.DIABETIC_RETINOPATHY_CLASS_NAMES,
            mc_dropout_result["mean_class_probabilities"],
            mc_dropout_result["per_class_standard_deviation"],
        ):
            st.write(f"- {class_name}: {probability:.1%} (std {std:.3f})")

        st.write("**Preprocessing pipeline steps:**")
        preprocessing_step_columns = st.columns(len(pipeline_result["intermediate_steps"]))
        for column, (step_name, step_image) in zip(
            preprocessing_step_columns, pipeline_result["intermediate_steps"].items()
        ):
            display_image = step_image if step_image.ndim == 2 else cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
            column.image(display_image, caption=step_name, use_container_width=True)

    original_image_pil = Image.fromarray(
        cv2.cvtColor(pipeline_result["preprocessed_image_uint8"], cv2.COLOR_BGR2RGB)
    )
    overlay_image_pil = Image.fromarray(cv2.cvtColor(pipeline_result["overlay_image_bgr"], cv2.COLOR_BGR2RGB))
    report_pdf_bytes = report_generator.build_screening_report_pdf(
        original_image_pil,
        overlay_image_pil,
        mc_dropout_result,
        confidence_assessment,
        referral_recommendation,
        pipeline_result["explanation_text"],
        patient_reference_label=uploaded_file.name,
    )
    st.download_button(
        "Download screening report (PDF)",
        data=report_pdf_bytes,
        file_name=f"dr_screening_report_{uploaded_file.name}.pdf",
        mime="application/pdf",
    )


if __name__ == "__main__":
    render_streamlit_interface()
