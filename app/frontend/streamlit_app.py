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

UI/UX layer (this file only -- no backend logic changes):
  - A styled header banner, sidebar-driven upload flow, and card-style result sections.
  - Colour-coded badges for urgency level and confidence level, so the most safety-critical
    information (how urgent, how confident) is visually scannable rather than buried in text.
  - A staged progress indicator (st.status) while the pipeline runs, so the person watching can
    see which stage is executing -- useful both for transparency and for the video demo.
  - A small in-session grading history in the sidebar, since a real screening session would
    involve grading more than one image in a sitting.
"""

import os
import sys
from datetime import datetime

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


# ---------------------------------------------------------------------------
# Visual styling: colours/icons for the badges, and the injected CSS.
# Kept separate from Streamlit's own internal styling (no attempt to override
# Streamlit's internal DOM/class names, which change between versions --
# everything here targets our own custom "dr-*" classes only).
# ---------------------------------------------------------------------------

URGENCY_BADGE_STYLE_BY_LEVEL = {
    "Routine": {"background_colour": "#E3F2E8", "text_colour": "#1B5E3A", "icon": "\u2705"},
    "Routine referral": {"background_colour": "#FFF4E0", "text_colour": "#8A5A00", "icon": "\U0001F7E1"},
    "Urgent": {"background_colour": "#FFE8E0", "text_colour": "#B3401A", "icon": "\u26A0\uFE0F"},
    "Emergency": {"background_colour": "#FCE1E4", "text_colour": "#A3122B", "icon": "\U0001F6A8"},
}

CONFIDENCE_BADGE_STYLE_BY_LEVEL = {
    "High": {"background_colour": "#E3F2E8", "text_colour": "#1B5E3A"},
    "Moderate": {"background_colour": "#FFF4E0", "text_colour": "#8A5A00"},
    "Low": {"background_colour": "#FFE8E0", "text_colour": "#B3401A"},
}


def inject_custom_css():
    """Add a small set of custom CSS classes (header banner, badges, disclaimer box, footer).

    Deliberately does not touch Streamlit's own internal elements/classes -- only styles the
    custom "dr-*" divs and spans this file renders itself, so this won't silently break on a
    Streamlit version upgrade.
    """
    st.markdown(
        """
        <style>
        .dr-header {
            background: linear-gradient(135deg, #0F6FA8 0%, #14919B 100%);
            padding: 28px 32px;
            border-radius: 14px;
            color: #FFFFFF;
            margin-bottom: 24px;
            box-shadow: 0 4px 14px rgba(15, 111, 168, 0.25);
        }
        .dr-header h1 {
            margin: 0 0 6px 0;
            font-size: 1.9rem;
            color: #FFFFFF;
        }
        .dr-header p {
            margin: 0;
            font-size: 0.98rem;
            opacity: 0.92;
        }
        .dr-badge {
            display: inline-block;
            padding: 5px 16px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }
        .dr-disclaimer {
            background-color: #FFF8E1;
            border-left: 4px solid #F0AD4E;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #6B4A00;
            margin-top: 12px;
        }
        .dr-footer {
            text-align: center;
            color: #8A94A6;
            font-size: 0.8rem;
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #E3E8EF;
        }
        .dr-section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_badge_html(label_text, background_colour, text_colour):
    """Build one small coloured pill-shaped badge as raw HTML (used for urgency/confidence)."""
    return (
        f'<span class="dr-badge" style="background-color:{background_colour}; '
        f'color:{text_colour};">{label_text}</span>'
    )


# ---------------------------------------------------------------------------
# Model loading (cached) and the grading pipeline itself -- unchanged in behaviour from the
# original version, only with an optional progress_callback added so the UI can show staged
# progress messages while it runs.
# ---------------------------------------------------------------------------

@st.cache_resource
def get_cached_model_and_gradcam_gradient_model():
    """Load the trained model once per Streamlit session -- reloading it on every click would be
    slow and pointless, since the weights never change."""
    trained_model = inference.load_trained_diabetic_retinopathy_model()
    gradcam_gradient_model = gradcam.build_gradcam_gradient_model(trained_model)
    return trained_model, gradcam_gradient_model


def run_full_grading_pipeline(uploaded_image_pil, trained_model, gradcam_gradient_model, progress_callback=None):
    """Run preprocessing -> MC-Dropout prediction -> Grad-CAM -> triage -> explanation, in order.

    progress_callback, if given, is called with a short human-readable string before each stage,
    purely for UI feedback -- this function has no Streamlit import and stays usable outside a
    Streamlit context (e.g. from a test) if progress_callback is left as None.
    """
    def report_progress(message):
        if progress_callback is not None:
            progress_callback(message)

    report_progress("Preprocessing the fundus image (crop, resize, CLAHE, illumination correction)...")
    raw_image_bgr = cv2.cvtColor(np.array(uploaded_image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    preprocessed_image_uint8, intermediate_steps = preprocessing.preprocess_fundus_image_for_inference(
        raw_image_bgr, return_intermediate_steps=True
    )

    report_progress("Running Monte Carlo Dropout prediction (30 stochastic forward passes)...")
    mc_dropout_result = inference.run_monte_carlo_dropout_prediction(trained_model, preprocessed_image_uint8)
    confidence_assessment = uncertainty.classify_prediction_confidence_level(mc_dropout_result)

    report_progress("Generating the Grad-CAM explanation overlay...")
    model_input_batch = np.expand_dims(preprocessed_image_uint8.astype(np.float32), axis=0)
    heatmap = gradcam.generate_gradcam_heatmap(
        gradcam_gradient_model, model_input_batch, target_class_index=mc_dropout_result["predicted_class_index"]
    )
    overlay_image_bgr = gradcam.overlay_gradcam_heatmap_on_image(preprocessed_image_uint8.astype(np.uint8), heatmap)
    heatmap_region_description = gradcam.describe_heatmap_activation_region(heatmap)

    report_progress("Building the referral recommendation and plain-language explanation...")
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

    report_progress("Done.")
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


def get_or_compute_pipeline_result(uploaded_file, uploaded_image_pil, trained_model, gradcam_gradient_model):
    """Return the cached pipeline result for this exact upload, computing it only once.

    Streamlit re-runs this whole script on every interaction, including clicking the PDF download
    button below. Since Monte Carlo Dropout is genuinely stochastic, re-running the pipeline on
    every interaction would silently give slightly different numbers each time for the SAME
    image. Caching by (filename, file size) keeps the on-screen numbers and the downloaded PDF
    consistent with each other, and only recomputes when a genuinely new image is uploaded.
    """
    current_upload_signature = (uploaded_file.name, uploaded_file.size)

    if (
        "pipeline_result" not in st.session_state
        or st.session_state.get("upload_signature") != current_upload_signature
    ):
        with st.status("Running the grading pipeline...", expanded=True) as status_box:
            def update_status(message):
                status_box.write(message)

            computed_result = run_full_grading_pipeline(
                uploaded_image_pil, trained_model, gradcam_gradient_model, progress_callback=update_status
            )
            status_box.update(label="Grading complete", state="complete", expanded=False)

        st.session_state["pipeline_result"] = computed_result
        st.session_state["upload_signature"] = current_upload_signature

        # Record this grading in the session history, shown in the sidebar. Only added on a
        # genuinely new computation, not on every re-run/cache-hit.
        grading_history = st.session_state.setdefault("grading_history", [])
        grading_history.insert(0, {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "file_name": uploaded_file.name,
            "predicted_grade": computed_result["mc_dropout_result"]["predicted_class_name"].replace("_", " "),
            "confidence_level": computed_result["confidence_assessment"]["confidence_level"],
        })

    return st.session_state["pipeline_result"]


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def render_header():
    st.markdown(
        """
        <div class="dr-header">
            <h1>\U0001FA7A Diabetic Retinopathy Grading Assistant</h1>
            <p>Coursework prototype &mdash; uploads a fundus image, grades it with the trained
            EfficientNetB0 model, and explains the prediction with confidence-aware, clinically-framed
            decision support. Not a certified diagnostic device.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar (upload control, how-it-works, session history, short disclaimer) and
    return the uploaded file, if any."""
    with st.sidebar:
        st.markdown("### \U0001F4E4 Upload")
        uploaded_file = st.file_uploader("Upload a fundus photograph", type=["png", "jpg", "jpeg"])

        st.markdown("---")
        with st.expander("\u2139\ufe0f How this works"):
            st.write(
                "1. Your image is preprocessed with the exact pipeline from Notebook 2.\n"
                "2. The trained EfficientNetB0 model grades it 30 times with Monte Carlo Dropout "
                "active, giving both a grade and an uncertainty estimate.\n"
                "3. A Grad-CAM overlay shows which region of the image drove the prediction.\n"
                "4. The grade and confidence are mapped onto a referral-urgency recommendation.\n"
                "5. Everything is bundled into a downloadable PDF screening report."
            )

        grading_history = st.session_state.get("grading_history", [])
        if grading_history:
            st.markdown("---")
            st.markdown("### \U0001F553 Session history")
            for history_entry in grading_history[:8]:
                st.caption(
                    f"**{history_entry['timestamp']}** &mdash; {history_entry['file_name']}: "
                    f"{history_entry['predicted_grade']} ({history_entry['confidence_level']} confidence)"
                )

        st.markdown("---")
        st.caption(
            "This is a screening-support coursework prototype, not a certified diagnostic device. "
            "Every result must be reviewed by a qualified ophthalmologist or optometrist."
        )

    return uploaded_file


def render_images_section(pipeline_result):
    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Fundus image and explanation overlay</div>', unsafe_allow_html=True)
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


def render_result_summary_section(pipeline_result):
    mc_dropout_result = pipeline_result["mc_dropout_result"]
    confidence_assessment = pipeline_result["confidence_assessment"]
    referral_recommendation = pipeline_result["referral_recommendation"]

    with st.container(border=True):
        st.markdown(
            f'<div class="dr-section-title">Predicted grade: '
            f'{mc_dropout_result["predicted_class_name"].replace("_", " ")}</div>',
            unsafe_allow_html=True,
        )

        metric_columns = st.columns(3)
        metric_columns[0].metric("Confidence", f"{mc_dropout_result['predicted_class_confidence']:.0%}")
        metric_columns[1].metric("Uncertainty (std)", f"{mc_dropout_result['predicted_class_uncertainty_std']:.3f}")
        metric_columns[2].metric("Confidence level", confidence_assessment["confidence_level"])

        urgency_style = URGENCY_BADGE_STYLE_BY_LEVEL.get(
            referral_recommendation["urgency_level"],
            {"background_colour": "#EEF1F5", "text_colour": "#333333", "icon": ""},
        )
        urgency_badge_html = render_badge_html(
            f'{urgency_style["icon"]} {referral_recommendation["urgency_level"]}',
            urgency_style["background_colour"],
            urgency_style["text_colour"],
        )
        st.markdown(
            f'<div style="margin-top:10px;">{urgency_badge_html} '
            f'<span style="margin-left:8px;">{referral_recommendation["recommended_action"]}</span></div>',
            unsafe_allow_html=True,
        )


def render_explanation_section(pipeline_result):
    referral_recommendation = pipeline_result["referral_recommendation"]
    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Explanation</div>', unsafe_allow_html=True)
        st.markdown(pipeline_result["explanation_text"])
        st.markdown(
            f'<div class="dr-disclaimer">{referral_recommendation["disclaimer"]}</div>',
            unsafe_allow_html=True,
        )


def render_details_expander(pipeline_result):
    mc_dropout_result = pipeline_result["mc_dropout_result"]

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


def render_download_section(pipeline_result, uploaded_file):
    mc_dropout_result = pipeline_result["mc_dropout_result"]
    confidence_assessment = pipeline_result["confidence_assessment"]
    referral_recommendation = pipeline_result["referral_recommendation"]

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

    with st.container(border=True):
        st.markdown('<div class="dr-section-title">\U0001F4C4 Screening report</div>', unsafe_allow_html=True)
        st.write("Download a one-page PDF bundling the image, the Grad-CAM overlay, the grade, and the recommendation.")
        st.download_button(
            "Download screening report (PDF)",
            data=report_pdf_bytes,
            file_name=f"dr_screening_report_{uploaded_file.name}.pdf",
            mime="application/pdf",
        )


def render_footer():
    st.markdown(
        '<div class="dr-footer">Diabetic Retinopathy Grading Assistant &middot; Computer Vision '
        "Coursework Prototype &middot; Not a certified diagnostic device</div>",
        unsafe_allow_html=True,
    )


def render_streamlit_interface():
    st.set_page_config(page_title="DR Grading Assistant", layout="wide", page_icon="\U0001FA7A")
    inject_custom_css()
    render_header()

    trained_model, gradcam_gradient_model = get_cached_model_and_gradcam_gradient_model()
    uploaded_file = render_sidebar()

    if uploaded_file is None:
        st.info("Upload a retinal fundus image from the sidebar to get a grade, an explanation, and a downloadable report.")
        render_footer()
        return

    uploaded_image_pil = Image.open(uploaded_file)
    pipeline_result = get_or_compute_pipeline_result(
        uploaded_file, uploaded_image_pil, trained_model, gradcam_gradient_model
    )

    render_images_section(pipeline_result)
    render_result_summary_section(pipeline_result)
    render_explanation_section(pipeline_result)
    render_details_expander(pipeline_result)
    render_download_section(pipeline_result, uploaded_file)
    render_footer()


if __name__ == "__main__":
    render_streamlit_interface()