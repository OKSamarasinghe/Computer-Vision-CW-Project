"""
Streamlit UI for the Diabetic Retinopathy Grading Assistant -- the application-layer innovation
deliverable for Criterion 9 (Innovation, Practical Impact & Critical Discussion).

Run locally with (from the app/ directory):

    streamlit run frontend/streamlit_app.py

BACKEND (unchanged by this file -- no model is trained, fine-tuned, or modified anywhere):
  1. The exact Notebook 2 preprocessing pipeline (backend/preprocessing.py).
  2. The trained Notebook 5 model, used purely for inference (backend/inference.py).
  3. Monte Carlo Dropout uncertainty estimation (backend/inference.py, backend/uncertainty.py).
  4. Grad-CAM explanation overlays, ported from Notebook 6 (backend/gradcam.py).
  5. A clinical referral-urgency mapping (backend/clinical_triage.py).
  6. A plain-language explanation generator (backend/explanation.py).
  7. A downloadable PDF screening report (backend/report_generator.py).
  8. Real measured model performance figures (backend/model_performance.py).

UI STRUCTURE -- five pages, selected from the sidebar navigation:
  - Dashboard        : at-a-glance model status, session activity, and how the tool works.
  - New Scan         : upload and grade a fundus image (the core workflow).
  - Scan History     : every image graded this session, reviewable and re-downloadable.
  - Model Performance: the project's real, measured evaluation results -- including its genuine
                       weaknesses, shown deliberately rather than hidden.
  - About            : purpose, pipeline, methodology, limitations, and citations.

The sidebar deliberately holds ONLY navigation and a standing safety disclaimer. Upload controls,
explanatory text and history each live on the page they belong to, so no page mixes "where am I
going" with "what am I doing here".
"""

import os
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Make the app/ directory importable regardless of the working directory Streamlit was launched
# from, so `import config` and `from backend import ...` resolve whether this file is run via
# `streamlit run frontend/streamlit_app.py` from app/, or from the repository root.
APP_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIRECTORY not in sys.path:
    sys.path.insert(0, APP_DIRECTORY)

import config
from backend import (
    preprocessing,
    inference,
    gradcam,
    uncertainty,
    clinical_triage,
    explanation,
    report_generator,
    model_performance,
)


# ---------------------------------------------------------------------------
# Design tokens: one place for every colour/icon used by the custom components.
# Everything here styles only our own "dr-*" CSS classes -- Streamlit's internal
# DOM/class names are never targeted, since those change between versions.
# ---------------------------------------------------------------------------

BRAND_PRIMARY_COLOUR = "#0F6FA8"
BRAND_SECONDARY_COLOUR = "#14919B"

URGENCY_BADGE_STYLE_BY_LEVEL = {
    "Routine": {"background_colour": "#E3F2E8", "text_colour": "#1B5E3A", "icon": "\u2705"},
    "Routine referral": {"background_colour": "#FFF4E0", "text_colour": "#8A5A00", "icon": "\U0001F7E1"},
    "Urgent": {"background_colour": "#FFE8E0", "text_colour": "#B3401A", "icon": "\u26A0\uFE0F"},
    "Emergency": {"background_colour": "#FCE1E4", "text_colour": "#A3122B", "icon": "\U0001F6A8"},
}

CONFIDENCE_BADGE_STYLE_BY_LEVEL = {
    "High": {"background_colour": "#E3F2E8", "text_colour": "#1B5E3A", "icon": "\u25CF"},
    "Moderate": {"background_colour": "#FFF4E0", "text_colour": "#8A5A00", "icon": "\u25D1"},
    "Low": {"background_colour": "#FFE8E0", "text_colour": "#B3401A", "icon": "\u25CB"},
}

# Severity ordering drives the colour ramp on the per-class probability bars, so the visual
# weight of a prediction rises with clinical seriousness rather than being arbitrary.
CLASS_SEVERITY_COLOUR = {
    "No_DR": "#2E9E5B",
    "Mild": "#8CC63F",
    "Moderate": "#F0AD4E",
    "Severe": "#E8743B",
    "Proliferative_DR": "#C4314B",
}

PAGE_NAMES = ["Dashboard", "New Scan", "Scan History", "Model Performance", "About"]
PAGE_ICONS = {
    "Dashboard": "\U0001F4CA",
    "New Scan": "\U0001F50D",
    "Scan History": "\U0001F553",
    "Model Performance": "\U0001F4C8",
    "About": "\u2139\uFE0F",
}


def inject_custom_css():
    """Inject the application's custom CSS (banner, cards, badges, callouts, footer)."""
    st.markdown(
        f"""
        <style>
        .dr-header {{
            background: linear-gradient(135deg, {BRAND_PRIMARY_COLOUR} 0%, {BRAND_SECONDARY_COLOUR} 100%);
            padding: 26px 32px;
            border-radius: 14px;
            color: #FFFFFF;
            margin-bottom: 22px;
            box-shadow: 0 4px 14px rgba(15, 111, 168, 0.25);
        }}
        .dr-header h1 {{
            margin: 0 0 6px 0;
            font-size: 1.75rem;
            color: #FFFFFF;
        }}
        .dr-header p {{
            margin: 0;
            font-size: 0.95rem;
            opacity: 0.92;
        }}
        .dr-badge {{
            display: inline-block;
            padding: 5px 16px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }}
        .dr-disclaimer {{
            background-color: #FFF8E1;
            border-left: 4px solid #F0AD4E;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #6B4A00;
            margin-top: 12px;
        }}
        .dr-note {{
            background-color: #EAF4FB;
            border-left: 4px solid {BRAND_PRIMARY_COLOUR};
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.87rem;
            margin: 10px 0;
        }}
        .dr-limitation {{
            background-color: #FDF0F2;
            border-left: 4px solid #C4314B;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            margin-bottom: 8px;
        }}
        .dr-section-title {{
            font-size: 1.12rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .dr-stat-card {{
            background: linear-gradient(135deg, #F7FAFC 0%, #EDF3F8 100%);
            border: 1px solid #DCE6EE;
            border-radius: 12px;
            padding: 16px 18px;
            text-align: center;
        }}
        .dr-stat-value {{
            font-size: 1.7rem;
            font-weight: 700;
            color: {BRAND_PRIMARY_COLOUR};
            line-height: 1.1;
        }}
        .dr-stat-label {{
            font-size: 0.78rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }}
        .dr-footer {{
            text-align: center;
            color: #8A94A6;
            font-size: 0.78rem;
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #E3E8EF;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_badge_html(label_text, background_colour, text_colour):
    """Build one pill-shaped coloured badge as raw HTML."""
    return (
        f'<span class="dr-badge" style="background-color:{background_colour}; '
        f'color:{text_colour};">{label_text}</span>'
    )


def render_stat_card_html(value_text, label_text):
    """Build one small statistic card (big number over a small uppercase label)."""
    return (
        f'<div class="dr-stat-card"><div class="dr-stat-value">{value_text}</div>'
        f'<div class="dr-stat-label">{label_text}</div></div>'
    )


# ---------------------------------------------------------------------------
# Model loading and the grading pipeline
# ---------------------------------------------------------------------------

@st.cache_resource
def get_cached_model_and_gradcam_gradient_model():
    """Load the trained model once per Streamlit session -- reloading it on every interaction
    would be slow and pointless, since the weights never change."""
    trained_model = inference.load_trained_diabetic_retinopathy_model()
    gradcam_gradient_model = gradcam.build_gradcam_gradient_model(trained_model)
    return trained_model, gradcam_gradient_model


def run_full_grading_pipeline(uploaded_image_pil, trained_model, gradcam_gradient_model, progress_callback=None):
    """Run preprocessing -> MC-Dropout prediction -> Grad-CAM -> triage -> explanation, in order.

    progress_callback, if given, is called with a short human-readable string before each stage,
    purely for UI feedback. This function imports nothing from Streamlit itself, so it stays usable
    outside a Streamlit context (e.g. from a test) when progress_callback is left as None.
    """
    def report_progress(message):
        if progress_callback is not None:
            progress_callback(message)

    report_progress("Preprocessing the fundus image (crop, resize, CLAHE, illumination correction)...")
    raw_image_bgr = cv2.cvtColor(np.array(uploaded_image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    preprocessed_image_uint8, intermediate_steps = preprocessing.preprocess_fundus_image_for_inference(
        raw_image_bgr, return_intermediate_steps=True
    )

    report_progress(
        f"Running Monte Carlo Dropout prediction "
        f"({config.MONTE_CARLO_DROPOUT_FORWARD_PASSES} stochastic forward passes)..."
    )
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
    """Return the pipeline result for this exact upload, computing it only once.

    Streamlit re-runs the whole script on every interaction, including clicking a download button
    or switching page. Since Monte Carlo Dropout is genuinely stochastic, recomputing on every
    interaction would silently give slightly different numbers each time for the SAME image --
    meaning the figure on screen and the figure in the downloaded PDF could disagree. Caching by
    (filename, file size) keeps them identical, and recomputes only for a genuinely new image.
    """
    current_upload_signature = (uploaded_file.name, uploaded_file.size)

    if (
        "current_pipeline_result" not in st.session_state
        or st.session_state.get("current_upload_signature") != current_upload_signature
    ):
        with st.status("Running the grading pipeline...", expanded=True) as status_box:
            def update_status(message):
                status_box.write(message)

            computed_result = run_full_grading_pipeline(
                uploaded_image_pil, trained_model, gradcam_gradient_model, progress_callback=update_status
            )
            status_box.update(label="Grading complete", state="complete", expanded=False)

        st.session_state["current_pipeline_result"] = computed_result
        st.session_state["current_upload_signature"] = current_upload_signature

        # Record the full result in the session history so the Scan History page can redisplay it
        # (including re-generating its PDF) without recomputing and getting different numbers.
        scan_history = st.session_state.setdefault("scan_history", [])
        scan_history.insert(0, {
            "scan_timestamp": datetime.now(),
            "file_name": uploaded_file.name,
            "pipeline_result": computed_result,
        })

    return st.session_state["current_pipeline_result"]


# ---------------------------------------------------------------------------
# Shared rendering components
# ---------------------------------------------------------------------------

def render_page_header(title_text, subtitle_text):
    st.markdown(
        f'<div class="dr-header"><h1>{title_text}</h1><p>{subtitle_text}</p></div>',
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<div class="dr-footer">Diabetic Retinopathy Grading Assistant &middot; Computer Vision '
        "Coursework Prototype &middot; Not a certified diagnostic device</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_navigation():
    """Sidebar holds navigation and the standing safety disclaimer -- nothing else.

    Upload controls, explanations and history belong on their own pages, so the sidebar stays a
    pure "where am I going" control rather than mixing navigation with task content.
    """
    with st.sidebar:
        st.markdown(
            f'<div style="padding:14px 0 10px 0;">'
            f'<div style="font-size:1.05rem;font-weight:700;color:{BRAND_PRIMARY_COLOUR};">'
            f"\U0001FA7A DR Grading Assistant</div>"
            f'<div style="font-size:0.75rem;color:#64748B;">EfficientNetB0 &middot; APTOS 2019</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        selected_page_name = st.radio(
            "Navigation",
            PAGE_NAMES,
            format_func=lambda page_name: f"{PAGE_ICONS[page_name]}  {page_name}",
            label_visibility="collapsed",
        )

        st.markdown("---")
        scan_history = st.session_state.get("scan_history", [])
        st.caption(f"Scans this session: **{len(scan_history)}**")

        st.markdown("---")
        st.caption(
            "\u26A0\uFE0F Screening-support coursework prototype. Not a certified diagnostic device. "
            "Every result requires review by a qualified ophthalmologist or optometrist."
        )

    return selected_page_name


def render_per_class_probability_chart(mc_dropout_result):
    """Show per-class mean probability as a horizontal bar chart, coloured by clinical severity.

    A chart rather than a list, because the shape of the distribution (one confident peak vs. two
    classes competing) is the thing a clinician actually needs to see at a glance -- and that shape
    is exactly what a column of percentages hides.
    """
    probability_dataframe = pd.DataFrame({
        "Class": config.DIABETIC_RETINOPATHY_CLASS_NAMES,
        "Probability": [float(value) for value in mc_dropout_result["mean_class_probabilities"]],
        "Uncertainty (std)": [float(value) for value in mc_dropout_result["per_class_standard_deviation"]],
    })
    probability_dataframe["Probability %"] = (probability_dataframe["Probability"] * 100).round(1)

    st.bar_chart(
        probability_dataframe.set_index("Class")["Probability %"],
        horizontal=True,
        color=BRAND_PRIMARY_COLOUR,
        height=220,
    )
    st.dataframe(
        probability_dataframe[["Class", "Probability %", "Uncertainty (std)"]],
        hide_index=True,
        use_container_width=True,
    )


def render_result_detail(pipeline_result, uploaded_file_name, show_preprocessing_steps=True):
    """Render one complete grading result: images, grade, badges, explanation, details, PDF.

    Shared between the New Scan page and the Scan History page so a past scan is presented exactly
    the same way as a fresh one -- and, because the stored result is reused rather than recomputed,
    with exactly the same numbers it originally produced.
    """
    mc_dropout_result = pipeline_result["mc_dropout_result"]
    confidence_assessment = pipeline_result["confidence_assessment"]
    referral_recommendation = pipeline_result["referral_recommendation"]

    # --- Images ---
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
                caption="Grad-CAM explanation overlay (warmer = more influential)",
                use_container_width=True,
            )

    # --- Headline result ---
    with st.container(border=True):
        urgency_style = URGENCY_BADGE_STYLE_BY_LEVEL.get(
            referral_recommendation["urgency_level"],
            {"background_colour": "#EEF1F5", "text_colour": "#333333", "icon": ""},
        )
        confidence_style = CONFIDENCE_BADGE_STYLE_BY_LEVEL.get(
            confidence_assessment["confidence_level"],
            {"background_colour": "#EEF1F5", "text_colour": "#333333", "icon": ""},
        )

        predicted_grade_display_name = mc_dropout_result["predicted_class_name"].replace("_", " ")
        st.markdown(
            f'<div class="dr-section-title">Predicted grade: {predicted_grade_display_name}</div>',
            unsafe_allow_html=True,
        )

        # Build each badge's HTML separately first, purely for readability -- nesting these calls
        # inside the markdown f-string below would make it very hard to follow.
        urgency_badge_label = f"{urgency_style['icon']} {referral_recommendation['urgency_level']}"
        urgency_badge_html = render_badge_html(
            urgency_badge_label, urgency_style["background_colour"], urgency_style["text_colour"]
        )
        confidence_badge_label = f"{confidence_style['icon']} {confidence_assessment['confidence_level']} confidence"
        confidence_badge_html = render_badge_html(
            confidence_badge_label, confidence_style["background_colour"], confidence_style["text_colour"]
        )

        st.markdown(
            f'<div style="margin-bottom:12px;">{urgency_badge_html}&nbsp;&nbsp;{confidence_badge_html}</div>',
            unsafe_allow_html=True,
        )

        metric_columns = st.columns(3)
        metric_columns[0].metric("Confidence", f"{mc_dropout_result['predicted_class_confidence']:.0%}")
        metric_columns[1].metric("Uncertainty (std)", f"{mc_dropout_result['predicted_class_uncertainty_std']:.3f}")
        metric_columns[2].metric("MC Dropout passes", f"{mc_dropout_result['number_of_forward_passes']}")

        st.markdown(f"**Recommended action:** {referral_recommendation['recommended_action']}")

    # --- Probability distribution ---
    with st.container(border=True):
        st.markdown(
            '<div class="dr-section-title">Per-class probability distribution</div>', unsafe_allow_html=True
        )
        st.caption(
            "Mean probability across all Monte Carlo Dropout passes, with the standard deviation "
            "showing how much each class's probability varied between passes."
        )
        render_per_class_probability_chart(mc_dropout_result)

    # --- Explanation ---
    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Explanation</div>', unsafe_allow_html=True)
        st.markdown(pipeline_result["explanation_text"])
        st.markdown(
            f'<div class="dr-disclaimer">{referral_recommendation["disclaimer"]}</div>',
            unsafe_allow_html=True,
        )

    # --- Preprocessing steps ---
    if show_preprocessing_steps and "intermediate_steps" in pipeline_result:
        with st.expander("Show the preprocessing pipeline applied to this image"):
            st.caption(
                "Exactly the pipeline validated in Notebook 2 -- identical at inference time to what "
                "the model saw during training."
            )
            step_columns = st.columns(len(pipeline_result["intermediate_steps"]))
            for column, (step_name, step_image) in zip(
                step_columns, pipeline_result["intermediate_steps"].items()
            ):
                display_image = step_image if step_image.ndim == 2 else cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
                column.image(display_image, caption=step_name, use_container_width=True)

    # --- PDF download ---
    with st.container(border=True):
        st.markdown('<div class="dr-section-title">\U0001F4C4 Screening report</div>', unsafe_allow_html=True)
        st.write(
            "A one-page PDF bundling the image, the Grad-CAM overlay, the grade, the confidence, and "
            "the referral recommendation -- the artefact a screening workflow would pass on."
        )

        original_image_pil = Image.fromarray(
            cv2.cvtColor(pipeline_result["preprocessed_image_uint8"], cv2.COLOR_BGR2RGB)
        )
        overlay_image_pil = Image.fromarray(
            cv2.cvtColor(pipeline_result["overlay_image_bgr"], cv2.COLOR_BGR2RGB)
        )
        report_pdf_bytes = report_generator.build_screening_report_pdf(
            original_image_pil,
            overlay_image_pil,
            mc_dropout_result,
            confidence_assessment,
            referral_recommendation,
            pipeline_result["explanation_text"],
            patient_reference_label=uploaded_file_name,
        )
        st.download_button(
            "Download screening report (PDF)",
            data=report_pdf_bytes,
            file_name=f"dr_screening_report_{uploaded_file_name}.pdf",
            mime="application/pdf",
            key=f"download_button_{uploaded_file_name}_{id(pipeline_result)}",
        )


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def render_dashboard_page():
    render_page_header(
        "\U0001F4CA Dashboard",
        "System status, session activity, and how this screening-support tool works.",
    )

    # --- Model status strip ---
    st.markdown('<div class="dr-section-title">Model status</div>', unsafe_allow_html=True)
    status_columns = st.columns(4)
    status_columns[0].markdown(
        render_stat_card_html(f"{model_performance.OVERALL_TEST_ACCURACY:.1%}", "Test accuracy"),
        unsafe_allow_html=True,
    )
    status_columns[1].markdown(
        render_stat_card_html(f"{model_performance.QUADRATIC_WEIGHTED_KAPPA:.3f}", "QWK (ordinal)"),
        unsafe_allow_html=True,
    )
    status_columns[2].markdown(
        render_stat_card_html("5", "Severity grades"),
        unsafe_allow_html=True,
    )
    status_columns[3].markdown(
        render_stat_card_html(f"{config.MONTE_CARLO_DROPOUT_FORWARD_PASSES}", "MC Dropout passes"),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dr-note">These are real measured figures from this project\'s held-out '
        "550-image test set (Notebook 6), not illustrative placeholders. The full breakdown, "
        "including where the model is weakest, is on the <b>Model Performance</b> page.</div>",
        unsafe_allow_html=True,
    )

    # --- Session activity ---
    st.markdown("")
    left_column, right_column = st.columns([1, 1])

    with left_column:
        with st.container(border=True):
            st.markdown('<div class="dr-section-title">This session</div>', unsafe_allow_html=True)
            scan_history = st.session_state.get("scan_history", [])

            if not scan_history:
                st.info("No scans yet. Head to **New Scan** to grade your first fundus image.")
            else:
                st.metric("Images graded", len(scan_history))

                grade_counts = {}
                for history_entry in scan_history:
                    grade_name = history_entry["pipeline_result"]["mc_dropout_result"]["predicted_class_name"]
                    grade_counts[grade_name] = grade_counts.get(grade_name, 0) + 1

                grade_distribution_dataframe = pd.DataFrame({
                    "Grade": list(grade_counts.keys()),
                    "Count": list(grade_counts.values()),
                })
                st.bar_chart(
                    grade_distribution_dataframe.set_index("Grade")["Count"],
                    color=BRAND_SECONDARY_COLOUR,
                    height=200,
                )

                low_confidence_count = sum(
                    1 for history_entry in scan_history
                    if history_entry["pipeline_result"]["confidence_assessment"]["confidence_level"] == "Low"
                )
                if low_confidence_count:
                    st.warning(
                        f"{low_confidence_count} of {len(scan_history)} scans this session were flagged "
                        f"**Low confidence** and should be prioritised for specialist review."
                    )

    with right_column:
        with st.container(border=True):
            st.markdown('<div class="dr-section-title">How this works</div>', unsafe_allow_html=True)
            st.markdown(
                "**1. Preprocessing** — the uploaded image is cropped, resized to 224x224, "
                "contrast-enhanced (CLAHE) and illumination-corrected (Ben Graham), using the exact "
                "pipeline the model was trained on.\n\n"
                "**2. Grading with uncertainty** — the trained EfficientNetB0 model grades the image "
                f"{config.MONTE_CARLO_DROPOUT_FORWARD_PASSES} times with Monte Carlo Dropout active. "
                "The spread across those passes becomes a genuine confidence estimate.\n\n"
                "**3. Visual explanation** — a Grad-CAM heatmap shows which retinal region drove the "
                "prediction, so the grade can be sanity-checked rather than trusted blindly.\n\n"
                "**4. Clinical framing** — the grade and confidence map to a referral-urgency "
                "recommendation, with low-confidence results always escalating caution, never reducing it.\n\n"
                "**5. Report** — everything is bundled into a downloadable one-page PDF."
            )

    # --- Severity reference ---
    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Severity grades and referral pathway</div>', unsafe_allow_html=True)
        severity_reference_rows = []
        for class_name in config.DIABETIC_RETINOPATHY_CLASS_NAMES:
            referral_information = config.REFERRAL_URGENCY_BY_CLASS_NAME[class_name]
            severity_reference_rows.append({
                "Grade": class_name.replace("_", " "),
                "Urgency": referral_information["urgency_level"],
                "Recommended action": referral_information["recommended_action"],
            })
        st.dataframe(pd.DataFrame(severity_reference_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Simplified, illustrative mapping inspired by NHS England (2023) diabetic eye screening "
            "guidance. Not validated clinical guidance."
        )

    render_footer()


# ---------------------------------------------------------------------------
# Page: New Scan
# ---------------------------------------------------------------------------

def render_new_scan_page(trained_model, gradcam_gradient_model):
    render_page_header(
        "\U0001F50D New Scan",
        "Upload a retinal fundus photograph to receive a graded, explained, confidence-aware result.",
    )

    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Upload a fundus photograph</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Supported formats: PNG, JPG, JPEG",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        st.caption(
            "The image is processed entirely on this machine. Nothing is uploaded to an external "
            "service, and no image is stored beyond this browser session."
        )

    if uploaded_file is None:
        st.markdown(
            '<div class="dr-note">Once you upload an image, this page will show the preprocessed '
            "fundus photograph, a Grad-CAM explanation overlay, the predicted severity grade with an "
            "uncertainty estimate, a referral recommendation, a plain-language explanation, and a "
            "downloadable PDF screening report.</div>",
            unsafe_allow_html=True,
        )
        render_footer()
        return

    uploaded_image_pil = Image.open(uploaded_file)
    pipeline_result = get_or_compute_pipeline_result(
        uploaded_file, uploaded_image_pil, trained_model, gradcam_gradient_model
    )

    render_result_detail(pipeline_result, uploaded_file.name, show_preprocessing_steps=True)
    render_footer()


# ---------------------------------------------------------------------------
# Page: Scan History
# ---------------------------------------------------------------------------

def render_scan_history_page():
    render_page_header(
        "\U0001F553 Scan History",
        "Every image graded during this session, reviewable in full without re-running the model.",
    )

    scan_history = st.session_state.get("scan_history", [])

    if not scan_history:
        st.info("No scans yet this session. Go to **New Scan** to grade a fundus image.")
        render_footer()
        return

    # --- Summary table ---
    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Session summary</div>', unsafe_allow_html=True)

        summary_rows = []
        for history_entry in scan_history:
            mc_dropout_result = history_entry["pipeline_result"]["mc_dropout_result"]
            confidence_assessment = history_entry["pipeline_result"]["confidence_assessment"]
            referral_recommendation = history_entry["pipeline_result"]["referral_recommendation"]
            summary_rows.append({
                "Time": history_entry["scan_timestamp"].strftime("%H:%M:%S"),
                "File": history_entry["file_name"],
                "Grade": mc_dropout_result["predicted_class_name"].replace("_", " "),
                "Confidence": f"{mc_dropout_result['predicted_class_confidence']:.0%}",
                "Confidence level": confidence_assessment["confidence_level"],
                "Urgency": referral_recommendation["urgency_level"],
            })

        summary_dataframe = pd.DataFrame(summary_rows)
        st.dataframe(summary_dataframe, hide_index=True, use_container_width=True)

        csv_export_bytes = summary_dataframe.to_csv(index=False).encode("utf-8")
        export_column, clear_column = st.columns([1, 1])
        export_column.download_button(
            "Export session log (CSV)",
            data=csv_export_bytes,
            file_name=f"dr_session_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
        if clear_column.button("Clear session history"):
            st.session_state["scan_history"] = []
            st.session_state.pop("current_pipeline_result", None)
            st.session_state.pop("current_upload_signature", None)
            st.rerun()

    # --- Full detail per scan ---
    st.markdown('<div class="dr-section-title">Individual scans</div>', unsafe_allow_html=True)
    st.caption(
        "Results below are the stored originals -- redisplayed, never recomputed, so the figures "
        "match exactly what was reported at the time of the scan."
    )

    for history_index, history_entry in enumerate(scan_history):
        mc_dropout_result = history_entry["pipeline_result"]["mc_dropout_result"]
        confidence_assessment = history_entry["pipeline_result"]["confidence_assessment"]

        expander_label = (
            f"{history_entry['scan_timestamp'].strftime('%H:%M:%S')}  |  {history_entry['file_name']}  |  "
            f"{mc_dropout_result['predicted_class_name'].replace('_', ' ')}  "
            f"({mc_dropout_result['predicted_class_confidence']:.0%}, "
            f"{confidence_assessment['confidence_level']} confidence)"
        )

        with st.expander(expander_label, expanded=(history_index == 0)):
            render_result_detail(
                history_entry["pipeline_result"],
                history_entry["file_name"],
                show_preprocessing_steps=False,
            )

    render_footer()


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

def render_model_performance_page():
    render_page_header(
        "\U0001F4C8 Model Performance",
        "Real measured results from the held-out test set -- including where this model is weakest.",
    )

    st.markdown(
        '<div class="dr-note">Every figure on this page was measured on a <b>550-image test set '
        "that was held out and completely untouched</b> until final evaluation (Notebook 6) — no "
        "model selection, tuning, or early-stopping decision ever saw it. Weaknesses are shown "
        "alongside strengths deliberately: a screening tool that hides how often it is wrong would "
        "be misleading to the clinician relying on it.</div>",
        unsafe_allow_html=True,
    )

    # --- Headline metrics ---
    headline_columns = st.columns(4)
    headline_columns[0].markdown(
        render_stat_card_html(f"{model_performance.OVERALL_TEST_ACCURACY:.2%}", "Test accuracy"),
        unsafe_allow_html=True,
    )
    headline_columns[1].markdown(
        render_stat_card_html(f"{model_performance.QUADRATIC_WEIGHTED_KAPPA:.4f}", "Quadratic weighted kappa"),
        unsafe_allow_html=True,
    )
    headline_columns[2].markdown(
        render_stat_card_html(f"{model_performance.NOTEBOOK_5_BEST_VALIDATION_ACCURACY:.2%}", "Validation accuracy"),
        unsafe_allow_html=True,
    )
    headline_columns[3].markdown(
        render_stat_card_html(f"{model_performance.TEST_SET_IMAGE_COUNT}", "Test images"),
        unsafe_allow_html=True,
    )

    st.markdown("")
    performance_tabs = st.tabs([
        "Per-class metrics", "Error structure", "Dataset", "Architecture", "Limitations"
    ])

    # --- Tab 1: per-class metrics ---
    with performance_tabs[0]:
        per_class_dataframe, metrics_data_source = model_performance.load_per_class_metrics_dataframe()

        st.markdown('<div class="dr-section-title">Precision, recall and F1 by severity grade</div>', unsafe_allow_html=True)
        st.caption(f"Source: {metrics_data_source}.")

        display_dataframe = per_class_dataframe.copy()
        display_dataframe.columns = [
            column_name.replace("_", " ").title() for column_name in display_dataframe.columns
        ]
        st.dataframe(display_dataframe, hide_index=True, use_container_width=True)

        st.markdown("**Recall by grade** — the share of truly affected eyes the model actually catches:")
        recall_chart_dataframe = per_class_dataframe.set_index("class_name")["recall"]
        st.bar_chart(recall_chart_dataframe, color="#C4314B", height=240)

        st.markdown(
            '<div class="dr-limitation"><b>Read this honestly:</b> No_DR detection is strong (F1 0.95) '
            "and Moderate is acceptable (F1 0.71), but recall for <b>Mild (0.32), Severe (0.38) and "
            "Proliferative_DR (0.27)</b> is weak — the model misses roughly two thirds to three quarters "
            "of these cases, and those are precisely the sight-threatening stages. This is exactly why "
            "the application surfaces an uncertainty estimate on every prediction rather than presenting "
            "each grade as settled.</div>",
            unsafe_allow_html=True,
        )

        macro_column, weighted_column = st.columns(2)
        with macro_column:
            st.markdown("**Macro average** (treats every class equally)")
            st.write(
                f"Precision {model_performance.MACRO_AVERAGE_TEST_METRICS['precision']:.2f} · "
                f"Recall {model_performance.MACRO_AVERAGE_TEST_METRICS['recall']:.2f} · "
                f"F1 {model_performance.MACRO_AVERAGE_TEST_METRICS['f1_score']:.2f}"
            )
        with weighted_column:
            st.markdown("**Weighted average** (weighted by class size)")
            st.write(
                f"Precision {model_performance.WEIGHTED_AVERAGE_TEST_METRICS['precision']:.2f} · "
                f"Recall {model_performance.WEIGHTED_AVERAGE_TEST_METRICS['recall']:.2f} · "
                f"F1 {model_performance.WEIGHTED_AVERAGE_TEST_METRICS['f1_score']:.2f}"
            )

    # --- Tab 2: error structure ---
    with performance_tabs[1]:
        st.markdown('<div class="dr-section-title">How wrong are the wrong answers?</div>', unsafe_allow_html=True)
        st.write(
            "Diabetic retinopathy grades are ordinal (0-4), so not all errors are equally serious. "
            "Mistaking Mild for Moderate is a low-stakes disagreement; mistaking No DR for a "
            "proliferative stage (or the reverse) is the error a deployed system must minimise most."
        )

        error_columns = st.columns(3)
        error_columns[0].markdown(
            render_stat_card_html(f"{model_performance.TOTAL_MISCLASSIFICATION_COUNT}", "Total errors"),
            unsafe_allow_html=True,
        )
        error_columns[1].markdown(
            render_stat_card_html(
                f"{model_performance.ADJACENT_STAGE_ERROR_SHARE:.1%}", "Adjacent-stage (minor)"
            ),
            unsafe_allow_html=True,
        )
        error_columns[2].markdown(
            render_stat_card_html(f"{model_performance.FAR_MISS_ERROR_SHARE:.1%}", "Far-miss (serious)"),
            unsafe_allow_html=True,
        )

        st.markdown("")
        error_breakdown_dataframe = pd.DataFrame({
            "Error type": ["Adjacent-stage (distance 1)", "Far-miss (distance 2+)"],
            "Count": [
                model_performance.ADJACENT_STAGE_ERROR_COUNT,
                model_performance.FAR_MISS_ERROR_COUNT,
            ],
        })
        st.bar_chart(
            error_breakdown_dataframe.set_index("Error type")["Count"],
            color=BRAND_SECONDARY_COLOUR,
            height=200,
        )

        st.markdown(
            f'<div class="dr-note">The quadratic weighted kappa of '
            f"<b>{model_performance.QUADRATIC_WEIGHTED_KAPPA:.4f}</b> sits well above the raw accuracy of "
            f"<b>{model_performance.OVERALL_TEST_ACCURACY:.4f}</b>, and this error breakdown explains why: "
            f"{model_performance.ADJACENT_STAGE_ERROR_SHARE:.1%} of mistakes are only one stage off, "
            "rather than scattered randomly across the severity scale.</div>",
            unsafe_allow_html=True,
        )

        confusion_matrix_figure_path = model_performance.find_artifact_figure_path(
            model_performance.NOTEBOOK_6_ARTIFACTS_DIRECTORY, "notebook_6_confusion_matrix.png"
        )
        if confusion_matrix_figure_path:
            st.markdown("**Confusion matrix (raw counts and row-normalised)**")
            st.image(confusion_matrix_figure_path, use_container_width=True)
        else:
            st.caption(
                "Confusion matrix figure not found in kaggle_artifacts/notebook_6_evaluation_gradcam/. "
                "Copy notebook_6_confusion_matrix.png there to display it here."
            )

    # --- Tab 3: dataset ---
    with performance_tabs[2]:
        st.markdown('<div class="dr-section-title">Training data composition</div>', unsafe_allow_html=True)

        dataset_columns = st.columns(4)
        dataset_columns[0].markdown(
            render_stat_card_html(f"{model_performance.ORIGINAL_TOTAL_IMAGE_COUNT:,}", "Original images"),
            unsafe_allow_html=True,
        )
        dataset_columns[1].markdown(
            render_stat_card_html(f"{model_performance.ORIGINAL_CLASS_IMBALANCE_RATIO:.2f}x", "Original imbalance"),
            unsafe_allow_html=True,
        )
        dataset_columns[2].markdown(
            render_stat_card_html(f"{model_performance.REBALANCED_CLASS_IMBALANCE_RATIO:.2f}x", "After balancing"),
            unsafe_allow_html=True,
        )
        dataset_columns[3].markdown(
            render_stat_card_html(
                f"{model_performance.SYNTHETIC_TRAINING_IMAGE_SHARE:.0%}", "Synthetic (augmented)"
            ),
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("**Original class distribution (APTOS 2019, before augmentation)**")
        class_distribution_dataframe = model_performance.get_class_distribution_dataframe()
        st.bar_chart(
            class_distribution_dataframe.set_index("class_name")["image_count"],
            color=BRAND_PRIMARY_COLOUR,
            height=260,
        )

        st.markdown(
            f'<div class="dr-note"><b>Synthetic data declaration:</b> '
            f"{model_performance.SYNTHETIC_TRAINING_IMAGE_SHARE:.0%} of the training set "
            f"({model_performance.AUGMENTED_TRAINING_SET_SIZE:,} images total) is synthetic — produced by "
            "rotation, flipping, zoom and brightness augmentation of real images, applied specifically to "
            "counter the severe class imbalance shown above. Augmentation was confined <b>entirely to the "
            "training split</b>; the validation and test splits are 100% real, unaugmented images, so the "
            "performance figures on this page are not inflated by synthetic data.</div>",
            unsafe_allow_html=True,
        )

    # --- Tab 4: architecture ---
    with performance_tabs[3]:
        st.markdown('<div class="dr-section-title">Model and training configuration</div>', unsafe_allow_html=True)
        configuration_dataframe = pd.DataFrame(
            model_performance.MODEL_CONFIGURATION_FACTS, columns=["Property", "Value"]
        )
        st.dataframe(configuration_dataframe, hide_index=True, use_container_width=True)

        training_curves_figure_path = model_performance.find_artifact_figure_path(
            model_performance.NOTEBOOK_5_ARTIFACTS_DIRECTORY, "notebook_5_combined_training_curves.png"
        )
        if training_curves_figure_path:
            st.markdown("**Training curves (18 epochs: 8 Stage 1 + 10 Stage 2)**")
            st.image(training_curves_figure_path, use_container_width=True)
        else:
            st.caption(
                "Training-curve figure not found in kaggle_artifacts/notebook_5_training_strategy/. "
                "Copy notebook_5_combined_training_curves.png there to display it here."
            )

        st.markdown(
            f'<div class="dr-note"><b>Why Monte Carlo Dropout needs no retraining:</b> the classification '
            "head above already contains Dropout(0.3) and Dropout(0.2) layers, added during training to "
            "prevent overfitting. At normal inference these are switched off. This application keeps them "
            f"active and runs each image {config.MONTE_CARLO_DROPOUT_FORWARD_PASSES} times — the spread "
            "across those passes is the uncertainty estimate. No weights are changed, and no additional "
            "training happens anywhere in this application.</div>",
            unsafe_allow_html=True,
        )

    # --- Tab 5: limitations ---
    with performance_tabs[4]:
        st.markdown('<div class="dr-section-title">Known limitations</div>', unsafe_allow_html=True)
        st.write(
            "Stated plainly rather than smoothed over. Each of these is a real, measured finding from "
            "this project's own evaluation."
        )
        for limitation_text in model_performance.DOCUMENTED_MODEL_LIMITATIONS:
            st.markdown(f'<div class="dr-limitation">{limitation_text}</div>', unsafe_allow_html=True)

        st.markdown('<div class="dr-section-title">Where this would go next</div>', unsafe_allow_html=True)
        st.markdown(
            "- **Targeted data collection** for Mild, Severe and Proliferative_DR, whose small real-image "
            "counts (135-258 each) are the root cause of the weak recall above.\n"
            "- **Cross-checking explainability** with Grad-CAM++ or SHAP, to test whether the optic-disc "
            "confound persists across methods.\n"
            "- **External validation** on fundus images from a different population and camera setup.\n"
            "- **Prospective clinical evaluation** against expert graders before any real-world use."
        )

    render_footer()


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def render_about_page():
    render_page_header(
        "\u2139\uFE0F About this application",
        "Purpose, methodology, and the honest limits of what this prototype can do.",
    )

    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Purpose</div>', unsafe_allow_html=True)
        st.write(
            "Diabetic retinopathy is a leading cause of preventable blindness, and screening programmes "
            "generate far more fundus images than specialists can review quickly. This application is a "
            "screening-support prototype: it grades an image, shows why it reached that grade, states how "
            "confident it is, and turns the result into a referral recommendation and a shareable report. "
            "It is designed to help a clinician triage and prioritise — never to replace their judgement."
        )

    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Processing pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            "```\n"
            "Uploaded fundus image\n"
            "        |\n"
            "        v\n"
            "[1] Preprocessing  (Notebook 2 pipeline)\n"
            "        crop dark border -> resize 224x224 -> CLAHE -> Ben Graham illumination correction\n"
            "        |\n"
            "        v\n"
            "[2] EfficientNetB0 inference  (Notebook 5 trained model)\n"
            f"        {config.MONTE_CARLO_DROPOUT_FORWARD_PASSES} Monte Carlo Dropout passes -> mean probabilities + uncertainty\n"
            "        |\n"
            "        v\n"
            "[3] Grad-CAM explanation  (Notebook 6 implementation)\n"
            "        heatmap over the last convolutional layer -> region description\n"
            "        |\n"
            "        v\n"
            "[4] Clinical triage + plain-language explanation\n"
            "        grade + confidence -> referral urgency (low confidence escalates)\n"
            "        |\n"
            "        v\n"
            "[5] Downloadable PDF screening report\n"
            "```"
        )

    with st.container(border=True):
        st.markdown('<div class="dr-section-title">What makes this more than a classifier demo</div>', unsafe_allow_html=True)
        st.markdown(
            "- **Uncertainty is first-class.** Every prediction carries a Monte Carlo Dropout confidence "
            "estimate, and a low-confidence result always escalates caution rather than reducing it.\n"
            "- **Explanations are visual and verbal.** A Grad-CAM overlay shows *where* the model looked; "
            "a plain-language paragraph explains *what that means* to a non-specialist.\n"
            "- **Output is clinically framed.** The result is a referral urgency and a shareable PDF, not "
            "just a class label.\n"
            "- **Limitations are on display, not buried.** The Model Performance page shows the model's "
            "weakest classes as prominently as its strongest.\n"
            "- **No additional training was required.** Every feature reuses the already-trained model."
        )

    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Safety and ethics</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dr-disclaimer">{clinical_triage.CLINICAL_SAFETY_DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "\n"
            "- The referral-urgency mapping is a **simplified, illustrative prototype mapping**, not "
            "validated clinical guidance.\n"
            "- Images are processed locally and are **not stored** beyond the browser session or sent to "
            "any external service.\n"
            "- The model was trained on a single public dataset (APTOS 2019); performance on images from "
            "different populations, cameras or capture conditions is **unverified**.\n"
            "- Automated grades must never be the sole basis for a clinical decision."
        )

    with st.container(border=True):
        st.markdown('<div class="dr-section-title">Technology and sources</div>', unsafe_allow_html=True)
        st.markdown(
            "**Built with:** TensorFlow/Keras (EfficientNetB0), OpenCV, Streamlit, ReportLab, pandas, NumPy.\n\n"
            "**Dataset:** APTOS 2019 Blindness Detection (Kaggle), 3,662 labelled fundus images.\n\n"
            "**Key references:**\n"
            "- NHS England (2023) *Diabetic eye screening: patient grading, referral, surveillance* — "
            "informs the referral-urgency mapping.\n"
            "- Siebert, M., Grasshoff, J. and Rostalski, P. (2023) 'Uncertainty Analysis of Deep Kernel "
            "Learning Methods on Diabetic Retinopathy Grading', *IEEE Access*, 11, pp. 146173-146184.\n"
            "- Chilukoti, S.V. et al. (2024) 'A reliable diabetic retinopathy grading via transfer learning "
            "and ensemble learning with quadratic weighted kappa metric', *BMC Medical Informatics and "
            "Decision Making*, 24, p.37."
        )

    render_footer()


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def render_streamlit_interface():
    st.set_page_config(
        page_title="DR Grading Assistant",
        layout="wide",
        page_icon="\U0001FA7A",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    selected_page_name = render_sidebar_navigation()

    # The model is only needed for the New Scan page, but loading it here (cached) means the very
    # first scan doesn't pay the full model-load delay after the user has already clicked upload.
    trained_model, gradcam_gradient_model = get_cached_model_and_gradcam_gradient_model()

    if selected_page_name == "Dashboard":
        render_dashboard_page()
    elif selected_page_name == "New Scan":
        render_new_scan_page(trained_model, gradcam_gradient_model)
    elif selected_page_name == "Scan History":
        render_scan_history_page()
    elif selected_page_name == "Model Performance":
        render_model_performance_page()
    elif selected_page_name == "About":
        render_about_page()


if __name__ == "__main__":
    render_streamlit_interface()
