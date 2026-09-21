"""
Builds a downloadable one-page PDF "screening report" per graded image, bundling the uploaded
image, the Grad-CAM overlay, the predicted grade, confidence, and referral recommendation into a
single artefact a real screening workflow could actually pass along -- the concrete, downloadable
output that makes the "practical impact" argument tangible rather than just a UI demo.

Uses reportlab (pure Python, no external binaries needed) so this runs identically on the coursework
author's development machine and the target 4GB-GPU laptop.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def build_screening_report_pdf(
    original_image_pil,
    gradcam_overlay_image_pil,
    mc_dropout_result,
    confidence_assessment,
    referral_recommendation,
    explanation_text,
    patient_reference_label="Unlabelled upload",
):
    """Render a single-page PDF report and return it as bytes, ready for a Streamlit download button."""
    pdf_buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=A4)
    page_width, page_height = A4

    pdf_canvas.setFont("Helvetica-Bold", 16)
    pdf_canvas.drawString(20 * mm, page_height - 20 * mm, "Diabetic Retinopathy Screening Report")

    pdf_canvas.setFont("Helvetica", 9)
    pdf_canvas.drawString(
        20 * mm,
        page_height - 27 * mm,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}    Reference: {patient_reference_label}",
    )

    image_display_size_mm = 75
    pdf_canvas.drawImage(
        ImageReader(original_image_pil), 20 * mm, page_height - 110 * mm,
        width=image_display_size_mm * mm, height=image_display_size_mm * mm,
        preserveAspectRatio=True, anchor="c",
    )
    pdf_canvas.drawImage(
        ImageReader(gradcam_overlay_image_pil), 105 * mm, page_height - 110 * mm,
        width=image_display_size_mm * mm, height=image_display_size_mm * mm,
        preserveAspectRatio=True, anchor="c",
    )

    pdf_canvas.setFont("Helvetica", 8)
    pdf_canvas.drawString(20 * mm, page_height - 113 * mm, "Uploaded fundus image (preprocessed)")
    pdf_canvas.drawString(105 * mm, page_height - 113 * mm, "Grad-CAM explanation overlay")

    text_start_y = page_height - 125 * mm
    pdf_canvas.setFont("Helvetica-Bold", 12)
    pdf_canvas.drawString(
        20 * mm, text_start_y,
        f"Predicted grade: {mc_dropout_result['predicted_class_name'].replace('_', ' ')}",
    )
    pdf_canvas.setFont("Helvetica", 10)
    pdf_canvas.drawString(
        20 * mm, text_start_y - 6 * mm,
        f"Confidence: {confidence_assessment['confidence_level']} "
        f"({mc_dropout_result['predicted_class_confidence']:.0%}, "
        f"std={mc_dropout_result['predicted_class_uncertainty_std']:.3f})",
    )
    pdf_canvas.drawString(
        20 * mm, text_start_y - 12 * mm,
        f"Recommended action: {referral_recommendation['urgency_level']}",
    )

    text_object = pdf_canvas.beginText(20 * mm, text_start_y - 22 * mm)
    text_object.setFont("Helvetica", 9)
    plain_explanation_text = explanation_text.replace("**", "")
    for paragraph in plain_explanation_text.split("\n\n"):
        for wrapped_line in _wrap_text_to_width(paragraph, 95):
            text_object.textLine(wrapped_line)
        text_object.textLine("")
    pdf_canvas.drawText(text_object)

    pdf_canvas.setFont("Helvetica-Oblique", 7)
    pdf_canvas.drawString(20 * mm, 15 * mm, referral_recommendation["disclaimer"])

    pdf_canvas.showPage()
    pdf_canvas.save()
    pdf_buffer.seek(0)

    print(f"[APP] Generated screening report PDF for reference '{patient_reference_label}'")
    return pdf_buffer.getvalue()


def _wrap_text_to_width(text, max_characters_per_line):
    """Minimal word-wrap helper so long explanation sentences fit inside the PDF's text box."""
    words = text.split()
    lines, current_line = [], ""
    for word in words:
        candidate_line = f"{current_line} {word}".strip()
        if len(candidate_line) > max_characters_per_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = candidate_line
    if current_line:
        lines.append(current_line)
    return lines
