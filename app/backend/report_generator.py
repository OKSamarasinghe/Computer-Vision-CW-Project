"""
Builds the downloadable PDF screening report -- the concrete, shareable artefact that makes this
application's "practical impact" argument tangible rather than just an on-screen demo.

The report is written for a clinical reader rather than a technical one: it leads with the grade and
the recommended action, states plainly how confident the model is, shows the full probability
breakdown so a borderline result is visible rather than hidden behind a single label, and carries
the safety disclaimer on every page. A short methodology appendix is included so a reviewer can see
exactly how the result was produced without needing the source code.

Built with ReportLab's platypus layout engine (rather than raw canvas drawing) so that long
explanation text flows and paginates correctly instead of overrunning a fixed text box.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
    KeepTogether,
)


# Colour tokens kept consistent with the web interface, so a downloaded report visually matches
# the screen it came from.
BRAND_PRIMARY_COLOUR = colors.HexColor("#0E8C55")
BRAND_DEEP_COLOUR = colors.HexColor("#0A6B42")
NEUTRAL_TEXT_COLOUR = colors.HexColor("#1F2933")
MUTED_TEXT_COLOUR = colors.HexColor("#5C6B7A")
LIGHT_RULE_COLOUR = colors.HexColor("#D8E3DC")

# Four-step escalation scale. Yellow/amber has been removed from the palette entirely; teal
# replaces it as the middle tier. Warm colours are retained ONLY for the genuinely time-critical
# levels, because collapsing the scale to a single colour would remove a reader's ability to tell
# a routine result from a sight-threatening one at a glance.
URGENCY_COLOUR_BY_LEVEL = {
    "Routine": colors.HexColor("#1B5E3A"),
    "Routine referral": colors.HexColor("#0F5C73"),
    "Urgent": colors.HexColor("#B3401A"),
    "Emergency": colors.HexColor("#A3122B"),
}
URGENCY_BACKGROUND_BY_LEVEL = {
    "Routine": colors.HexColor("#E3F2E8"),
    "Routine referral": colors.HexColor("#DEEFF4"),
    "Urgent": colors.HexColor("#FFE8E0"),
    "Emergency": colors.HexColor("#FCE1E4"),
}

DEFAULT_CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

CLASS_CLINICAL_SUMMARY = {
    "No_DR": "No visible diabetic retinopathy",
    "Mild": "Mild non-proliferative (microaneurysms only)",
    "Moderate": "Moderate non-proliferative",
    "Severe": "Severe non-proliferative",
    "Proliferative_DR": "Proliferative diabetic retinopathy",
}


def _build_paragraph_styles():
    """Create the named paragraph styles this report uses."""
    base_stylesheet = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base_stylesheet["Title"], fontName="Helvetica-Bold",
            fontSize=17, leading=21, textColor=BRAND_DEEP_COLOUR, alignment=TA_LEFT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base_stylesheet["Normal"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=MUTED_TEXT_COLOUR, spaceAfter=8,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading", parent=base_stylesheet["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=BRAND_DEEP_COLOUR, spaceBefore=10, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyText2", parent=base_stylesheet["Normal"], fontName="Helvetica",
            fontSize=9, leading=12.5, textColor=NEUTRAL_TEXT_COLOUR, spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base_stylesheet["Normal"], fontName="Helvetica-Oblique",
            fontSize=7.5, leading=9.5, textColor=MUTED_TEXT_COLOUR,
        ),
        "headline_caption": ParagraphStyle(
            "HeadlineCaption", parent=base_stylesheet["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13, textColor=BRAND_DEEP_COLOUR, alignment=2,
        ),
        "grade_headline": ParagraphStyle(
            "GradeHeadline", parent=base_stylesheet["Normal"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, textColor=NEUTRAL_TEXT_COLOUR,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=base_stylesheet["Normal"], fontName="Helvetica-Oblique",
            fontSize=7.5, leading=10, textColor=colors.HexColor("#14452F"),
        ),
    }


def _convert_pil_image_to_reportlab_flowable(pil_image, display_width_mm):
    """Convert a PIL image into a ReportLab flowable, preserving its aspect ratio.

    The image is written to an in-memory PNG buffer rather than a temp file, so nothing is left on
    disk after a report is generated -- consistent with the application's claim that uploaded
    images are not persisted.
    """
    image_buffer = io.BytesIO()
    pil_image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    original_width, original_height = pil_image.size
    display_width = display_width_mm * mm
    display_height = display_width * (original_height / original_width)

    return ReportLabImage(image_buffer, width=display_width, height=display_height)


def _build_header_footer_drawer(patient_reference_label, generation_timestamp_text):
    """Return a callback that draws the running header rule, footer disclaimer and page number.

    Drawn on every page rather than only the first, so a detached second page is still clearly
    identifiable and still carries the safety disclaimer -- important for a document that could be
    printed and passed around a clinic.
    """
    def draw_page_furniture(pdf_canvas, document_template):
        page_width, page_height = A4
        pdf_canvas.saveState()

        # Header accent rule
        pdf_canvas.setStrokeColor(BRAND_PRIMARY_COLOUR)
        pdf_canvas.setLineWidth(2)
        pdf_canvas.line(18 * mm, page_height - 14 * mm, page_width - 18 * mm, page_height - 14 * mm)

        pdf_canvas.setFont("Helvetica", 7)
        pdf_canvas.setFillColor(MUTED_TEXT_COLOUR)
        pdf_canvas.drawString(18 * mm, page_height - 11 * mm, "Diabetic Retinopathy Screening Report")
        pdf_canvas.drawRightString(
            page_width - 18 * mm, page_height - 11 * mm,
            f"Ref: {patient_reference_label}  |  {generation_timestamp_text}",
        )

        # Footer rule, disclaimer and page number
        pdf_canvas.setStrokeColor(LIGHT_RULE_COLOUR)
        pdf_canvas.setLineWidth(0.5)
        pdf_canvas.line(18 * mm, 16 * mm, page_width - 18 * mm, 16 * mm)

        pdf_canvas.setFont("Helvetica-Oblique", 6.5)
        pdf_canvas.setFillColor(MUTED_TEXT_COLOUR)
        pdf_canvas.drawString(
            18 * mm, 12 * mm,
            "Automated screening-support prototype - NOT a certified diagnostic device. "
            "Requires review by a qualified ophthalmologist or optometrist.",
        )
        pdf_canvas.drawRightString(page_width - 18 * mm, 12 * mm, f"Page {pdf_canvas.getPageNumber()}")

        pdf_canvas.restoreState()

    return draw_page_furniture


def _build_result_summary_table(mc_dropout_result, confidence_assessment, referral_recommendation):
    """Build the headline grade / confidence / urgency summary block."""
    predicted_grade_display = mc_dropout_result["predicted_class_name"].replace("_", " ")
    clinical_summary = CLASS_CLINICAL_SUMMARY.get(mc_dropout_result["predicted_class_name"], "")
    urgency_level = referral_recommendation["urgency_level"]

    summary_rows = [
        ["Predicted grade", f"{predicted_grade_display}  -  {clinical_summary}"],
        ["Model confidence",
         f"{mc_dropout_result['predicted_class_confidence']:.1%}  "
         f"({confidence_assessment['confidence_level']} confidence, "
         f"std {mc_dropout_result['predicted_class_uncertainty_std']:.3f})"],
        ["Referral urgency", urgency_level],
        ["Recommended action", referral_recommendation["recommended_action"]],
    ]

    paragraph_styles = _build_paragraph_styles()
    formatted_rows = [
        [Paragraph(f"<b>{label}</b>", paragraph_styles["body"]), Paragraph(str(value), paragraph_styles["body"])]
        for label, value in summary_rows
    ]

    summary_table = Table(formatted_rows, colWidths=[38 * mm, 136 * mm])
    summary_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LIGHT_RULE_COLOUR),
        ("BACKGROUND", (0, 2), (-1, 2), URGENCY_BACKGROUND_BY_LEVEL.get(urgency_level, colors.white)),
        ("TEXTCOLOR", (1, 2), (1, 2), URGENCY_COLOUR_BY_LEVEL.get(urgency_level, NEUTRAL_TEXT_COLOUR)),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return summary_table


def _build_probability_table(mc_dropout_result, class_names):
    """Build the full per-class probability table, with a simple inline bar for each class.

    Showing every class -- not just the winner -- is deliberate: a 35%-vs-33% split between two
    adjacent grades means something very different clinically from a 95% single-class result, and
    that distinction is invisible if only the top grade is reported.
    """
    paragraph_styles = _build_paragraph_styles()

    header_row = [
        Paragraph("<b>Severity grade</b>", paragraph_styles["body"]),
        Paragraph("<b>Mean probability</b>", paragraph_styles["body"]),
        Paragraph("<b>Variation (std)</b>", paragraph_styles["body"]),
        Paragraph("<b>Relative</b>", paragraph_styles["body"]),
    ]

    table_rows = [header_row]
    predicted_index = mc_dropout_result["predicted_class_index"]

    for class_index, class_name in enumerate(class_names):
        probability_value = float(mc_dropout_result["mean_class_probabilities"][class_index])
        standard_deviation = float(mc_dropout_result["per_class_standard_deviation"][class_index])

        # A text-based proportional bar keeps the table self-contained (no chart image needed) and
        # still communicates the shape of the distribution at a glance.
        filled_block_count = int(round(probability_value * 20))
        proportional_bar = "\u2588" * filled_block_count if filled_block_count else "\u00b7"

        is_predicted_class = class_index == predicted_index
        name_markup = class_name.replace("_", " ")
        if is_predicted_class:
            name_markup = f"<b>{name_markup}</b>"

        table_rows.append([
            Paragraph(name_markup, paragraph_styles["body"]),
            Paragraph(f"{probability_value:.1%}", paragraph_styles["body"]),
            Paragraph(f"{standard_deviation:.3f}", paragraph_styles["body"]),
            Paragraph(f'<font color="#0E8C55">{proportional_bar}</font>', paragraph_styles["body"]),
        ])

    probability_table = Table(table_rows, colWidths=[46 * mm, 30 * mm, 28 * mm, 70 * mm])
    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F5EE")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BRAND_PRIMARY_COLOUR),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, LIGHT_RULE_COLOUR),
        ("BACKGROUND", (0, predicted_index + 1), (-1, predicted_index + 1), colors.HexColor("#F3FAF6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return probability_table


def build_screening_report_pdf(
    original_image_pil,
    gradcam_overlay_image_pil,
    mc_dropout_result,
    confidence_assessment,
    referral_recommendation,
    explanation_text,
    patient_reference_label="Unlabelled upload",
    class_names=None,
    heatmap_region_description=None,
):
    """Render the screening report and return it as PDF bytes, ready for a download button.

    Signature is backwards-compatible with the original one-page version; class_names and
    heatmap_region_description are optional enrichments.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    paragraph_styles = _build_paragraph_styles()
    generation_timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf_buffer = io.BytesIO()
    document_template = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=f"DR Screening Report - {patient_reference_label}",
        author="Diabetic Retinopathy Grading Assistant",
    )

    story_elements = []

    # --- Title block ---
    story_elements.append(Paragraph("Diabetic Retinopathy Screening Report", paragraph_styles["title"]))
    story_elements.append(Paragraph(
        f"Generated {generation_timestamp_text} &nbsp;|&nbsp; Image reference: {patient_reference_label} "
        f"&nbsp;|&nbsp; Automated screening-support analysis",
        paragraph_styles["subtitle"],
    ))

    # --- Headline grade ---
    predicted_grade_display = mc_dropout_result["predicted_class_name"].replace("_", " ")
    urgency_level = referral_recommendation["urgency_level"]
    clinical_description = CLASS_CLINICAL_SUMMARY.get(mc_dropout_result["predicted_class_name"], "")

    # The banner shows the GRADE only. The referral urgency deliberately does not appear here:
    # as a large standalone phrase ("ROUTINE REFERRAL") it reads as a shouted instruction stripped
    # of its context. It is presented instead in the result-summary table below, on the same row as
    # the recommended action that explains what it actually means.
    headline_table = Table(
        [[
            Paragraph(predicted_grade_display, paragraph_styles["grade_headline"]),
            Paragraph(clinical_description, paragraph_styles["headline_caption"]),
        ]],
        colWidths=[80 * mm, 94 * mm],
    )
    headline_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F5EE")),
        ("LINEBEFORE", (0, 0), (0, -1), 4, BRAND_PRIMARY_COLOUR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story_elements.append(headline_table)
    story_elements.append(Spacer(1, 8))

    # --- Result summary ---
    story_elements.append(Paragraph("1. Result summary", paragraph_styles["section_heading"]))
    story_elements.append(
        _build_result_summary_table(mc_dropout_result, confidence_assessment, referral_recommendation)
    )

    # --- Images ---
    story_elements.append(Paragraph("2. Fundus image and model attention", paragraph_styles["section_heading"]))
    image_display_width_mm = 80
    image_row = Table(
        [
            [
                _convert_pil_image_to_reportlab_flowable(original_image_pil, image_display_width_mm),
                _convert_pil_image_to_reportlab_flowable(gradcam_overlay_image_pil, image_display_width_mm),
            ],
            [
                Paragraph("Preprocessed fundus image", paragraph_styles["caption"]),
                Paragraph("Grad-CAM overlay (warmer = more influential)", paragraph_styles["caption"]),
            ],
        ],
        colWidths=[87 * mm, 87 * mm],
    )
    image_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("TOPPADDING", (0, 1), (-1, 1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story_elements.append(image_row)

    if heatmap_region_description:
        story_elements.append(Spacer(1, 4))
        story_elements.append(Paragraph(
            f"Model attention was {heatmap_region_description.get('concentration_description', 'distributed')}, "
            f"centred in the {heatmap_region_description.get('region_description', 'image')}.",
            paragraph_styles["caption"],
        ))

    # --- Probability breakdown ---
    story_elements.append(Paragraph("3. Full probability breakdown", paragraph_styles["section_heading"]))
    story_elements.append(Paragraph(
        f"Mean probability assigned to each severity grade across "
        f"{mc_dropout_result['number_of_forward_passes']} stochastic (Monte Carlo Dropout) inference passes. "
        f"A high standard deviation indicates the model's opinion changed between passes, which is itself "
        f"evidence the image is difficult to grade.",
        paragraph_styles["body"],
    ))
    story_elements.append(_build_probability_table(mc_dropout_result, class_names))

    # --- Interpretation ---
    story_elements.append(Paragraph("4. Interpretation", paragraph_styles["section_heading"]))
    plain_explanation_text = explanation_text.replace("**", "")
    for explanation_paragraph in plain_explanation_text.split("\n\n"):
        cleaned_paragraph = explanation_paragraph.strip()
        if cleaned_paragraph:
            story_elements.append(Paragraph(cleaned_paragraph, paragraph_styles["body"]))

    story_elements.append(Paragraph(
        f"<b>Confidence assessment:</b> {confidence_assessment['reason']}",
        paragraph_styles["body"],
    ))

    # --- Methodology + limitations, kept together so they never split across pages ---
    methodology_block = [
        Paragraph("5. How this result was produced", paragraph_styles["section_heading"]),
        Paragraph(
            "<b>Preprocessing:</b> dark-border crop, resize to 224x224, CLAHE local contrast enhancement "
            "(LAB lightness channel only), and Ben Graham local-average subtraction for illumination "
            "correction - identical to the pipeline used during model training.",
            paragraph_styles["body"],
        ),
        Paragraph(
            "<b>Model:</b> EfficientNetB0 convolutional neural network, ImageNet pre-trained and fine-tuned "
            "on the APTOS 2019 dataset (3,662 labelled fundus images) via two-stage transfer learning.",
            paragraph_styles["body"],
        ),
        Paragraph(
            f"<b>Uncertainty estimation:</b> the model's dropout layers are kept active at inference and the "
            f"image is graded {mc_dropout_result['number_of_forward_passes']} times. The spread of results "
            f"across those passes is reported above as the standard deviation.",
            paragraph_styles["body"],
        ),
        Paragraph(
            "<b>Explanation:</b> Grad-CAM gradients over the final convolutional layer identify which image "
            "regions most influenced the predicted grade.",
            paragraph_styles["body"],
        ),
        Paragraph(
            "<b>Reported model performance:</b> 75.82% accuracy and 0.8256 quadratic weighted kappa on a "
            "550-image held-out test set. Detection of referable disease is substantially weaker than "
            "detection of healthy retinas: recall is 0.32 (Mild), 0.38 (Severe) and 0.27 (Proliferative). "
            "A negative or low-grade result from this tool therefore does not rule out disease.",
            paragraph_styles["body"],
        ),
    ]
    story_elements.append(KeepTogether(methodology_block))

    # --- Closing disclaimer ---
    story_elements.append(Spacer(1, 6))
    disclaimer_table = Table(
        [[Paragraph(f"<b>Important:</b> {referral_recommendation['disclaimer']}", paragraph_styles["disclaimer"])]],
        colWidths=[174 * mm],
    )
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F5EE")),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BRAND_PRIMARY_COLOUR),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story_elements.append(disclaimer_table)

    page_furniture_drawer = _build_header_footer_drawer(patient_reference_label, generation_timestamp_text)
    document_template.build(
        story_elements, onFirstPage=page_furniture_drawer, onLaterPages=page_furniture_drawer
    )

    pdf_buffer.seek(0)
    print(f"[APP] Generated screening report PDF for reference '{patient_reference_label}'")
    return pdf_buffer.getvalue()
