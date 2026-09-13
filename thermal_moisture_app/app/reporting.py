"""PDF export for thermal-moisture inspection results."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _to_png(image: Image.Image | np.ndarray) -> BytesIO:
    """Convert a PIL image or NumPy image into an in-memory PNG."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _report_image(image: Image.Image | np.ndarray, width: float, height: float) -> ReportImage:
    buffer = _to_png(image)
    picture = ReportImage(buffer)
    picture._restrictSize(width, height)
    return picture


def _captioned_image(image: Image.Image | np.ndarray, caption: str, width: float, height: float, style: ParagraphStyle):
    return [
        _report_image(image, width, height),
        Spacer(1, 0.12 * cm),
        Paragraph(caption, style),
    ]


def _four_output_grid(
    *,
    original: Image.Image,
    overlay: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    style: ParagraphStyle,
) -> Table:
    """Create a compact 2 x 2 visual layout for one inspection."""
    image_width = 7.8 * cm
    image_height = 5.2 * cm
    mask_image = mask.astype(np.uint8) * 255
    probability_image = np.clip(probability * 255, 0, 255).astype(np.uint8)
    grid = Table(
        [
            [
                _captioned_image(original, "Original thermal image", image_width, image_height, style),
                _captioned_image(overlay, "Moisture inspection overlay", image_width, image_height, style),
            ],
            [
                _captioned_image(mask_image, "Binary humidity mask", image_width, image_height, style),
                _captioned_image(probability_image, "Ensemble humidity probability", image_width, image_height, style),
            ],
        ],
        colWidths=[8.8 * cm, 8.8 * cm],
    )
    grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return grid


def build_batch_inspection_report(*, inspections: list[dict], notes: str) -> bytes:
    """Build one report containing a section summary and every image inspection."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.55 * cm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("BatchReportTitle", parent=styles["Title"], textColor=colors.HexColor("#125c59"), fontSize=22, leading=27)
    section = ParagraphStyle("BatchSection", parent=styles["Heading2"], textColor=colors.HexColor("#125c59"), spaceBefore=10, spaceAfter=7)
    caption = ParagraphStyle("BatchCaption", parent=styles["BodyText"], textColor=colors.HexColor("#496663"), alignment=1, fontSize=8.5, leading=10)
    body = ParagraphStyle("BatchBody", parent=styles["BodyText"], leading=15)

    highest = max(inspections, key=lambda item: item["humid_area"])
    lowest = min(inspections, key=lambda item: item["humid_area"])
    average_area = sum(item["humid_area"] for item in inspections) / len(inspections)
    summary = [
        ["Images inspected", str(len(inspections))],
        ["Average humid area", f"{average_area:.1f}%"],
        ["Highest humid area", f"{highest['humid_area']:.1f}% - {escape(highest['filename'])}"],
        ["Lowest humid area", f"{lowest['humid_area']:.1f}% - {escape(lowest['filename'])}"],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    summary_table = Table(summary, colWidths=[4.4 * cm, 12.0 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f3f0")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#125c59")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c5ded8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    rows = [["Rank", "Thermal image", "Humid area"]]
    for rank, item in enumerate(sorted(inspections, key=lambda item: item["humid_area"], reverse=True), start=1):
        rows.append([str(rank), escape(item["filename"]), f"{item['humid_area']:.1f}%"])
    ranking_table = Table(rows, colWidths=[1.6 * cm, 11.8 * cm, 3.0 * cm], repeatRows=1)
    ranking_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#125c59")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c5ded8")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5faf8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story = [
        Paragraph("Thermal Moisture Batch Inspection Report", title),
        Spacer(1, 0.3 * cm),
        Paragraph("Building section summary", section),
        summary_table,
        Spacer(1, 0.35 * cm),
        Paragraph("Priority ranking", section),
        ranking_table,
    ]
    if notes.strip():
        story.extend([
            Spacer(1, 0.35 * cm),
            Paragraph("Inspector notes", section),
            Paragraph(escape(notes).replace("\n", "<br/>"), body),
        ])

    for index, item in enumerate(inspections, start=1):
        original = item["image"]
        metadata = [
            ["Inspection file", escape(item["filename"])],
            ["Original image size", f"{original.width} x {original.height} pixels"],
            ["Detected humid area", f"{item['humid_area']:.1f}%"],
        ]
        metadata_table = Table(metadata, colWidths=[4.4 * cm, 12.0 * cm])
        metadata_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f3f0")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#125c59")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c5ded8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        output_grid = _four_output_grid(
            original=original,
            overlay=item["overlay"],
            mask=item["mask"],
            probability=item["probability"],
            style=caption,
        )

        story.extend([
            PageBreak(),
            Paragraph(f"Image {index} - {escape(item['filename'])}", title),
            Spacer(1, 0.2 * cm),
            metadata_table,
            Spacer(1, 0.25 * cm),
            output_grid,
        ])

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c5ded8"))
        canvas.line(1.45 * cm, 1.15 * cm, A4[0] - 1.45 * cm, 1.15 * cm)
        canvas.setFillColor(colors.HexColor("#496663"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(1.45 * cm, 0.75 * cm, "ELBALI mohammed - Thermal Moisture Inspector")
        canvas.drawRightString(A4[0] - 1.45 * cm, 0.75 * cm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return output.getvalue()


def build_inspection_report(
    *,
    original: Image.Image,
    overlay: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    filename: str,
    humid_area: float,
    notes: str,
) -> bytes:
    """Build a two-page, self-contained inspection report in memory."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.55 * cm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#125c59"), fontSize=22, leading=27)
    section = ParagraphStyle("Section", parent=styles["Heading2"], textColor=colors.HexColor("#125c59"), spaceBefore=10, spaceAfter=7)
    caption = ParagraphStyle("Caption", parent=styles["BodyText"], textColor=colors.HexColor("#496663"), alignment=1, fontSize=8.5, leading=10)
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], leading=15)

    metadata = [
        ["Inspection file", escape(filename)],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Original image size", f"{original.width} x {original.height} pixels"],
        ["Detected humid area", f"{humid_area:.1f}%"],
    ]
    metadata_table = Table(metadata, colWidths=[4.4 * cm, 12.0 * cm])
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f3f0")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#125c59")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c5ded8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    output_grid = _four_output_grid(
        original=original,
        overlay=overlay,
        mask=mask,
        probability=probability,
        style=caption,
    )

    story = [
        Paragraph("Thermal Moisture Inspection Report", title),
        Spacer(1, 0.3 * cm),
        Paragraph("Inspection summary", section),
        metadata_table,
        Spacer(1, 0.35 * cm),
        Paragraph("Inspection outputs", section),
        output_grid,
    ]
    if notes.strip():
        story.extend([
            Spacer(1, 0.45 * cm),
            Paragraph("Inspector notes", section),
            Paragraph(escape(notes).replace("\n", "<br/>"), body),
        ])

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c5ded8"))
        canvas.line(1.45 * cm, 1.15 * cm, A4[0] - 1.45 * cm, 1.15 * cm)
        canvas.setFillColor(colors.HexColor("#496663"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(1.45 * cm, 0.75 * cm, "ELBALI mohammed - Thermal Moisture Inspector")
        canvas.drawRightString(A4[0] - 1.45 * cm, 0.75 * cm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return output.getvalue()
