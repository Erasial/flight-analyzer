from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import re
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_LABELS_UA = {
    "Flight Duration (s)": "Тривалість польоту (с)",
    "Distance Traveled (m)": "Пройдена дистанція (м)",
    "Elevation Gain (m)": "Набір висоти (м)",
    "Max Horizontal Speed (m/s)": "Макс. горизонтальна швидкість (м/с)",
    "Max Vertical Speed (m/s)": "Макс. вертикальна швидкість (м/с)",
    "Max Altitude (m)": "Макс. висота (м)",
    "Max Acc X (m/s^2)": "Макс. прискорення X (м/с^2)",
    "Max Acc Y (m/s^2)": "Макс. прискорення Y (м/с^2)",
    "Max Acc Z (m/s^2)": "Макс. прискорення Z (м/с^2)",
    "GPS Sample Rate (Hz)": "Частота GPS (Гц)",
    "IMU Sample Rate (Hz)": "Частота IMU (Гц)",
}


def _localize_metric_key(key: str) -> str:
    return METRIC_LABELS_UA.get(key, key)


def _strip_ai_markdown(line: str) -> str:
    text = line.strip()
    if not text:
        return ""

    # Remove common markdown artifacts from LLM output before PDF rendering.
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"^>\s*", "", text)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def export_metrics_csv_bytes(metrics: dict[str, float]) -> bytes:
    """Serialize metrics to CSV bytes for download/export."""
    rows = [{"Metric": key, "Value": value} for key, value in metrics.items()]
    frame = pd.DataFrame(rows)
    return frame.to_csv(index=False).encode("utf-8")


def _build_summary_rows(df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> list[list[str]]:
    gps_rows = len(df_gps)
    imu_rows = len(df_imu)

    gps_duration_s = "Н/Д"
    if "TimeUS" in df_gps.columns and gps_rows > 1:
        time_us = pd.to_numeric(df_gps["TimeUS"], errors="coerce").dropna()
        if len(time_us) > 1:
            gps_duration_s = f"{(float(time_us.iloc[-1]) - float(time_us.iloc[0])) / 1e6:.2f}"

    imu_duration_s = "Н/Д"
    if "TimeUS" in df_imu.columns and imu_rows > 1:
        time_us = pd.to_numeric(df_imu["TimeUS"], errors="coerce").dropna()
        if len(time_us) > 1:
            imu_duration_s = f"{(float(time_us.iloc[-1]) - float(time_us.iloc[0])) / 1e6:.2f}"

    return [
        ["Кількість GPS семплів", str(gps_rows)],
        ["Кількість IMU семплів", str(imu_rows)],
        ["Тривалість GPS (с)", gps_duration_s],
        ["Тривалість IMU (с)", imu_duration_s],
        ["Є швидкість з акселерометра", "Так" if "VelAccNorm" in df_imu.columns else "Ні"],
    ]


def _build_speed_comparison_png_bytes(
    df_gps: pd.DataFrame,
    df_imu: pd.DataFrame,
    speed_unit: str,
) -> bytes | None:
    if "Spd" not in df_gps.columns or "TimeUS" not in df_gps.columns:
        return None
    if "VelAccNorm" not in df_imu.columns or "TimeUS" not in df_imu.columns:
        return None

    gps = df_gps.copy()
    imu = df_imu.copy()
    gps["TimeUS"] = pd.to_numeric(gps["TimeUS"], errors="coerce")
    gps["Spd"] = pd.to_numeric(gps["Spd"], errors="coerce")
    imu["TimeUS"] = pd.to_numeric(imu["TimeUS"], errors="coerce")
    imu["VelAccNorm"] = pd.to_numeric(imu["VelAccNorm"], errors="coerce")

    gps = gps.dropna(subset=["TimeUS", "Spd"]).sort_values("TimeUS")
    imu = imu.dropna(subset=["TimeUS", "VelAccNorm"]).sort_values("TimeUS")
    if gps.empty or imu.empty:
        return None

    t0 = min(float(gps["TimeUS"].iloc[0]), float(imu["TimeUS"].iloc[0]))
    gps_t = (gps["TimeUS"] - t0) / 1e6
    imu_t = (imu["TimeUS"] - t0) / 1e6

    unit_factor = 3.6 if speed_unit == "км/год" else 1.0
    gps_speed = gps["Spd"] * unit_factor
    imu_speed = imu["VelAccNorm"] * unit_factor

    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    ax.plot(gps_t.to_numpy(dtype=float), gps_speed.to_numpy(dtype=float), label=f"Швидкість GPS ({speed_unit})", linewidth=1.8)
    ax.plot(
        imu_t.to_numpy(dtype=float),
        imu_speed.to_numpy(dtype=float),
        label=f"Швидкість з акселерометра ({speed_unit}) [із дрейфом]",
        linewidth=1.6,
    )
    ax.set_title("Порівняння швидкостей: GPS vs інтегрування акселерометра")
    ax.set_xlabel("Час (с)")
    ax.set_ylabel(f"Швидкість ({speed_unit})")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    out = BytesIO()
    fig.tight_layout()
    fig.savefig(out, format="png")
    plt.close(fig)
    return out.getvalue()


def _build_trajectory_sample_png_bytes(df_gps: pd.DataFrame) -> bytes | None:
    if df_gps.empty:
        return None

    required = {"East", "North", "Up"}
    if not required.issubset(set(df_gps.columns)):
        return None

    x = pd.to_numeric(df_gps["East"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_gps["North"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df_gps["Up"], errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 2:
        return None

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="#1f77b4", linewidth=1.8)
    ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=20, label="Start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=20, label="End")
    ax.set_title("3D-траєкторія (приклад)")
    ax.set_xlabel("Схід (м)")
    ax.set_ylabel("Північ (м)")
    ax.set_zlabel("Вгору (м)")
    ax.view_init(elev=28, azim=42)
    ax.legend(loc="upper left")

    out = BytesIO()
    fig.tight_layout()
    fig.savefig(out, format="png")
    plt.close(fig)
    return out.getvalue()


def export_full_report_pdf_bytes(
    metrics: dict[str, float],
    df_gps: pd.DataFrame,
    df_imu: pd.DataFrame,
    source_label: str = "",
    speed_unit: str = "м/с",
    ai_analysis_text: str | None = None,
) -> bytes:
    """Build a PDF report that includes metadata, metrics, and data summaries."""
    try:
        import matplotlib.font_manager as fm
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError as exc:
        raise ImportError("PDF export requires 'reportlab'. Install it with: pip install reportlab") from exc

    # Register Unicode font to render Cyrillic correctly in PDF.
    try:
        font_path = fm.findfont("DejaVu Sans")
        pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
        base_font = "DejaVuSans"
    except Exception:
        base_font = "Helvetica"

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name="UaTitle", parent=styles["Title"], fontName=base_font)
    heading2_style = ParagraphStyle(name="UaHeading2", parent=styles["Heading2"], fontName=base_font)
    heading3_style = ParagraphStyle(name="UaHeading3", parent=styles["Heading3"], fontName=base_font)
    normal_style = ParagraphStyle(name="UaNormal", parent=styles["Normal"], fontName=base_font)
    italic_style = ParagraphStyle(name="UaItalic", parent=styles["Italic"], fontName=base_font)
    story = []

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph("Звіт з аналізу польоту", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Сформовано: {generated_at}", normal_style))
    story.append(Paragraph(f"Джерело: {source_label or 'Н/Д'}", normal_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Метрики", heading2_style))
    metric_rows = [["Метрика", "Значення"]]
    for key, value in metrics.items():
        display_key = _localize_metric_key(str(key))
        if isinstance(value, (int, float)):
            metric_rows.append([display_key, f"{float(value):.4f}"])
        else:
            metric_rows.append([display_key, str(value)])

    metric_table = Table(metric_rows, colWidths=[110 * mm, 60 * mm], repeatRows=1)
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EEF8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), base_font),
                ("FONTNAME", (0, 1), (-1, -1), base_font),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Зведення телеметрії", heading2_style))
    summary_rows = [["Поле", "Значення"]] + _build_summary_rows(df_gps, df_imu)
    summary_table = Table(summary_rows, colWidths=[80 * mm, 90 * mm], repeatRows=1)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F4F8")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), base_font),
                ("FONTNAME", (0, 1), (-1, -1), base_font),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Примітка: швидкість, отримана інтегруванням акселерометра, накопичує дрейф з часом і придатна насамперед для трендового аналізу.",
            italic_style,
        )
    )

    if ai_analysis_text and ai_analysis_text.strip():
        story.append(Spacer(1, 12))
        story.append(Paragraph("AI-аналіз", heading2_style))
        for line in ai_analysis_text.strip().splitlines():
            clean = escape(_strip_ai_markdown(line))
            if clean:
                story.append(Paragraph(clean, normal_style))
            else:
                story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Приклади графіків", heading2_style))

    speed_png = _build_speed_comparison_png_bytes(df_gps, df_imu, speed_unit=speed_unit)
    if speed_png is not None:
        story.append(Paragraph("Порівняння дрейфу швидкості", heading3_style))
        story.append(RLImage(BytesIO(speed_png), width=175 * mm, height=73 * mm))
        story.append(Spacer(1, 8))

    traj_png = _build_trajectory_sample_png_bytes(df_gps)
    if traj_png is not None:
        story.append(Paragraph("3D-траєкторія (приклад ракурсу)", heading3_style))
        story.append(RLImage(BytesIO(traj_png), width=170 * mm, height=108 * mm))

    doc.build(story)
    return buffer.getvalue()
