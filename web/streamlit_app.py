import logging
import os
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.parsers.base import ParseResult, ParseStatus
from app.parsers.binary import BinaryDataParser
from app.services.ai_assistant import GeminiFlightAssistant
from app.services.analyzer import AnalysisService
from app.services.data_quality import MetricQualityAssessment, assess_metric_quality
from app.services.event_detector import FlightEventReport
from app.services.incident_report import IncidentReport
from app.services.pipeline import (
    ProcessedTelemetry,
    collect_metrics,
    filter_gps_by_timeframe,
    list_local_bin_files,
    parse_result_from_path,
    parse_uploaded_bin_result,
    prepare_telemetry_frames,
)
from app.services.reporting import export_full_report_pdf_bytes, export_metrics_csv_bytes
from visualization.flight_plotter import plot_flight_path_3d

st.set_page_config(page_title="Аналізатор польотних даних", layout="wide")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SidebarState:
    data: dict[str, pd.DataFrame] | None
    parse_result: ParseResult | None
    source_label: str
    imu_index: int
    color_by: str
    speed_unit: str
    show_ground: bool


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


def _format_metric(value: float) -> str:
    if value is None:
        return "Н/Д"
    if isinstance(value, (float, int)):
        return f"{value:.2f}"
    return str(value)


def _convert_speed_value(value: float, speed_unit: str) -> float:
    if speed_unit == "км/год":
        return value * 3.6
    return value


def _format_metrics_for_display(metrics: dict[str, float], speed_unit: str) -> dict[str, float]:
    display_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        localized_key = METRIC_LABELS_UA.get(key, key)
        if "(m/s)" in key:
            display_key = localized_key.replace("(м/с)", f"({speed_unit})")
            display_metrics[display_key] = _convert_speed_value(float(value), speed_unit)
            continue
        display_metrics[localized_key] = value
    return display_metrics


def _render_summary_tab(
    metrics: dict[str, float],
    speed_unit: str,
    metric_quality: dict[str, MetricQualityAssessment],
) -> None:
    display_metrics = _format_metrics_for_display(metrics, speed_unit)
    cols = st.columns(5)
    for idx, ((key, value), source_key) in enumerate(
        zip(display_metrics.items(), metrics, strict=True)
    ):
        assessment = metric_quality[source_key]
        confidence = assessment.confidence.value.upper()
        cols[idx % 5].metric(
            f"{key} [{confidence}]",
            _format_metric(value),
            help="; ".join(assessment.reasons)
            or (f"Довіра визначена за якістю потоку {assessment.source_stream}."),
        )


def _render_events_tab(
    event_report: FlightEventReport,
    incident_report: IncidentReport,
) -> None:
    summary = event_report.to_dict()["summary"]
    st.caption(
        f"Профіль: {summary['vehicle_profile'].upper()} | "
        f"Прошивка: {summary['firmware'] or 'невідома'}"
    )
    cols = st.columns(4)
    cols[0].metric("Події", summary["event_count"])
    cols[1].metric("Польотні сегменти", summary["flight_segment_count"])
    cols[2].metric("Критичні", summary["critical_count"])
    cols[3].metric("Попередження", summary["warning_count"])

    if event_report.segments:
        st.subheader("Сегменти ARM–DISARM")
        st.dataframe(
            pd.DataFrame(segment.to_dict() for segment in event_report.segments),
            width="stretch",
            hide_index=True,
        )

    if not event_report.events:
        st.info("У журналі не знайдено підтримуваних польотних подій.")
        return

    st.subheader("Часова шкала")
    event_rows = [
        {
            "TimeUS": event.time_us,
            "Відносний час, с": round(event.relative_time_s, 3),
            "Тип": event.kind.value,
            "Категорія": event.category.value,
            "Рівень": event.severity.value.upper(),
            "Подія": event.title,
            "Джерело": event.source,
            "Деталі": event.details,
        }
        for event in event_report.events
    ]
    st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)
    for warning in event_report.warnings:
        st.warning(warning)

    st.subheader("Incident report")
    if not incident_report.incidents:
        st.success("Підтримуваних інцидентів або failsafe не виявлено.")
    else:
        incident_rows = [
            {
                "№": incident.index,
                "Тип": incident.incident_type.value,
                "Сегмент": incident.segment_index,
                "Статус": incident.status.value.upper(),
                "Довіра": incident.confidence.value.upper(),
                "Реакція, с": incident.response_latency_s,
                "До землі, с": incident.time_to_ground_s,
                "До DISARM, с": incident.time_to_disarm_s,
                "Висновок": incident.narrative,
            }
            for incident in incident_report.incidents
        ]
        st.dataframe(pd.DataFrame(incident_rows), width="stretch", hide_index=True)


def _render_export_section(
    metrics: dict[str, float],
    df_gps: pd.DataFrame,
    df_imu: pd.DataFrame,
    source_label: str,
    speed_unit: str,
    ai_analysis_text: str | None,
) -> None:
    with st.expander("Експорт", expanded=False):
        left, right = st.columns(2)

        csv_bytes = export_metrics_csv_bytes(metrics)
        left.download_button(
            label="Метрики CSV",
            data=csv_bytes,
            file_name="flight_metrics.csv",
            mime="text/csv",
            width="content",
        )

        try:
            pdf_bytes = export_full_report_pdf_bytes(
                metrics=metrics,
                df_gps=df_gps,
                df_imu=df_imu,
                source_label=source_label,
                speed_unit=speed_unit,
                ai_analysis_text=ai_analysis_text,
            )
            right.download_button(
                label="Повний звіт PDF",
                data=pdf_bytes,
                file_name="flight_report.pdf",
                mime="application/pdf",
                width="content",
            )
        except ImportError as exc:
            st.info(str(exc))


def _render_timeframe_filter(df_gps: pd.DataFrame) -> tuple[float, float] | None:
    if "TimeUS" not in df_gps.columns or len(df_gps) <= 1:
        return None

    time_us = pd.to_numeric(df_gps["TimeUS"], errors="coerce")
    if time_us.isna().all():
        return None

    relative_seconds = (time_us - float(time_us.iloc[0])) / 1e6
    max_seconds = float(relative_seconds.max())
    if max_seconds <= 0:
        return None

    return st.slider(
        "Часовий інтервал графіків (с)",
        min_value=0.0,
        max_value=max_seconds,
        value=(0.0, max_seconds),
        key="sidebar_timeframe",
    )


def _render_trajectory_tab(
    df_gps: pd.DataFrame,
    color_by: str,
    speed_unit: str,
    show_ground: bool,
    timeframe_window: tuple[float, float] | None,
) -> None:
    stable_alt_origin = None
    if "Alt" in df_gps.columns:
        alt_series = pd.to_numeric(df_gps["Alt"], errors="coerce").dropna()
        if not alt_series.empty:
            stable_alt_origin = float(alt_series.iloc[0])

    filtered_df = df_gps
    if timeframe_window is not None:
        filtered_df = filter_gps_by_timeframe(df_gps, timeframe_window[0], timeframe_window[1])

    if len(filtered_df) < 2:
        st.warning("У вибраному інтервалі недостатньо точок для побудови траєкторії.")
        return

    try:
        fig = plot_flight_path_3d(
            filtered_df,
            output_html=None,
            auto_open=False,
            color_by=color_by,
            speed_unit=speed_unit,
            show_ground=show_ground,
            terrain_altitude_origin=stable_alt_origin,
        )
        st.plotly_chart(fig, width="stretch")
    except ValueError as exc:
        st.error(f"Не вдалося побудувати 3D-траєкторію: {exc}")


def _render_dataframes_tab(df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> None:
    st.subheader("GPS-дані")
    st.dataframe(df_gps, width="stretch", height=300)

    st.subheader("IMU-дані")
    if df_imu.empty:
        st.warning("Для вибраного індексу модуля IMU немає рядків.")
    else:
        st.dataframe(df_imu, width="stretch", height=300)


def _render_speed_drift_tab(
    df_gps: pd.DataFrame,
    df_imu: pd.DataFrame,
    speed_unit: str,
    timeframe_window: tuple[float, float] | None,
) -> None:
    if "Spd" not in df_gps.columns or "TimeUS" not in df_gps.columns:
        st.warning("Порівняння швидкостей недоступне: для GPS потрібні колонки 'Spd' і 'TimeUS'.")
        return

    if "VelAccNorm" not in df_imu.columns or "TimeUS" not in df_imu.columns:
        st.warning(
            "Швидкість з акселерометра недоступна: для IMU потрібні Acc/Gyro поля та TimeUS."
        )
        return

    gps = df_gps.copy()
    imu = df_imu.copy()
    gps["TimeUS"] = pd.to_numeric(gps["TimeUS"], errors="coerce")
    gps["Spd"] = pd.to_numeric(gps["Spd"], errors="coerce")
    imu["TimeUS"] = pd.to_numeric(imu["TimeUS"], errors="coerce")
    imu["VelAccNorm"] = pd.to_numeric(imu["VelAccNorm"], errors="coerce")

    gps = gps.dropna(subset=["TimeUS", "Spd"]).sort_values("TimeUS")
    imu = imu.dropna(subset=["TimeUS", "VelAccNorm"]).sort_values("TimeUS")
    if gps.empty or imu.empty:
        st.warning("Недостатньо узгоджених даних для відображення порівняння дрейфу.")
        return

    if timeframe_window is not None:
        start_s, end_s = timeframe_window
        gps = filter_gps_by_timeframe(gps, start_s, end_s)
        start_us = float(gps["TimeUS"].iloc[0]) if not gps.empty else None
        if start_us is not None:
            imu_rel = (imu["TimeUS"] - start_us) / 1e6
            lower = min(start_s, end_s)
            upper = max(start_s, end_s)
            imu = imu.loc[(imu_rel >= lower) & (imu_rel <= upper)]
        if gps.empty or imu.empty:
            st.warning("У вибраному інтервалі немає перетину сигналів швидкості GPS та IMU.")
            return

    t0 = min(float(gps["TimeUS"].iloc[0]), float(imu["TimeUS"].iloc[0]))
    gps_t = (gps["TimeUS"] - t0) / 1e6
    imu_t = (imu["TimeUS"] - t0) / 1e6

    unit_factor = 3.6 if speed_unit == "км/год" else 1.0
    gps_speed = gps["Spd"] * unit_factor
    imu_speed = imu["VelAccNorm"] * unit_factor

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gps_t,
            y=gps_speed,
            mode="lines",
            name=f"Швидкість GPS ({speed_unit})",
            line={"color": "#1f77b4", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=imu_t,
            y=imu_speed,
            mode="lines",
            name=f"Швидкість з акселерометра ({speed_unit}) [із дрейфом]",
            line={"color": "#d62728", "width": 2},
        )
    )
    fig.update_layout(
        title="Порівняння швидкостей: GPS vs інтегрування акселерометра",
        xaxis_title="Час (с)",
        yaxis_title=f"Швидкість ({speed_unit})",
        legend={"x": 0.01, "y": 0.99, "yanchor": "top"},
        margin={"l": 24, "r": 12, "b": 24, "t": 44},
    )

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Примітка: швидкість, отримана інтегруванням акселерометра, накопичує дрейф з часом."
    )


def _render_ai_tab(
    metrics: dict[str, float],
    df_gps: pd.DataFrame,
    df_imu: pd.DataFrame,
    gemini_api_key: str,
) -> None:
    st.subheader("AI-аналіз польоту (Gemini)")
    st.caption("Звіт формується українською мовою на основі метрик польоту та зведення телеметрії.")

    if not gemini_api_key:
        st.info(
            "Вкажіть Gemini API ключ у бічній панелі (або через змінну середовища GEMINI_API_KEY)."
        )
        return

    if st.button("Згенерувати AI-аналіз", width="stretch"):
        try:
            assistant = GeminiFlightAssistant(api_key=gemini_api_key, model_name="gemini-2.5-flash")
            with st.spinner("Генерується аналітичний звіт..."):
                analysis = assistant.generate_analysis(
                    metrics=metrics,
                    df_gps=df_gps,
                    df_imu=df_imu,
                )
            st.session_state["ai_analysis"] = analysis
            st.rerun()
        except (ValueError, ImportError, RuntimeError) as exc:
            LOGGER.exception("AI analysis generation failed")
            st.error(f"Не вдалося згенерувати AI-аналіз: {exc}")
            return

    existing = st.session_state.get("ai_analysis")
    if existing:
        st.markdown(existing)


def _load_data_from_sidebar(parser: BinaryDataParser) -> SidebarState:
    source_mode = st.radio("Джерело даних", ["Локальний файл", "Завантажити BIN"], index=0)

    data = st.session_state.get("loaded_data")
    parse_result = st.session_state.get("loaded_parse_result")
    source_label = st.session_state.get("loaded_source_label", "")

    if source_mode == "Локальний файл":
        data_files = list_local_bin_files("data")
        if not data_files:
            st.warning("У директорії data/ не знайдено .BIN файлів.")
        else:
            selected = st.selectbox("Оберіть BIN-файл", data_files, format_func=lambda p: str(p))
            if st.button("Завантажити дані", width="stretch"):
                try:
                    parse_result = parse_result_from_path(parser, str(selected))
                    if parse_result.status is ParseStatus.REJECTED:
                        raise ValueError(parse_result.error or "BIN file could not be decoded")
                    data = parse_result.dataframes
                    source_label = str(selected)
                    st.session_state["loaded_data"] = data
                    st.session_state["loaded_parse_result"] = parse_result
                    st.session_state["loaded_source_label"] = source_label
                    st.session_state.pop("ai_analysis", None)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    LOGGER.exception("Failed to parse local BIN file")
                    st.error(f"Не вдалося завантажити локальний файл: {exc}")
    else:
        uploaded = st.file_uploader("Завантажте BIN-файл", type=["bin", "BIN"])
        if uploaded is not None:
            if st.button("Завантажити дані", width="stretch"):
                try:
                    parse_result = parse_uploaded_bin_result(parser, uploaded)
                    if parse_result.status is ParseStatus.REJECTED:
                        raise ValueError(parse_result.error or "BIN file could not be decoded")
                    data = parse_result.dataframes
                    source_label = uploaded.name
                    st.session_state["loaded_data"] = data
                    st.session_state["loaded_parse_result"] = parse_result
                    st.session_state["loaded_source_label"] = source_label
                    st.session_state.pop("ai_analysis", None)
                except (OSError, ValueError) as exc:
                    LOGGER.exception("Failed to parse uploaded BIN file")
                    st.error(f"Не вдалося обробити завантажений файл: {exc}")

    if source_label:
        st.caption(f"Поточний завантажений файл: {source_label}")

    if st.button("Очистити завантажені дані", width="stretch"):
        st.session_state.pop("loaded_data", None)
        st.session_state.pop("loaded_parse_result", None)
        st.session_state.pop("loaded_source_label", None)
        st.session_state.pop("ai_analysis", None)
        data = None
        parse_result = None
        source_label = ""

    st.header("Фільтри")
    imu_index = st.number_input("Індекс модуля IMU", min_value=0, max_value=9, value=0, step=1)
    speed_unit = st.selectbox("Одиниця швидкості", ["м/с", "км/год"], index=0)
    color_by = st.selectbox("Режим кольору 3D", ["combined", "ground", "vertical", "time"], index=0)
    show_ground = st.checkbox("Показувати поверхню землі", value=True)

    return SidebarState(
        data=data,
        parse_result=parse_result,
        source_label=source_label,
        imu_index=int(imu_index),
        color_by=color_by,
        speed_unit=speed_unit,
        show_ground=show_ground,
    )


def main() -> None:
    st.title("Аналізатор польотних даних")
    st.caption("Інтерактивний аналіз телеметрії з логів ArduPilot BIN")

    parser = BinaryDataParser()
    analyzer = AnalysisService()

    with st.sidebar:
        st.header("Вхідні дані")
        state = _load_data_from_sidebar(parser)

    if state.data is None:
        st.info("Оберіть джерело і натисніть 'Завантажити дані', щоб почати.")
        return

    if state.source_label:
        st.success(f"Завантажено: {state.source_label}")

    if state.parse_result is not None:
        integrity_label = state.parse_result.status.value.upper()
        if state.parse_result.status is ParseStatus.COMPLETE:
            st.success(f"Цілісність даних: {integrity_label}")
        else:
            st.warning(f"Цілісність даних: {integrity_label}")
            for warning in state.parse_result.warnings:
                st.warning(warning)

    try:
        telemetry: ProcessedTelemetry = prepare_telemetry_frames(
            analyzer, state.data, imu_index=state.imu_index
        )

    except (ValueError, KeyError) as exc:
        LOGGER.exception("Telemetry preparation failed")
        st.error(f"Не вдалося обробити телеметрію: {exc}")
        return

    df_gps = telemetry.df_gps
    df_imu = telemetry.df_imu

    if df_gps.empty:
        st.error("GPS-дані відсутні або порожні у цьому логу.")
        return

    quality_report = telemetry.quality_report
    if quality_report.status.value == "good":
        st.success("Якість телеметрії: GOOD")
    else:
        st.warning(f"Якість телеметрії: {quality_report.status.value.upper()}")

    quality_rows = []
    for stream_name, stream_report in quality_report.streams.items():
        quality_rows.append(
            {
                "Потік": stream_name,
                "Статус": stream_report.status.value.upper(),
                "Усього": stream_report.total_records,
                "Валідні": stream_report.valid_records,
                "Відхилені": stream_report.rejected_records,
                "Timestamp outliers": stream_report.timestamp_outliers,
                "Дублікати": stream_report.duplicate_timestamps,
                "Розриви": stream_report.gap_count,
                "Value outliers": stream_report.value_outliers,
            }
        )
    with st.expander("Якість і очищення телеметрії", expanded=False):
        st.dataframe(pd.DataFrame(quality_rows), width="stretch", hide_index=True)
        for stream_report in quality_report.streams.values():
            for warning in stream_report.warnings:
                st.warning(warning)

    with st.sidebar:
        st.subheader("Часовий інтервал")
        timeframe_window = _render_timeframe_filter(df_gps)
        st.header("AI-асистент")
        gemini_api_key = st.text_input(
            "Gemini API ключ",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Ключ не зберігається в коді. Також можна використати змінну середовища GEMINI_API_KEY.",
        ).strip()

    metrics = collect_metrics(analyzer, df_gps, df_imu)
    metric_quality = assess_metric_quality(metrics, quality_report)

    tabs = st.tabs(
        [
            "Підсумок",
            "3D-траєкторія",
            "Дрейф швидкості",
            "Таблиці даних",
            "AI-аналіз",
            "Події",
        ]
    )

    with tabs[0]:
        _render_summary_tab(metrics, state.speed_unit, metric_quality)
        _render_export_section(
            metrics,
            df_gps,
            df_imu,
            state.source_label,
            state.speed_unit,
            st.session_state.get("ai_analysis"),
        )

    with tabs[1]:
        _render_trajectory_tab(
            df_gps,
            state.color_by,
            state.speed_unit,
            state.show_ground,
            timeframe_window,
        )

    with tabs[2]:
        _render_speed_drift_tab(df_gps, df_imu, state.speed_unit, timeframe_window)

    with tabs[3]:
        _render_dataframes_tab(df_gps, df_imu)

    with tabs[4]:
        _render_ai_tab(metrics, df_gps, df_imu, gemini_api_key)

    with tabs[5]:
        _render_events_tab(telemetry.event_report, telemetry.incident_report)


if __name__ == "__main__":
    main()
