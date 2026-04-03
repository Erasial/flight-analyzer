from dataclasses import dataclass
import logging
import os

import pandas as pd
import streamlit as st

from app.parsers.binary import BinaryDataParser
from app.services.ai_assistant import GeminiFlightAssistant
from app.services.analyzer import AnalysisService
from app.services.pipeline import (
    collect_metrics,
    list_local_bin_files,
    parse_data_from_path,
    parse_uploaded_bin,
    ProcessedTelemetry,
    prepare_telemetry_frames,
)
from visualization.flight_plotter import plot_flight_path_3d


st.set_page_config(page_title="Flight Data Analyzer", layout="wide")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SidebarState:
    data: dict[str, pd.DataFrame] | None
    source_label: str
    imu_index: int
    color_by: str
    speed_unit: str
    gemini_api_key: str


def _format_metric(value: float) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, int)):
        return f"{value:.2f}"
    return str(value)


def _convert_speed_value(value: float, speed_unit: str) -> float:
    if speed_unit == "km/h":
        return value * 3.6
    return value


def _format_metrics_for_display(metrics: dict[str, float], speed_unit: str) -> dict[str, float]:
    display_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if "(m/s)" in key:
            display_key = key.replace("(m/s)", f"({speed_unit})")
            display_metrics[display_key] = _convert_speed_value(float(value), speed_unit)
            continue
        display_metrics[key] = value
    return display_metrics


def _render_summary_tab(metrics: dict[str, float], speed_unit: str) -> None:
    display_metrics = _format_metrics_for_display(metrics, speed_unit)
    cols = st.columns(5)
    for idx, (key, value) in enumerate(display_metrics.items()):
        cols[idx % 5].metric(key, _format_metric(value))


def _render_trajectory_tab(df_gps: pd.DataFrame, color_by: str, speed_unit: str) -> None:
    try:
        fig = plot_flight_path_3d(
            df_gps,
            output_html=None,
            auto_open=False,
            color_by=color_by,
            speed_unit=speed_unit,
        )
        st.plotly_chart(fig, width="stretch")
    except ValueError as exc:
        st.error(f"Unable to render 3D trajectory: {exc}")


def _render_dataframes_tab(df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> None:
    st.subheader("GPS Data")
    st.dataframe(df_gps, width="stretch", height=300)

    st.subheader("IMU Data")
    if df_imu.empty:
        st.warning("No IMU rows available for the selected module index.")
    else:
        st.dataframe(df_imu, width="stretch", height=300)


def _render_ai_tab(
    metrics: dict[str, float],
    df_gps: pd.DataFrame,
    df_imu: pd.DataFrame,
    state: SidebarState,
) -> None:
    st.subheader("AI Flight Analysis (Gemini)")
    st.caption("The report is generated in English from flight metrics and telemetry summary.")

    if not state.gemini_api_key:
        st.info("Provide a Gemini API key in the sidebar (or via GEMINI_API_KEY environment variable).")
        return

    if st.button("Generate AI Analysis", width="stretch"):
        try:
            assistant = GeminiFlightAssistant(api_key=state.gemini_api_key, model_name="gemini-2.5-flash")
            with st.spinner("Generating analysis report..."):
                analysis = assistant.generate_analysis(
                    metrics=metrics,
                    df_gps=df_gps,
                    df_imu=df_imu,
                )
            st.session_state["ai_analysis"] = analysis
        except (ValueError, ImportError, RuntimeError) as exc:
            LOGGER.exception("AI analysis generation failed")
            st.error(f"Failed to generate AI analysis: {exc}")
            return

    existing = st.session_state.get("ai_analysis")
    if existing:
        st.markdown(existing)


def _load_data_from_sidebar(parser: BinaryDataParser) -> SidebarState:
    source_mode = st.radio("Data source", ["Local file", "Upload BIN"], index=0)

    data = st.session_state.get("loaded_data")
    source_label = st.session_state.get("loaded_source_label", "")

    if source_mode == "Local file":
        data_files = list_local_bin_files("data")
        if not data_files:
            st.warning("No .BIN files found in the data/ directory.")
        else:
            selected = st.selectbox("Select BIN file", data_files, format_func=lambda p: str(p))
            if st.button("Load Data", width="stretch"):
                try:
                    data = parse_data_from_path(parser, str(selected))
                    source_label = str(selected)
                    st.session_state["loaded_data"] = data
                    st.session_state["loaded_source_label"] = source_label
                    st.session_state.pop("ai_analysis", None)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    LOGGER.exception("Failed to parse local BIN file")
                    st.error(f"Failed to load local file: {exc}")
    else:
        uploaded = st.file_uploader("Upload a BIN file", type=["bin", "BIN"])
        if uploaded is not None:
            if st.button("Load Data", width="stretch"):
                try:
                    data = parse_uploaded_bin(parser, uploaded)
                    source_label = uploaded.name
                    st.session_state["loaded_data"] = data
                    st.session_state["loaded_source_label"] = source_label
                    st.session_state.pop("ai_analysis", None)
                except (OSError, ValueError) as exc:
                    LOGGER.exception("Failed to parse uploaded BIN file")
                    st.error(f"Failed to load uploaded file: {exc}")

    if source_label:
        st.caption(f"Current loaded file: {source_label}")

    if st.button("Clear Loaded Data", width="stretch"):
        st.session_state.pop("loaded_data", None)
        st.session_state.pop("loaded_source_label", None)
        st.session_state.pop("ai_analysis", None)
        data = None
        source_label = ""

    st.header("Filters")
    imu_index = st.number_input("IMU Module Index", min_value=0, max_value=9, value=0, step=1)
    speed_unit = st.selectbox("Speed Unit", ["m/s", "km/h"], index=0)
    color_by = st.selectbox("3D Color Mode", ["combined", "ground", "vertical"], index=0)

    st.header("AI Assistant")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="The key is not stored in code. You can also set GEMINI_API_KEY in your environment.",
    )

    return SidebarState(
        data=data,
        source_label=source_label,
        imu_index=int(imu_index),
        color_by=color_by,
        speed_unit=speed_unit,
        gemini_api_key=gemini_api_key.strip(),
    )


def main() -> None:
    st.title("Flight Data Analyzer")
    st.caption("Interactive telemetry analysis from ArduPilot BIN logs")

    parser = BinaryDataParser()
    analyzer = AnalysisService()

    with st.sidebar:
        st.header("Inputs")
        state = _load_data_from_sidebar(parser)

    if state.data is None:
        st.info("Select a source and click 'Load Data' to begin.")
        return

    if state.source_label:
        st.success(f"Loaded: {state.source_label}")

    try:
        telemetry: ProcessedTelemetry = prepare_telemetry_frames(analyzer, state.data, imu_index=state.imu_index)

    except (ValueError, KeyError) as exc:
        LOGGER.exception("Telemetry preparation failed")
        st.error(f"Failed to process telemetry data: {exc}")
        return

    df_gps = telemetry.df_gps
    df_imu = telemetry.df_imu

    if df_gps.empty:
        st.error("GPS data is missing or empty in this log file.")
        return

    metrics = collect_metrics(analyzer, df_gps, df_imu)

    tabs = st.tabs(["Summary", "3D Trajectory", "DataFrames", "AI Analysis"])

    with tabs[0]:
        _render_summary_tab(metrics, state.speed_unit)

    with tabs[1]:
        _render_trajectory_tab(df_gps, state.color_by, state.speed_unit)

    with tabs[2]:
        _render_dataframes_tab(df_gps, df_imu)

    with tabs[3]:
        _render_ai_tab(metrics, df_gps, df_imu, state)


if __name__ == "__main__":
    main()
