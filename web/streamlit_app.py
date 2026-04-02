from dataclasses import dataclass

import pandas as pd
import streamlit as st

from app.parsers.binary import BinaryDataParser
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


@dataclass(frozen=True)
class SidebarState:
    data: dict[str, pd.DataFrame] | None
    source_label: str
    imu_index: int
    color_by: str


def _format_metric(value: float) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, int)):
        return f"{value:.2f}"
    return str(value)


def _render_summary_tab(analyzer: AnalysisService, df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> None:
    metrics = collect_metrics(analyzer, df_gps, df_imu)
    cols = st.columns(5)
    for idx, (key, value) in enumerate(metrics.items()):
        cols[idx % 5].metric(key, _format_metric(value))


def _render_trajectory_tab(df_gps: pd.DataFrame, color_by: str) -> None:
    try:
        fig = plot_flight_path_3d(df_gps, output_html=None, auto_open=False, color_by=color_by)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Unable to render 3D trajectory: {exc}")


def _render_dataframes_tab(df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> None:
    st.subheader("GPS Data")
    st.dataframe(df_gps, use_container_width=True, height=300)

    st.subheader("IMU Data")
    if df_imu.empty:
        st.warning("No IMU rows available for the selected module index.")
    else:
        st.dataframe(df_imu, use_container_width=True, height=300)


def _load_data_from_sidebar(parser: BinaryDataParser) -> SidebarState:
    source_mode = st.radio("Data source", ["Local file", "Upload BIN"], index=0)

    data = None
    source_label = ""

    if source_mode == "Local file":
        data_files = list_local_bin_files("data")
        if not data_files:
            st.warning("No .BIN files found in the data/ directory.")
        else:
            selected = st.selectbox("Select BIN file", data_files, format_func=lambda p: str(p))
            source_label = str(selected)
            if st.button("Load Data", use_container_width=True):
                data = parse_data_from_path(parser, str(selected))
    else:
        uploaded = st.file_uploader("Upload a BIN file", type=["bin", "BIN"])
        if uploaded is not None:
            source_label = uploaded.name
            if st.button("Load Data", use_container_width=True):
                data = parse_uploaded_bin(parser, uploaded)

    st.header("Filters")
    imu_index = st.number_input("IMU Module Index", min_value=0, max_value=9, value=0, step=1)
    color_by = st.selectbox("3D Color Mode", ["combined", "ground", "vertical"], index=0)

    return SidebarState(data=data, source_label=source_label, imu_index=int(imu_index), color_by=color_by)


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

    except Exception as exc:
        st.error(f"Failed to process telemetry data: {exc}")
        return

    df_gps = telemetry.df_gps
    df_imu = telemetry.df_imu

    if df_gps.empty:
        st.error("GPS data is missing or empty in this log file.")
        return

    tabs = st.tabs(["Summary", "3D Trajectory", "DataFrames"])

    with tabs[0]:
        _render_summary_tab(analyzer, df_gps, df_imu)

    with tabs[1]:
        _render_trajectory_tab(df_gps, state.color_by)

    with tabs[2]:
        _render_dataframes_tab(df_gps, df_imu)


if __name__ == "__main__":
    main()
