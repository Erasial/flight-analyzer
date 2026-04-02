import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.utils import wgs84_to_enu
from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from visualization.flight_plotter import plot_flight_path_3d


st.set_page_config(page_title="Flight Data Analyzer", layout="wide")


def _format_metric(value: float) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, int)):
        return f"{value:.2f}"
    return str(value)


def _collect_metrics(analyzer: AnalysisService, df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> dict:
    max_acceleration = analyzer.get_max_acceleration(df_imu) if df_imu is not None and not df_imu.empty else {}

    return {
        "Flight Duration (s)": analyzer.get_flight_duration(df_gps),
        "Distance Traveled (m)": analyzer.get_distance_traveled(df_gps),
        "Max Horizontal Speed (m/s)": analyzer.get_max_horizontal_speed(df_gps),
        "Max Vertical Speed (m/s)": analyzer.get_max_vertical_speed(df_gps),
        "Max Altitude (m)": analyzer.get_max_altitude(df_gps),
        "Max Acc X (m/s^2)": max_acceleration.get("AccX", 0.0),
        "Max Acc Y (m/s^2)": max_acceleration.get("AccY", 0.0),
        "Max Acc Z (m/s^2)": max_acceleration.get("AccZ", 0.0),
        "GPS Sample Rate (Hz)": analyzer.get_sample_rate(df_gps),
        "IMU Sample Rate (Hz)": analyzer.get_sample_rate(df_imu),
    }


def _parse_from_path(parser: BinaryDataParser, file_path: str) -> dict:
    return parser.parse(file_path)


def _parse_uploaded(parser: BinaryDataParser, uploaded_file) -> dict:
    suffix = Path(uploaded_file.name).suffix or ".BIN"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name
        return parser.parse(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> None:
    st.title("Flight Data Analyzer")
    st.caption("Interactive telemetry analysis from ArduPilot BIN logs")

    parser = BinaryDataParser()
    analyzer = AnalysisService()

    with st.sidebar:
        st.header("Inputs")
        source_mode = st.radio("Data source", ["Local file", "Upload BIN"], index=0)

        data = None
        source_label = ""

        if source_mode == "Local file":
            data_files = sorted(Path("data").glob("*.BIN"))
            if not data_files:
                st.warning("No .BIN files found in the data/ directory.")
            else:
                selected = st.selectbox("Select BIN file", data_files, format_func=lambda p: str(p))
                source_label = str(selected)
                if st.button("Load Data", use_container_width=True):
                    data = _parse_from_path(parser, str(selected))

        else:
            uploaded = st.file_uploader("Upload a BIN file", type=["bin", "BIN"])
            if uploaded is not None:
                source_label = uploaded.name
                if st.button("Load Data", use_container_width=True):
                    data = _parse_uploaded(parser, uploaded)

        st.header("Filters")
        imu_index = st.number_input("IMU Module Index", min_value=0, max_value=9, value=0, step=1)
        color_by = st.selectbox("3D Color Mode", ["combined", "ground", "vertical"], index=0)

    if data is None:
        st.info("Select a source and click 'Load Data' to begin.")
        return

    if source_label:
        st.success(f"Loaded: {source_label}")

    df_gps = data.get("GPS", pd.DataFrame())
    df_imu = data.get("IMU", pd.DataFrame())

    if df_gps.empty:
        st.error("GPS data is missing or empty in this log file.")
        return

    try:
        df_gps = analyzer.filter_gps_low_quality_samples(df_gps)
        df_gps = wgs84_to_enu(df_gps)

        if not df_imu.empty:
            df_imu = analyzer.filter_imu_module(df_imu, imu_index=int(imu_index))

    except Exception as exc:
        st.error(f"Failed to process telemetry data: {exc}")
        return

    tabs = st.tabs(["Summary", "3D Trajectory", "DataFrames"])

    with tabs[0]:
        metrics = _collect_metrics(analyzer, df_gps, df_imu)
        cols = st.columns(5)
        for idx, (key, value) in enumerate(metrics.items()):
            cols[idx % 5].metric(key, _format_metric(value))

    with tabs[1]:
        try:
            fig = plot_flight_path_3d(df_gps, output_html=None, auto_open=False, color_by=color_by)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to render 3D trajectory: {exc}")

    with tabs[2]:
        st.subheader("GPS Data")
        st.dataframe(df_gps, use_container_width=True, height=300)

        st.subheader("IMU Data")
        if df_imu.empty:
            st.warning("No IMU rows available for the selected module index.")
        else:
            st.dataframe(df_imu, use_container_width=True, height=300)


if __name__ == "__main__":
    main()
