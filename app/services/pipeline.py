import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.utils import wgs84_to_enu
from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService


@dataclass(frozen=True)
class ProcessedTelemetry:
    df_gps: pd.DataFrame
    df_imu: pd.DataFrame
    df_att: pd.DataFrame = field(default_factory=pd.DataFrame)


def list_local_bin_files(data_dir: str = "data") -> list[Path]:
    return sorted(Path(data_dir).glob("*.BIN"))


def parse_data_from_path(parser: BinaryDataParser, file_path: str) -> dict[str, pd.DataFrame]:
    return parser.parse(file_path)


def parse_uploaded_bin(parser: BinaryDataParser, uploaded_file: Any) -> dict[str, pd.DataFrame]:
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


def prepare_telemetry_frames(
    analyzer: AnalysisService,
    dataframes: dict[str, pd.DataFrame],
    imu_index: int = 0,
) -> ProcessedTelemetry:
    df_gps = dataframes.get("GPS", pd.DataFrame())
    df_imu = dataframes.get("IMU", pd.DataFrame())
    df_att = dataframes.get("ATT", pd.DataFrame())

    if df_gps.empty:
        return ProcessedTelemetry(df_gps=df_gps, df_imu=df_imu, df_att=df_att)

    # Improved GPS processing: filter outliers in speed/altitude if necessary
    df_gps = analyzer.filter_gps_low_quality_samples(df_gps)
    df_gps = analyzer.filter_outliers(df_gps, 'Alt', threshold=5.0) # ArduPilot Alt can jump
    df_gps = wgs84_to_enu(df_gps)

    if not df_imu.empty:
        df_imu = analyzer.filter_imu_module(df_imu, imu_index=imu_index)
        # Smooth IMU data as it is often noisy
        for col in ['AccX', 'AccY', 'AccZ']:
            if col in df_imu.columns:
                df_imu = analyzer.smooth_signal(df_imu, col, window=5)

    if not df_att.empty:
        df_att = analyzer.process_attitude(df_att)

    return ProcessedTelemetry(df_gps=df_gps, df_imu=df_imu, df_att=df_att)


def collect_metrics(analyzer: AnalysisService, df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> dict[str, float]:
    max_acceleration = analyzer.get_max_acceleration(df_imu) if not df_imu.empty else {}

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


def filter_gps_by_timeframe(df_gps: pd.DataFrame, start_seconds: float, end_seconds: float) -> pd.DataFrame:
    if df_gps.empty or "TimeUS" not in df_gps.columns:
        return df_gps

    time_us = pd.to_numeric(df_gps["TimeUS"], errors="coerce")
    if time_us.isna().all():
        return df_gps

    start_us = float(time_us.iloc[0])
    relative_seconds = (time_us - start_us) / 1e6
    lower = min(start_seconds, end_seconds)
    upper = max(start_seconds, end_seconds)
    mask = (relative_seconds >= lower) & (relative_seconds <= upper)
    return df_gps.loc[mask].reset_index(drop=True)
