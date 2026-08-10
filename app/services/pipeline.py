import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.utils import integrate_velocity_from_imu_trapezoidal, wgs84_to_enu
from app.parsers.base import ParseResult, ParseStatus
from app.parsers.binary import BinaryDataParser, BinaryParseError
from app.services.analyzer import AnalysisService
from app.services.data_quality import DataQualityReport, validate_gps, validate_imu
from app.services.event_detector import FlightEventReport, detect_flight_events


@dataclass(frozen=True)
class ProcessedTelemetry:
    df_gps: pd.DataFrame
    df_imu: pd.DataFrame
    df_att: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality_report: DataQualityReport = field(default_factory=DataQualityReport)
    event_report: FlightEventReport = field(default_factory=FlightEventReport)


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _integrate_imu_velocity(df_imu: pd.DataFrame) -> pd.DataFrame:
    if df_imu.empty:
        return df_imu

    required_base = {"TimeUS", "AccX", "AccY", "AccZ"}
    if not required_base.issubset(set(df_imu.columns)):
        return df_imu

    gyro_x_col = _pick_first_existing(df_imu, ["GyrX", "GyroX", "GX", "p"])
    gyro_y_col = _pick_first_existing(df_imu, ["GyrY", "GyroY", "GY", "q"])
    gyro_z_col = _pick_first_existing(df_imu, ["GyrZ", "GyroZ", "GZ", "r"])
    if gyro_x_col is None or gyro_y_col is None or gyro_z_col is None:
        return df_imu

    imu = df_imu.copy()
    imu["TimeUS"] = pd.to_numeric(imu["TimeUS"], errors="coerce")
    imu["AccX"] = pd.to_numeric(imu["AccX"], errors="coerce")
    imu["AccY"] = pd.to_numeric(imu["AccY"], errors="coerce")
    imu["AccZ"] = pd.to_numeric(imu["AccZ"], errors="coerce")
    imu[gyro_x_col] = pd.to_numeric(imu[gyro_x_col], errors="coerce")
    imu[gyro_y_col] = pd.to_numeric(imu[gyro_y_col], errors="coerce")
    imu[gyro_z_col] = pd.to_numeric(imu[gyro_z_col], errors="coerce")

    valid_mask = (
        imu["TimeUS"].notna()
        & imu["AccX"].notna()
        & imu["AccY"].notna()
        & imu["AccZ"].notna()
        & imu[gyro_x_col].notna()
        & imu[gyro_y_col].notna()
        & imu[gyro_z_col].notna()
    )
    valid = imu.loc[valid_mask].copy()
    if len(valid) < 2:
        return df_imu

    valid = valid.sort_values("TimeUS").reset_index(drop=True)
    time_s = valid["TimeUS"].to_numpy(dtype=float) / 1e6
    acc_body = valid[["AccX", "AccY", "AccZ"]].to_numpy(dtype=float)
    gyro_body = valid[[gyro_x_col, gyro_y_col, gyro_z_col]].to_numpy(dtype=float)

    # Heuristic unit normalization: many logs store gyro in deg/s.
    if float(np.nanmax(np.abs(gyro_body))) > 10.0:
        gyro_body = np.radians(gyro_body)

    integrated = integrate_velocity_from_imu_trapezoidal(
        time_s=time_s,
        acc_body_xyz=acc_body,
        gyro_body_xyz=gyro_body,
        gravity_mps2=9.80665,
    )

    mapped = valid[["TimeUS"]].copy()
    mapped["AccEarthX"] = integrated["acc_earth_x"]
    mapped["AccEarthY"] = integrated["acc_earth_y"]
    mapped["AccEarthZ"] = integrated["acc_earth_z"]
    mapped["VelAccX"] = integrated["vel_earth_x"]
    mapped["VelAccY"] = integrated["vel_earth_y"]
    mapped["VelAccZ"] = integrated["vel_earth_z"]
    mapped["VelAccNorm"] = integrated["vel_norm"]

    imu = imu.merge(mapped, on="TimeUS", how="left")
    return imu


def list_local_bin_files(data_dir: str = "data") -> list[Path]:
    return sorted(Path(data_dir).glob("*.BIN"))


def parse_data_from_path(parser: BinaryDataParser, file_path: str) -> dict[str, pd.DataFrame]:
    return parser.parse(file_path)


def parse_result_from_path(parser: BinaryDataParser, file_path: str) -> ParseResult:
    return parser.parse_with_diagnostics(file_path)


def parse_uploaded_bin(parser: BinaryDataParser, uploaded_file: Any) -> dict[str, pd.DataFrame]:
    result = parse_uploaded_bin_result(parser, uploaded_file)
    if result.status is ParseStatus.REJECTED:
        raise BinaryParseError(result.error or "BIN file could not be decoded")
    return result.dataframes


def parse_uploaded_bin_result(parser: BinaryDataParser, uploaded_file: Any) -> ParseResult:
    suffix = Path(uploaded_file.name).suffix or ".BIN"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name
        return parser.parse_with_diagnostics(temp_path)
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
    event_report = detect_flight_events(dataframes)

    gps_validation = validate_gps(df_gps)
    df_gps = gps_validation.dataframe

    imu_validation = validate_imu(pd.DataFrame())
    if not df_imu.empty:
        df_imu = analyzer.filter_imu_module(df_imu, imu_index=imu_index)
        imu_validation = validate_imu(df_imu)
        df_imu = imu_validation.dataframe

    quality_report = DataQualityReport(
        streams={
            "GPS": gps_validation.report,
            "IMU": imu_validation.report,
        }
    )
    if df_gps.empty:
        return ProcessedTelemetry(
            df_gps=df_gps,
            df_imu=df_imu,
            df_att=df_att,
            quality_report=quality_report,
            event_report=event_report,
        )

    df_gps = analyzer.filter_outliers(df_gps, "Alt", threshold=5.0)
    df_gps = wgs84_to_enu(df_gps)

    if not df_imu.empty:
        # Smooth IMU data as it is often noisy
        for col in ["AccX", "AccY", "AccZ"]:
            if col in df_imu.columns:
                df_imu = analyzer.smooth_signal(df_imu, col, window=5)
        df_imu = _integrate_imu_velocity(df_imu)

    if not df_att.empty:
        df_att = analyzer.process_attitude(df_att)

    return ProcessedTelemetry(
        df_gps=df_gps,
        df_imu=df_imu,
        df_att=df_att,
        quality_report=quality_report,
        event_report=event_report,
    )


def collect_metrics(analyzer: AnalysisService, df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> dict[str, float]:
    max_acceleration = analyzer.get_max_acceleration(df_imu) if not df_imu.empty else {}

    return {
        "Flight Duration (s)": analyzer.get_flight_duration(df_gps),
        "Distance Traveled (m)": analyzer.get_distance_traveled(df_gps),
        "Elevation Gain (m)": analyzer.get_elevation_gain(df_gps),
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
