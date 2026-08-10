from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.analyzer import AnalysisService
from app.services.event_detector import EventKind, FlightEventReport, VehicleProfile
from app.services.incident_report import IncidentReport


@dataclass(frozen=True)
class SegmentAnalysis:
    index: int
    segment_type: str
    label: str
    start_time_us: int
    end_time_us: int
    duration_s: float
    complete: bool
    gps_records: int
    imu_records: int
    metrics: dict[str, float]
    incident_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "segment_type": self.segment_type,
            "label": self.label,
            "start_time_us": self.start_time_us,
            "end_time_us": self.end_time_us,
            "duration_s": self.duration_s,
            "complete": self.complete,
            "gps_records": self.gps_records,
            "imu_records": self.imu_records,
            "metrics": self.metrics,
            "incident_indices": list(self.incident_indices),
        }


@dataclass(frozen=True)
class SegmentAnalysisReport:
    segments: tuple[SegmentAnalysis, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "segment_count": len(self.segments),
                "complete_segment_count": sum(segment.complete for segment in self.segments),
                "armed_flight_count": sum(
                    segment.segment_type == "armed_flight" for segment in self.segments
                ),
                "rocket_phase_count": sum(
                    segment.segment_type == "rocket_phase" for segment in self.segments
                ),
            },
            "segments": [segment.to_dict() for segment in self.segments],
        }


def _last_telemetry_time_us(*dataframes: pd.DataFrame) -> int | None:
    maxima: list[int] = []
    for dataframe in dataframes:
        if dataframe.empty or "TimeUS" not in dataframe.columns:
            continue
        values = pd.to_numeric(dataframe["TimeUS"], errors="coerce").dropna()
        if not values.empty:
            maxima.append(int(values.max()))
    return max(maxima) if maxima else None


def _slice_time(
    dataframe: pd.DataFrame,
    start_us: int,
    end_us: int,
    *,
    include_end: bool,
) -> pd.DataFrame:
    if dataframe.empty or "TimeUS" not in dataframe.columns:
        return pd.DataFrame(columns=dataframe.columns)
    timestamps = pd.to_numeric(dataframe["TimeUS"], errors="coerce")
    if include_end:
        mask = (timestamps >= start_us) & (timestamps <= end_us)
    else:
        mask = (timestamps >= start_us) & (timestamps < end_us)
    return dataframe.loc[mask].reset_index(drop=True)


def _available_metrics(
    analyzer: AnalysisService,
    df_gps: pd.DataFrame,
    df_imu: pd.DataFrame,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not df_gps.empty:
        metrics.update(
            {
                "Flight Duration (s)": analyzer.get_flight_duration(df_gps),
                "Distance Traveled (m)": analyzer.get_distance_traveled(df_gps),
                "Elevation Gain (m)": analyzer.get_elevation_gain(df_gps),
                "Max Horizontal Speed (m/s)": analyzer.get_max_horizontal_speed(df_gps),
                "Max Vertical Speed (m/s)": analyzer.get_max_vertical_speed(df_gps),
                "Max Altitude (m)": analyzer.get_max_altitude(df_gps),
                "GPS Sample Rate (Hz)": analyzer.get_sample_rate(df_gps),
            }
        )
    if not df_imu.empty:
        acceleration = analyzer.get_max_acceleration(df_imu)
        metrics.update(
            {
                "Max Acc X (m/s^2)": acceleration.get("AccX", 0.0),
                "Max Acc Y (m/s^2)": acceleration.get("AccY", 0.0),
                "Max Acc Z (m/s^2)": acceleration.get("AccZ", 0.0),
                "IMU Sample Rate (Hz)": analyzer.get_sample_rate(df_imu),
            }
        )
    return metrics


def build_segment_analysis(
    analyzer: AnalysisService,
    event_report: FlightEventReport,
    incident_report: IncidentReport,
    df_gps: pd.DataFrame,
    df_imu: pd.DataFrame,
) -> SegmentAnalysisReport:
    analyses: list[SegmentAnalysis] = []
    last_time_us = _last_telemetry_time_us(df_gps, df_imu)

    for segment in event_report.segments:
        end_time_us = segment.disarm_time_us or last_time_us
        if end_time_us is None or end_time_us < segment.arm_time_us:
            continue
        gps_slice = _slice_time(df_gps, segment.arm_time_us, end_time_us, include_end=True)
        imu_slice = _slice_time(df_imu, segment.arm_time_us, end_time_us, include_end=True)
        incident_indices = tuple(
            incident.index
            for incident in incident_report.incidents
            if incident.segment_index == segment.index
        )
        analyses.append(
            SegmentAnalysis(
                index=segment.index,
                segment_type="armed_flight",
                label=f"Flight {segment.index}",
                start_time_us=segment.arm_time_us,
                end_time_us=end_time_us,
                duration_s=(end_time_us - segment.arm_time_us) / 1e6,
                complete=segment.complete,
                gps_records=len(gps_slice),
                imu_records=len(imu_slice),
                metrics=_available_metrics(analyzer, gps_slice, imu_slice),
                incident_indices=incident_indices,
            )
        )

    if event_report.vehicle_profile is VehicleProfile.ROCKET:
        phase_events = [
            event for event in event_report.events if event.kind is EventKind.ROCKET_STAGE
        ]
        for phase_index, phase_event in enumerate(phase_events, start=1):
            next_event = phase_events[phase_index] if phase_index < len(phase_events) else None
            end_time_us = next_event.time_us if next_event is not None else last_time_us
            if end_time_us is None or end_time_us < phase_event.time_us:
                continue
            gps_slice = _slice_time(
                df_gps,
                phase_event.time_us,
                end_time_us,
                include_end=next_event is None,
            )
            imu_slice = _slice_time(
                df_imu,
                phase_event.time_us,
                end_time_us,
                include_end=next_event is None,
            )
            phase_name = str(phase_event.evidence.get("to_stage", "UNKNOWN"))
            analyses.append(
                SegmentAnalysis(
                    index=phase_index,
                    segment_type="rocket_phase",
                    label=phase_name,
                    start_time_us=phase_event.time_us,
                    end_time_us=end_time_us,
                    duration_s=(end_time_us - phase_event.time_us) / 1e6,
                    complete=next_event is not None,
                    gps_records=len(gps_slice),
                    imu_records=len(imu_slice),
                    metrics=_available_metrics(analyzer, gps_slice, imu_slice),
                )
            )

    return SegmentAnalysisReport(segments=tuple(analyses))
