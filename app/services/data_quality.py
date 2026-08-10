from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd


class QualityStatus(StrEnum):
    GOOD = "good"
    WARNING = "warning"
    POOR = "poor"
    UNUSABLE = "unusable"


class MetricConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MetricQualityAssessment:
    metric: str
    source_stream: str
    confidence: MetricConfidence
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "source_stream": self.source_stream,
            "confidence": self.confidence.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StreamQualityReport:
    stream: str
    status: QualityStatus
    total_records: int
    valid_records: int
    rejected_records: int
    invalid_numeric_records: int = 0
    timestamp_outliers: int = 0
    duplicate_timestamps: int = 0
    non_monotonic_timestamps: int = 0
    gap_count: int = 0
    value_outliers: int = 0
    clipping_records: int = 0
    median_interval_us: float | None = None
    warnings: tuple[str, ...] = ()

    @property
    def valid_ratio(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.valid_records / self.total_records

    def to_dict(self) -> dict[str, object]:
        return {
            "stream": self.stream,
            "status": self.status.value,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "rejected_records": self.rejected_records,
            "valid_ratio": self.valid_ratio,
            "invalid_numeric_records": self.invalid_numeric_records,
            "timestamp_outliers": self.timestamp_outliers,
            "duplicate_timestamps": self.duplicate_timestamps,
            "non_monotonic_timestamps": self.non_monotonic_timestamps,
            "gap_count": self.gap_count,
            "value_outliers": self.value_outliers,
            "clipping_records": self.clipping_records,
            "median_interval_us": self.median_interval_us,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DataQualityReport:
    streams: dict[str, StreamQualityReport] = field(default_factory=dict)

    @property
    def status(self) -> QualityStatus:
        if not self.streams:
            return QualityStatus.UNUSABLE
        priority = {
            QualityStatus.GOOD: 0,
            QualityStatus.WARNING: 1,
            QualityStatus.POOR: 2,
            QualityStatus.UNUSABLE: 3,
        }
        return max((item.status for item in self.streams.values()), key=priority.get)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "streams": {name: report.to_dict() for name, report in self.streams.items()},
        }


@dataclass(frozen=True)
class ValidatedStream:
    dataframe: pd.DataFrame
    report: StreamQualityReport


GPS_METRICS = frozenset(
    {
        "Flight Duration (s)",
        "Distance Traveled (m)",
        "Elevation Gain (m)",
        "Max Horizontal Speed (m/s)",
        "Max Vertical Speed (m/s)",
        "Max Altitude (m)",
        "GPS Sample Rate (Hz)",
    }
)
IMU_METRICS = frozenset(
    {
        "Max Acc X (m/s^2)",
        "Max Acc Y (m/s^2)",
        "Max Acc Z (m/s^2)",
        "IMU Sample Rate (Hz)",
    }
)


def assess_metric_quality(
    metrics: dict[str, float],
    quality_report: DataQualityReport,
) -> dict[str, MetricQualityAssessment]:
    confidence_by_status = {
        QualityStatus.GOOD: MetricConfidence.HIGH,
        QualityStatus.WARNING: MetricConfidence.MEDIUM,
        QualityStatus.POOR: MetricConfidence.LOW,
        QualityStatus.UNUSABLE: MetricConfidence.UNAVAILABLE,
    }
    assessments: dict[str, MetricQualityAssessment] = {}
    for metric in metrics:
        if metric in GPS_METRICS:
            stream_name = "GPS"
        elif metric in IMU_METRICS:
            stream_name = "IMU"
        else:
            stream_name = "UNKNOWN"

        stream_report = quality_report.streams.get(stream_name)
        if stream_report is None:
            confidence = MetricConfidence.UNAVAILABLE
            reasons = (f"No quality report is available for {stream_name}.",)
        else:
            confidence = confidence_by_status[stream_report.status]
            reasons = stream_report.warnings

        assessments[metric] = MetricQualityAssessment(
            metric=metric,
            source_stream=stream_name,
            confidence=confidence,
            reasons=reasons,
        )
    return assessments


def _empty_result(stream: str, total_records: int = 0) -> ValidatedStream:
    return ValidatedStream(
        dataframe=pd.DataFrame(),
        report=StreamQualityReport(
            stream=stream,
            status=QualityStatus.UNUSABLE,
            total_records=total_records,
            valid_records=0,
            rejected_records=total_records,
            warnings=(f"{stream} data is missing or empty.",),
        ),
    )


def _isolated_timestamp_outliers(timestamps: pd.Series) -> pd.Series:
    values = timestamps.to_numpy(dtype=float)
    outliers = np.zeros(len(values), dtype=bool)
    if len(values) < 3:
        return pd.Series(outliers, index=timestamps.index)

    positive_diffs = np.diff(values)
    positive_diffs = positive_diffs[
        np.isfinite(positive_diffs) & (positive_diffs > 0) & (positive_diffs <= 60e6)
    ]
    if positive_diffs.size == 0:
        return pd.Series(outliers, index=timestamps.index)

    typical_interval = float(np.median(positive_diffs))
    local_gap_limit = max(10e6, typical_interval * 100)

    for index in range(1, len(values) - 1):
        previous_value = values[index - 1]
        current_value = values[index]
        next_value = values[index + 1]
        neighbor_gap = next_value - previous_value
        if not (0 < neighbor_gap <= local_gap_limit):
            continue
        if current_value < previous_value or current_value > next_value:
            outliers[index] = True

    return pd.Series(outliers, index=timestamps.index)


def _quality_status(total: int, valid: int, structural_warnings: int) -> QualityStatus:
    if total == 0 or valid == 0:
        return QualityStatus.UNUSABLE
    rejected_ratio = (total - valid) / total
    if rejected_ratio > 0.05:
        return QualityStatus.POOR
    if rejected_ratio > 0.001 or structural_warnings:
        return QualityStatus.WARNING
    return QualityStatus.GOOD


def _timestamp_diagnostics(
    timestamps: pd.Series,
) -> tuple[pd.Series, int, int, int, int, int, float | None]:
    numeric = pd.to_numeric(timestamps, errors="coerce")
    finite = pd.Series(np.isfinite(numeric), index=timestamps.index)
    valid_numeric = finite & (numeric >= 0)
    isolated_outliers = _isolated_timestamp_outliers(numeric.where(valid_numeric))
    keep_mask = valid_numeric & ~isolated_outliers

    kept = numeric.loc[keep_mask]
    duplicate_count = int(kept.duplicated(keep="first").sum())
    keep_mask.loc[kept.index[kept.duplicated(keep="first")]] = False

    cleaned_timestamps = numeric.loc[keep_mask]
    diffs = cleaned_timestamps.diff().dropna()
    non_monotonic_count = int((diffs <= 0).sum())
    positive_diffs = diffs[diffs > 0]
    median_interval = float(positive_diffs.median()) if not positive_diffs.empty else None
    gap_count = 0
    if median_interval is not None:
        gap_threshold = max(1e6, median_interval * 10)
        gap_count = int((positive_diffs > gap_threshold).sum())

    return (
        keep_mask,
        int((~valid_numeric).sum()),
        int(isolated_outliers.sum()),
        duplicate_count,
        non_monotonic_count,
        gap_count,
        median_interval,
    )


def validate_gps(df: pd.DataFrame) -> ValidatedStream:
    if df is None or df.empty:
        return _empty_result("GPS", 0 if df is None else len(df))
    if "TimeUS" not in df.columns:
        return _empty_result("GPS", len(df))

    working = df.copy()
    total = len(working)
    (
        keep_mask,
        invalid_numeric,
        timestamp_outliers,
        duplicates,
        non_monotonic,
        gaps,
        median_interval,
    ) = _timestamp_diagnostics(working["TimeUS"])

    value_mask = pd.Series(True, index=working.index)
    for column, lower, upper in (
        ("Lat", -90.0, 90.0),
        ("Lng", -180.0, 180.0),
        ("Alt", -1_000.0, 30_000.0),
        ("Spd", 0.0, 200.0),
        ("VZ", -100.0, 100.0),
    ):
        if column not in working.columns:
            continue
        values = pd.to_numeric(working[column], errors="coerce")
        value_mask &= values.notna() & np.isfinite(values) & values.between(lower, upper)
        working[column] = values

    if "Lat" in working.columns and "Lng" in working.columns:
        value_mask &= ~((working["Lat"] == 0) & (working["Lng"] == 0))
    if "Status" in working.columns:
        status = pd.to_numeric(working["Status"], errors="coerce")
        value_mask &= status >= 3
    if "GWk" in working.columns:
        gps_week = pd.to_numeric(working["GWk"], errors="coerce")
        value_mask &= gps_week > 0

    value_outliers = int((keep_mask & ~value_mask).sum())
    final_mask = keep_mask & value_mask
    cleaned = working.loc[final_mask].sort_values("TimeUS").reset_index(drop=True)
    rejected = total - len(cleaned)
    structural_warnings = duplicates + non_monotonic + gaps + timestamp_outliers
    warnings: list[str] = []
    if timestamp_outliers:
        warnings.append(f"GPS: rejected {timestamp_outliers} isolated timestamp outlier(s).")
    if value_outliers:
        warnings.append(f"GPS: rejected {value_outliers} invalid or implausible record(s).")
    if duplicates:
        warnings.append(f"GPS: removed {duplicates} duplicate timestamp(s).")
    if non_monotonic:
        warnings.append(f"GPS: detected {non_monotonic} non-monotonic timestamp(s).")
    if gaps:
        warnings.append(f"GPS: detected {gaps} telemetry gap(s).")

    return ValidatedStream(
        dataframe=cleaned,
        report=StreamQualityReport(
            stream="GPS",
            status=_quality_status(total, len(cleaned), structural_warnings),
            total_records=total,
            valid_records=len(cleaned),
            rejected_records=rejected,
            invalid_numeric_records=invalid_numeric,
            timestamp_outliers=timestamp_outliers,
            duplicate_timestamps=duplicates,
            non_monotonic_timestamps=non_monotonic,
            gap_count=gaps,
            value_outliers=value_outliers,
            median_interval_us=median_interval,
            warnings=tuple(warnings),
        ),
    )


def validate_imu(df: pd.DataFrame) -> ValidatedStream:
    if df is None or df.empty:
        return _empty_result("IMU", 0 if df is None else len(df))
    if "TimeUS" not in df.columns:
        return _empty_result("IMU", len(df))

    working = df.copy()
    total = len(working)
    (
        keep_mask,
        invalid_numeric,
        timestamp_outliers,
        duplicates,
        non_monotonic,
        gaps,
        median_interval,
    ) = _timestamp_diagnostics(working["TimeUS"])

    value_mask = pd.Series(True, index=working.index)
    for column in ("AccX", "AccY", "AccZ"):
        if column not in working.columns:
            continue
        values = pd.to_numeric(working[column], errors="coerce")
        value_mask &= values.notna() & np.isfinite(values) & (values.abs() <= 200.0)
        working[column] = values
    for column in ("GyrX", "GyrY", "GyrZ", "GyroX", "GyroY", "GyroZ"):
        if column not in working.columns:
            continue
        values = pd.to_numeric(working[column], errors="coerce")
        value_mask &= values.notna() & np.isfinite(values) & (values.abs() <= 2_000.0)
        working[column] = values

    clipping_records = 0
    for column in ("Clip", "Clipping", "Clip0", "Clip1", "Clip2"):
        if column in working.columns:
            clipping = pd.to_numeric(working[column], errors="coerce").fillna(0)
            clipping_records += int((clipping > 0).sum())

    value_outliers = int((keep_mask & ~value_mask).sum())
    final_mask = keep_mask & value_mask
    cleaned = working.loc[final_mask].sort_values("TimeUS").reset_index(drop=True)
    rejected = total - len(cleaned)
    structural_warnings = duplicates + non_monotonic + gaps + timestamp_outliers + clipping_records
    warnings: list[str] = []
    if timestamp_outliers:
        warnings.append(f"IMU: rejected {timestamp_outliers} isolated timestamp outlier(s).")
    if value_outliers:
        warnings.append(f"IMU: rejected {value_outliers} invalid or implausible record(s).")
    if duplicates:
        warnings.append(f"IMU: removed {duplicates} duplicate timestamp(s).")
    if non_monotonic:
        warnings.append(f"IMU: detected {non_monotonic} non-monotonic timestamp(s).")
    if gaps:
        warnings.append(f"IMU: detected {gaps} telemetry gap(s).")
    if clipping_records:
        warnings.append(f"IMU: detected clipping in {clipping_records} record(s).")

    return ValidatedStream(
        dataframe=cleaned,
        report=StreamQualityReport(
            stream="IMU",
            status=_quality_status(total, len(cleaned), structural_warnings),
            total_records=total,
            valid_records=len(cleaned),
            rejected_records=rejected,
            invalid_numeric_records=invalid_numeric,
            timestamp_outliers=timestamp_outliers,
            duplicate_timestamps=duplicates,
            non_monotonic_timestamps=non_monotonic,
            gap_count=gaps,
            value_outliers=value_outliers,
            clipping_records=clipping_records,
            median_interval_us=median_interval,
            warnings=tuple(warnings),
        ),
    )
