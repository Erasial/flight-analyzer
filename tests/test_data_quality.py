import unittest

import pandas as pd

from app.services.data_quality import (
    DataQualityReport,
    MetricConfidence,
    QualityStatus,
    StreamQualityReport,
    assess_metric_quality,
    validate_gps,
    validate_imu,
)


class TestDataQuality(unittest.TestCase):
    def test_gps_rejects_isolated_timestamp_outlier(self) -> None:
        gps = pd.DataFrame(
            {
                "TimeUS": [0, 100_000, 10_000_000_000, 300_000, 400_000],
                "Lat": [48.0] * 5,
                "Lng": [30.0] * 5,
                "Alt": [100.0] * 5,
                "Spd": [1.0] * 5,
                "Status": [3] * 5,
                "GWk": [2_400] * 5,
            }
        )

        result = validate_gps(gps)

        self.assertEqual(result.report.status, QualityStatus.POOR)
        self.assertEqual(result.report.timestamp_outliers, 1)
        self.assertEqual(result.report.valid_records, 4)
        self.assertNotIn(10_000_000_000, result.dataframe["TimeUS"].tolist())

    def test_gps_rejects_invalid_position_and_low_quality_fix(self) -> None:
        gps = pd.DataFrame(
            {
                "TimeUS": [0, 100_000, 200_000, 300_000],
                "Lat": [48.0, 91.0, 0.0, 48.1],
                "Lng": [30.0, 30.0, 0.0, 30.1],
                "Alt": [100.0] * 4,
                "Spd": [1.0] * 4,
                "Status": [3, 3, 3, 2],
                "GWk": [2_400] * 4,
            }
        )

        result = validate_gps(gps)

        self.assertEqual(result.report.valid_records, 1)
        self.assertEqual(result.report.value_outliers, 3)
        self.assertEqual(result.report.status, QualityStatus.POOR)

    def test_imu_reports_duplicates_clipping_and_implausible_acceleration(self) -> None:
        imu = pd.DataFrame(
            {
                "TimeUS": [0, 100_000, 100_000, 300_000],
                "AccX": [0.0, 1.0, 1.0, 500.0],
                "AccY": [0.0, 0.0, 0.0, 0.0],
                "AccZ": [9.8, 9.8, 9.8, 9.8],
                "GyrX": [0.0] * 4,
                "GyrY": [0.0] * 4,
                "GyrZ": [0.0] * 4,
                "Clip": [0, 1, 0, 0],
            }
        )

        result = validate_imu(imu)

        self.assertEqual(result.report.duplicate_timestamps, 1)
        self.assertEqual(result.report.clipping_records, 1)
        self.assertEqual(result.report.value_outliers, 1)
        self.assertEqual(result.report.valid_records, 2)
        self.assertEqual(result.report.status, QualityStatus.POOR)

    def test_overall_report_uses_worst_stream_status(self) -> None:
        gps = validate_gps(pd.DataFrame()).report
        imu = validate_imu(
            pd.DataFrame(
                {
                    "TimeUS": [0, 100_000],
                    "AccX": [0.0, 0.0],
                    "AccY": [0.0, 0.0],
                    "AccZ": [9.8, 9.8],
                }
            )
        ).report

        report = DataQualityReport(streams={"GPS": gps, "IMU": imu})

        self.assertEqual(report.status, QualityStatus.UNUSABLE)
        self.assertEqual(report.to_dict()["streams"]["GPS"]["valid_records"], 0)

    def test_metric_confidence_follows_its_source_stream(self) -> None:
        report = DataQualityReport(
            streams={
                "GPS": StreamQualityReport(
                    stream="GPS",
                    status=QualityStatus.GOOD,
                    total_records=100,
                    valid_records=100,
                    rejected_records=0,
                ),
                "IMU": StreamQualityReport(
                    stream="IMU",
                    status=QualityStatus.WARNING,
                    total_records=100,
                    valid_records=99,
                    rejected_records=1,
                    warnings=("IMU gap detected.",),
                ),
            }
        )
        metrics = {
            "Distance Traveled (m)": 42.0,
            "Max Acc X (m/s^2)": 5.0,
            "Custom Metric": 1.0,
        }

        result = assess_metric_quality(metrics, report)

        self.assertEqual(
            result["Distance Traveled (m)"].confidence,
            MetricConfidence.HIGH,
        )
        self.assertEqual(
            result["Max Acc X (m/s^2)"].confidence,
            MetricConfidence.MEDIUM,
        )
        self.assertEqual(result["Max Acc X (m/s^2)"].reasons, ("IMU gap detected.",))
        self.assertEqual(
            result["Custom Metric"].confidence,
            MetricConfidence.UNAVAILABLE,
        )


if __name__ == "__main__":
    unittest.main()
