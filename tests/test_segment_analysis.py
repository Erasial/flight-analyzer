import unittest

import pandas as pd

from app.services.analyzer import AnalysisService
from app.services.event_detector import detect_flight_events
from app.services.incident_report import build_incident_report
from app.services.segment_analysis import build_segment_analysis


class TestSegmentAnalysis(unittest.TestCase):
    def test_builds_metrics_for_each_armed_flight(self) -> None:
        event_report = detect_flight_events(
            {
                "MSG": pd.DataFrame(
                    {
                        "TimeUS": [1_000_000, 4_000_000, 6_000_000, 9_000_000],
                        "Message": [
                            "Arming motors",
                            "Disarming motors",
                            "Arming motors",
                            "Disarming motors",
                        ],
                    }
                )
            }
        )
        gps = pd.DataFrame(
            {
                "TimeUS": [1_000_000, 2_000_000, 3_000_000, 6_000_000, 7_000_000],
                "Lat": [48.0, 48.0001, 48.0002, 49.0, 49.0001],
                "Lng": [30.0, 30.0, 30.0, 31.0, 31.0],
                "Alt": [100.0, 105.0, 110.0, 200.0, 205.0],
                "Spd": [0.0, 2.0, 3.0, 0.0, 1.0],
                "VZ": [0.0, 1.0, 1.0, 0.0, 1.0],
            }
        )
        incident_report = build_incident_report(event_report)

        report = build_segment_analysis(
            AnalysisService(), event_report, incident_report, gps, pd.DataFrame()
        )

        self.assertEqual(len(report.segments), 2)
        self.assertEqual(report.segments[0].duration_s, 3.0)
        self.assertEqual(report.segments[0].gps_records, 3)
        self.assertEqual(report.segments[1].gps_records, 2)
        self.assertGreater(report.segments[0].metrics["Distance Traveled (m)"], 0)

    def test_builds_rocket_phases_without_armed_segments(self) -> None:
        event_report = detect_flight_events(
            {
                "MSG": pd.DataFrame(
                    {
                        "TimeUS": [0, 1_000_000, 3_000_000, 5_000_000],
                        "Message": [
                            "TheRocket V4.6.1 (test)",
                            "FSTG: PAD_IDLE -> BOOST",
                            "FSTG: BOOST -> COAST",
                            "FSTG: COAST -> APOGEE",
                        ],
                    }
                )
            }
        )
        imu = pd.DataFrame(
            {
                "TimeUS": [1_000_000, 2_000_000, 3_000_000, 4_000_000, 6_000_000],
                "AccX": [1.0, 2.0, 0.5, 0.2, 0.1],
                "AccY": [0.0] * 5,
                "AccZ": [9.8] * 5,
            }
        )

        report = build_segment_analysis(
            AnalysisService(),
            event_report,
            build_incident_report(event_report),
            pd.DataFrame(),
            imu,
        )

        self.assertEqual(
            [segment.label for segment in report.segments], ["BOOST", "COAST", "APOGEE"]
        )
        self.assertTrue(report.segments[0].complete)
        self.assertFalse(report.segments[-1].complete)
        self.assertEqual(report.segments[0].metrics["Max Acc X (m/s^2)"], 2.0)


if __name__ == "__main__":
    unittest.main()
