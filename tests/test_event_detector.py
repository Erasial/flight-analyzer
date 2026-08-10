import unittest

import pandas as pd

from app.services.event_detector import (
    EventKind,
    EventSeverity,
    detect_flight_events,
)


class TestFlightEventDetector(unittest.TestCase):
    def test_deduplicates_msg_and_ev_and_builds_complete_segment(self) -> None:
        report = detect_flight_events(
            {
                "MSG": pd.DataFrame(
                    {
                        "TimeUS": [1_000_000, 5_000_000],
                        "Message": ["Arming motors", "Disarming motors"],
                    }
                ),
                "EV": pd.DataFrame(
                    {
                        "TimeUS": [1_000_000, 2_000_000, 5_000_000],
                        "Id": [10, 15, 11],
                    }
                ),
            }
        )

        self.assertEqual(
            [event.kind for event in report.events],
            [EventKind.ARM, EventKind.TAKEOFF, EventKind.DISARM],
        )
        self.assertEqual(len(report.segments), 1)
        self.assertTrue(report.segments[0].complete)
        self.assertEqual(report.segments[0].duration_s, 4.0)

    def test_radio_failsafe_and_automatic_rtl_keep_evidence(self) -> None:
        report = detect_flight_events(
            {
                "MSG": pd.DataFrame({"TimeUS": [10_000_000], "Message": ["Radio Failsafe"]}),
                "MODE": pd.DataFrame(
                    {
                        "TimeUS": [10_000_000],
                        "Mode": [6],
                        "ModeNum": [6],
                        "Rsn": [3],
                    }
                ),
                "ERR": pd.DataFrame({"TimeUS": [10_000_000], "Subsys": [5], "ECode": [1]}),
            }
        )

        failsafes = [event for event in report.events if event.kind is EventKind.RC_FAILSAFE]
        self.assertEqual(len(failsafes), 1)
        self.assertEqual(failsafes[0].source, "MSG")
        self.assertEqual(failsafes[0].severity, EventSeverity.CRITICAL)
        mode = next(event for event in report.events if event.kind is EventKind.MODE_CHANGE)
        self.assertEqual(mode.evidence["mode_name"], "RTL")
        self.assertEqual(mode.evidence["reason_name"], "radio_failsafe")

    def test_detects_gps_fix_loss_only_after_a_healthy_fix(self) -> None:
        report = detect_flight_events(
            {
                "GPS": pd.DataFrame(
                    {
                        "TimeUS": [0, 100_000, 200_000, 300_000],
                        "Status": [1, 3, 6, 1],
                    }
                )
            }
        )

        losses = [event for event in report.events if event.kind is EventKind.GPS_FIX_LOST]
        self.assertEqual(len(losses), 1)
        self.assertEqual(losses[0].time_us, 300_000)
        self.assertEqual(losses[0].evidence["previous_status"], 6.0)

    def test_ignores_prearm_and_autotest_narration(self) -> None:
        report = detect_flight_events(
            {
                "MSG": pd.DataFrame(
                    {
                        "TimeUS": [1, 2, 3],
                        "Message": [
                            "PreArm: Battery failsafe",
                            "GCS:AT-0001: Waiting for text: gcs failsafe",
                            "SRC=250/250: Test GCS Failsafe",
                        ],
                    }
                )
            }
        )

        self.assertEqual(report.events, ())

    def test_reports_incomplete_armed_segment(self) -> None:
        report = detect_flight_events({"EV": pd.DataFrame({"TimeUS": [1_000_000], "Id": [10]})})

        self.assertEqual(len(report.segments), 1)
        self.assertFalse(report.segments[0].complete)
        self.assertTrue(report.warnings)


if __name__ == "__main__":
    unittest.main()
