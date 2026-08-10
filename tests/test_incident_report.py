import unittest

import pandas as pd

from app.services.event_detector import EventKind, detect_flight_events
from app.services.incident_report import (
    IncidentConfidence,
    IncidentStatus,
    build_incident_report,
)


class TestIncidentReport(unittest.TestCase):
    def test_builds_rc_failsafe_action_and_outcome_chain(self) -> None:
        event_report = detect_flight_events(
            {
                "MSG": pd.DataFrame(
                    {
                        "TimeUS": [0, 1_000_000, 2_000_000, 8_000_000, 9_000_000],
                        "Message": [
                            "ArduCopter V4.8.0-dev (409226a6)",
                            "Arming motors",
                            "Radio Failsafe",
                            "SIM Hit ground at 0.5 m/s",
                            "Disarming motors",
                        ],
                    }
                ),
                "MODE": pd.DataFrame(
                    {
                        "TimeUS": [2_000_000],
                        "ModeNum": [6],
                        "Rsn": [3],
                    }
                ),
            }
        )

        report = build_incident_report(event_report)
        incident = report.incidents[0]

        self.assertEqual(incident.incident_type, EventKind.RC_FAILSAFE)
        self.assertEqual(incident.action_event.evidence["mode_name"], "RTL")
        self.assertEqual(incident.time_to_ground_s, 6.0)
        self.assertEqual(incident.time_to_disarm_s, 7.0)
        self.assertEqual(incident.confidence, IncidentConfidence.HIGH)
        self.assertEqual(incident.status, IncidentStatus.COMPLETED_BY_DISARM)

    def test_links_gps_loss_as_ekf_failsafe_precursor(self) -> None:
        event_report = detect_flight_events(
            {
                "GPS": pd.DataFrame({"TimeUS": [1_000_000, 2_000_000], "Status": [6, 1]}),
                "MSG": pd.DataFrame(
                    {
                        "TimeUS": [0, 5_000_000],
                        "Message": [
                            "ArduCopter V4.8.0-dev (409226a6)",
                            "EKF Failsafe: changed to Land Mode",
                        ],
                    }
                ),
                "MODE": pd.DataFrame({"TimeUS": [5_000_000], "ModeNum": [9], "Rsn": [6]}),
            }
        )

        incident = build_incident_report(event_report).incidents[0]

        self.assertEqual(incident.incident_type, EventKind.EKF_FAILSAFE)
        self.assertEqual(incident.trigger_event.kind, EventKind.GPS_FIX_LOST)
        self.assertIn("preceded", incident.narrative)

    def test_does_not_attribute_later_landing_after_failsafe_cleared(self) -> None:
        event_report = detect_flight_events(
            {
                "MSG": pd.DataFrame(
                    {
                        "TimeUS": [
                            1_000_000,
                            2_000_000,
                            3_000_000,
                            8_000_000,
                            9_000_000,
                        ],
                        "Message": [
                            "Arming motors",
                            "GCS Failsafe",
                            "GCS Failsafe Cleared",
                            "SIM Hit ground at 0.5 m/s",
                            "Disarming motors",
                        ],
                    }
                )
            }
        )

        incident = build_incident_report(event_report).incidents[0]

        self.assertEqual(incident.status, IncidentStatus.RESOLVED)
        self.assertIsNone(incident.ground_contact_event)
        self.assertIsNone(incident.disarm_event)


if __name__ == "__main__":
    unittest.main()
