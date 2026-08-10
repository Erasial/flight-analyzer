import os
import unittest
from collections import Counter
from pathlib import Path

from app.parsers.binary import BinaryDataParser
from app.services.event_detector import EventKind, detect_flight_events
from app.services.incident_report import (
    IncidentConfidence,
    IncidentStatus,
    build_incident_report,
)

CORPUS_ROOT = os.getenv("UAV_LOG_CORPUS")


@unittest.skipUnless(CORPUS_ROOT, "Set UAV_LOG_CORPUS to run external corpus tests")
class TestIncidentCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(CORPUS_ROOT or "")
        cls.parser = BinaryDataParser(max_diagnostic_lines=100)

    def _report(self, relative_path: str):
        parse_result = self.parser.parse_with_diagnostics(str(self.root / relative_path))
        events = detect_flight_events(parse_result.dataframes)
        return build_incident_report(events)

    def test_s06_rc_failsafe_chain_matches_ground_truth(self) -> None:
        incident = self._report("sitl/S06_rc_failsafe_rtl.BIN").incidents[0]

        self.assertEqual(incident.incident_type, EventKind.RC_FAILSAFE)
        self.assertEqual(incident.action_event.evidence["mode_name"], "RTL")
        self.assertEqual(incident.response_latency_s, 0.0)
        self.assertAlmostEqual(incident.time_to_ground_s, 26.991699, places=6)
        self.assertAlmostEqual(incident.time_to_disarm_s, 29.105020, places=6)
        self.assertEqual(incident.confidence, IncidentConfidence.HIGH)

    def test_s07_links_gps_loss_to_ekf_land(self) -> None:
        incident = self._report("sitl/S07_gps_loss_ekf_land.BIN").incidents[0]

        self.assertEqual(incident.trigger_event.kind, EventKind.GPS_FIX_LOST)
        self.assertEqual(incident.trigger_event.time_us, 834_199_520)
        self.assertEqual(incident.failsafe_event.time_us, 841_939_756)
        self.assertEqual(incident.action_event.evidence["mode_name"], "LAND")
        self.assertAlmostEqual(incident.time_to_disarm_s, 26.419428, places=6)

    def test_s09_links_critical_battery_to_land(self) -> None:
        incident = self._report("sitl/S09_battery_critical_land.BIN").incidents[0]

        self.assertEqual(incident.trigger_event.kind, EventKind.BATTERY_CRITICAL)
        self.assertEqual(incident.incident_type, EventKind.BATTERY_FAILSAFE)
        self.assertEqual(incident.action_event.evidence["mode_name"], "LAND")
        self.assertAlmostEqual(incident.time_to_ground_s, 29.116682, places=6)

    def test_at06_builds_thirteen_resolved_gcs_incidents(self) -> None:
        report = self._report("autotest/AT06_GCSFailsafe.BIN")
        modes = Counter(
            incident.action_event.evidence["mode_name"]
            for incident in report.incidents
            if incident.action_event is not None
        )

        self.assertEqual(len(report.incidents), 13)
        self.assertTrue(
            all(incident.status is IncidentStatus.RESOLVED for incident in report.incidents)
        )
        self.assertEqual(modes["SMART_RTL"], 4)
        self.assertEqual(modes["RTL"], 4)
        self.assertEqual(modes["LAND"], 1)

    def test_at10_builds_four_fence_to_rtl_landing_chains(self) -> None:
        report = self._report("autotest/AT10_MaxAltFence.BIN")

        self.assertEqual(len(report.incidents), 4)
        self.assertTrue(
            all(
                incident.action_event.evidence["mode_name"] == "RTL"
                for incident in report.incidents
            )
        )
        self.assertTrue(
            all(incident.ground_contact_event is not None for incident in report.incidents)
        )


if __name__ == "__main__":
    unittest.main()
