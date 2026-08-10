import os
import unittest
from collections import Counter
from pathlib import Path

from app.parsers.binary import BinaryDataParser
from app.services.event_detector import EventKind, detect_flight_events

CORPUS_ROOT = os.getenv("UAV_LOG_CORPUS")


@unittest.skipUnless(CORPUS_ROOT, "Set UAV_LOG_CORPUS to run external corpus tests")
class TestEventCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(CORPUS_ROOT or "")
        cls.parser = BinaryDataParser(max_diagnostic_lines=100)

    def _report(self, relative_path: str):
        parse_result = self.parser.parse_with_diagnostics(str(self.root / relative_path))
        return detect_flight_events(parse_result.dataframes)

    def test_s06_rc_failsafe_matches_ground_truth(self) -> None:
        report = self._report("sitl/S06_rc_failsafe_rtl.BIN")
        rc_event = next(event for event in report.events if event.kind is EventKind.RC_FAILSAFE)
        rtl_event = next(
            event
            for event in report.events
            if event.kind is EventKind.MODE_CHANGE and event.evidence.get("reason") == 3
        )

        self.assertEqual(rc_event.time_us, 213_984_372)
        self.assertEqual(rtl_event.time_us, 213_984_372)
        self.assertEqual(rtl_event.evidence["mode_name"], "RTL")
        self.assertEqual(report.segments[0].arm_time_us, 179_519_830)
        self.assertEqual(report.segments[0].disarm_time_us, 243_089_392)

    def test_s07_gps_loss_and_ekf_land_match_ground_truth(self) -> None:
        report = self._report("sitl/S07_gps_loss_ekf_land.BIN")
        by_kind = {event.kind: event for event in report.events}

        self.assertEqual(by_kind[EventKind.GPS_FIX_LOST].time_us, 834_199_520)
        self.assertEqual(by_kind[EventKind.EKF_FAILSAFE].time_us, 841_939_756)
        land_event = next(
            event
            for event in report.events
            if event.kind is EventKind.MODE_CHANGE and event.evidence.get("reason") == 6
        )
        self.assertEqual(land_event.evidence["mode_name"], "LAND")

    def test_s09_critical_battery_land_matches_ground_truth(self) -> None:
        report = self._report("sitl/S09_battery_critical_land.BIN")
        battery_event = next(
            event for event in report.events if event.kind is EventKind.BATTERY_FAILSAFE
        )
        land_event = next(
            event
            for event in report.events
            if event.kind is EventKind.MODE_CHANGE and event.evidence.get("reason") == 4
        )

        self.assertEqual(battery_event.time_us, 166_739_111)
        self.assertEqual(land_event.time_us, 166_739_111)
        self.assertEqual(land_event.evidence["mode_name"], "LAND")

    def test_at06_detects_all_gcs_failsafe_cases_and_segments(self) -> None:
        report = self._report("autotest/AT06_GCSFailsafe.BIN")
        counts = Counter(event.kind for event in report.events)

        self.assertEqual(counts[EventKind.GCS_FAILSAFE], 13)
        self.assertEqual(counts[EventKind.GCS_FAILSAFE_CLEARED], 13)
        self.assertEqual(len(report.segments), 8)
        self.assertTrue(all(segment.complete for segment in report.segments))

    def test_at10_detects_four_fence_breaches_and_segments(self) -> None:
        report = self._report("autotest/AT10_MaxAltFence.BIN")
        breaches = [event for event in report.events if event.kind is EventKind.FENCE_BREACH]

        self.assertEqual(
            [event.time_us for event in breaches],
            [136_199_665, 343_359_268, 618_039_352, 949_319_287],
        )
        self.assertEqual(len(report.segments), 4)
        self.assertTrue(all(segment.complete for segment in report.segments))


if __name__ == "__main__":
    unittest.main()
