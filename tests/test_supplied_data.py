import unittest
from pathlib import Path

from app.parsers.base import ParseStatus
from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from app.services.event_detector import EventKind, VehicleProfile
from app.services.pipeline import prepare_telemetry_frames


class TestSuppliedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = Path(__file__).resolve().parents[1] / "data"
        cls.parser = BinaryDataParser(max_diagnostic_lines=100)

    def _analyze(self, filename: str):
        result = self.parser.parse_with_diagnostics(str(self.data_root / filename))
        telemetry = prepare_telemetry_frames(AnalysisService(), result.dataframes)
        return result, telemetry

    def test_00000001_is_complete_high_dynamic_rocket_log(self) -> None:
        result, telemetry = self._analyze("00000001.BIN")

        self.assertEqual(result.status, ParseStatus.COMPLETE)
        self.assertEqual(telemetry.event_report.vehicle_profile, VehicleProfile.ROCKET)
        self.assertEqual(
            telemetry.event_report.firmware,
            "TheRocket V4.6.1 (93e7bd15 61d0d0fe)",
        )
        stages = [
            event.evidence["to_stage"]
            for event in telemetry.event_report.events
            if event.kind is EventKind.ROCKET_STAGE
        ]
        self.assertEqual(stages, ["PAD_IDLE", "BOOST", "COAST", "APOGEE", "DESCENT"])
        self.assertEqual(
            sum(
                event.kind is EventKind.PARACHUTE_RELEASE for event in telemetry.event_report.events
            ),
            1,
        )
        self.assertEqual(telemetry.quality_report.streams["GPS"].valid_records, 262)
        self.assertEqual(telemetry.incident_report.incidents, ())

    def test_00000019_filters_only_uninitialized_gps_sample(self) -> None:
        result, telemetry = self._analyze("00000019.BIN")

        self.assertEqual(result.status, ParseStatus.COMPLETE)
        self.assertEqual(telemetry.event_report.vehicle_profile, VehicleProfile.ROCKET)
        self.assertEqual(telemetry.quality_report.streams["GPS"].valid_records, 117)
        self.assertEqual(telemetry.quality_report.streams["GPS"].rejected_records, 1)
        mode_names = [
            event.evidence["mode_name"]
            for event in telemetry.event_report.events
            if event.kind is EventKind.MODE_CHANGE
        ]
        self.assertTrue(all(name.startswith("CUSTOM_MODE_") for name in mode_names))
        self.assertEqual(telemetry.incident_report.incidents, ())


if __name__ == "__main__":
    unittest.main()
