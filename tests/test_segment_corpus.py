import os
import unittest
from pathlib import Path

from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from app.services.pipeline import prepare_telemetry_frames

CORPUS_ROOT = os.getenv("UAV_LOG_CORPUS")


@unittest.skipUnless(CORPUS_ROOT, "Set UAV_LOG_CORPUS to run external corpus tests")
class TestSegmentCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(CORPUS_ROOT or "")
        cls.parser = BinaryDataParser(max_diagnostic_lines=100)

    def _telemetry(self, relative_path: str):
        result = self.parser.parse_with_diagnostics(str(self.root / relative_path))
        return prepare_telemetry_frames(AnalysisService(), result.dataframes)

    def test_at06_has_eight_separate_flights(self) -> None:
        telemetry = self._telemetry("autotest/AT06_GCSFailsafe.BIN")
        segments = telemetry.segment_report.segments

        self.assertEqual(len(segments), 8)
        self.assertTrue(all(segment.complete for segment in segments))
        self.assertEqual([segment.index for segment in segments], list(range(1, 9)))
        self.assertTrue(all(segment.gps_records > 0 for segment in segments))
        self.assertEqual(sum(len(segment.incident_indices) for segment in segments), 13)

    def test_at10_has_four_fence_flights(self) -> None:
        telemetry = self._telemetry("autotest/AT10_MaxAltFence.BIN")
        segments = telemetry.segment_report.segments

        self.assertEqual(len(segments), 4)
        self.assertTrue(all(segment.complete for segment in segments))
        self.assertTrue(all(len(segment.incident_indices) == 1 for segment in segments))
        self.assertTrue(all(segment.metrics["Distance Traveled (m)"] > 0 for segment in segments))


if __name__ == "__main__":
    unittest.main()
