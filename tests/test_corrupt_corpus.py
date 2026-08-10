import os
import unittest
from pathlib import Path

from app.parsers.base import ParseStatus
from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from app.services.data_quality import QualityStatus
from app.services.pipeline import prepare_telemetry_frames

CORPUS_ROOT = os.getenv("UAV_LOG_CORPUS")


@unittest.skipUnless(CORPUS_ROOT, "Set UAV_LOG_CORPUS to run external corpus tests")
class TestCorruptCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(CORPUS_ROOT or "")
        cls.parser = BinaryDataParser(max_diagnostic_lines=100)

    def test_c01_truncated_tail_detected_with_trusted_provenance(self) -> None:
        result = self.parser.parse_with_diagnostics(
            str(self.root / "corrupt/C01_truncated_tail.BIN"),
            expected_size_bytes=2_932_736,
            expected_sha256=(
                "33734e4050e95f6e95ec55bbcae4342d15cd60b03e5c1c88657f4500717886a4"
            ),
        )

        self.assertEqual(result.status, ParseStatus.PARTIAL)
        self.assertTrue(any("File size mismatch" in item for item in result.warnings))
        self.assertTrue(any("SHA-256 mismatch" in item for item in result.warnings))

    def test_c02_zeroed_prefix_has_bounded_diagnostics_and_no_gps(self) -> None:
        result = self.parser.parse_with_diagnostics(
            str(self.root / "corrupt/C02_zeroed_prefix.BIN")
        )

        self.assertEqual(result.status, ParseStatus.PARTIAL)
        self.assertGreater(result.diagnostics.total_lines, 100)
        self.assertLessEqual(len(result.diagnostics.captured_lines), 100)
        self.assertNotIn("GPS", result.dataframes)

    def test_c03_deleted_middle_block_is_partial(self) -> None:
        result = self.parser.parse_with_diagnostics(
            str(self.root / "corrupt/C03_deleted_middle_block.BIN")
        )

        self.assertEqual(result.status, ParseStatus.PARTIAL)
        self.assertGreater(result.diagnostics.total_lines, 0)

    def test_c04_byte_flips_keep_imu_rate_plausible(self) -> None:
        result = self.parser.parse_with_diagnostics(
            str(self.root / "corrupt/C04_deterministic_byte_flips.BIN")
        )
        telemetry = prepare_telemetry_frames(
            AnalysisService(),
            result.dataframes,
        )
        imu_rate = AnalysisService.get_sample_rate(telemetry.df_imu)

        self.assertEqual(result.status, ParseStatus.PARTIAL)
        self.assertEqual(
            telemetry.quality_report.streams["IMU"].status,
            QualityStatus.WARNING,
        )
        self.assertEqual(
            telemetry.quality_report.streams["IMU"].timestamp_outliers,
            1,
        )
        self.assertGreater(imu_rate, 1.0)
        self.assertLess(imu_rate, 1_000.0)

    def test_c05_appended_garbage_is_bounded_and_partial(self) -> None:
        result = self.parser.parse_with_diagnostics(
            str(self.root / "corrupt/C05_appended_garbage.BIN")
        )

        self.assertEqual(result.status, ParseStatus.PARTIAL)
        self.assertGreater(result.diagnostics.total_lines, 1_000_000)
        self.assertEqual(len(result.diagnostics.captured_lines), 100)
        self.assertGreater(result.diagnostics.suppressed_lines, 1_000_000)


if __name__ == "__main__":
    unittest.main()
