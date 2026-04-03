import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.services.analyzer import AnalysisService
from app.services.pipeline import collect_metrics, list_local_bin_files, prepare_telemetry_frames


class TestPipeline(unittest.TestCase):
    def test_list_local_bin_files_filters_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "b.BIN").write_text("x", encoding="utf-8")
            (root / "a.BIN").write_text("x", encoding="utf-8")
            (root / "ignore.txt").write_text("x", encoding="utf-8")

            files = list_local_bin_files(tmp_dir)

            self.assertEqual([p.name for p in files], ["a.BIN", "b.BIN"])

    def test_prepare_telemetry_frames_enriches_gps_and_filters_imu(self) -> None:
        analyzer = AnalysisService()
        dataframes = {
            "GPS": pd.DataFrame(
                {
                    "TimeUS": [1_000_000, 2_000_000],
                    "Lat": [48.0, 48.0001],
                    "Lng": [2.0, 2.0001],
                    "Alt": [100.0, 101.0],
                    "Status": [3, 3],
                    "GWk": [1, 1],
                    "Spd": [1.0, 2.0],
                    "VZ": [0.1, -0.2],
                }
            ),
            "IMU": pd.DataFrame(
                {
                    "I": [0, 1, 0],
                    "TimeUS": [1, 2, 3],
                    "AccX": [1.0, 5.0, 3.0],
                }
            ),
        }

        telemetry = prepare_telemetry_frames(analyzer, dataframes, imu_index=0)

        self.assertIn("East", telemetry.df_gps.columns)
        self.assertIn("North", telemetry.df_gps.columns)
        self.assertIn("Up", telemetry.df_gps.columns)
        self.assertTrue((telemetry.df_imu["I"] == 0).all())

    def test_collect_metrics_returns_expected_keys(self) -> None:
        analyzer = AnalysisService()
        df_gps = pd.DataFrame(
            {
                "TimeUS": [0, 1_000_000],
                "Lat": [48.0, 48.0001],
                "Lng": [2.0, 2.0001],
                "Alt": [100.0, 101.0],
                "Spd": [1.0, 2.0],
                "VZ": [0.1, 0.2],
            }
        )
        df_imu = pd.DataFrame(
            {
                "TimeUS": [0, 1_000_000],
                "I": [0, 0],
                "AccX": [1.0, 2.0],
                "AccY": [0.5, 0.7],
                "AccZ": [9.6, 9.8],
            }
        )

        metrics = collect_metrics(analyzer, df_gps, df_imu)

        self.assertIn("Flight Duration (s)", metrics)
        self.assertIn("Distance Traveled (m)", metrics)
        self.assertIn("Max Acc X (m/s^2)", metrics)


if __name__ == "__main__":
    unittest.main()
