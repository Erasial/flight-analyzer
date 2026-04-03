import unittest

import pandas as pd

from app.core.utils import wgs84_to_enu
from app.services.analyzer import AnalysisService


class TestAnalysisService(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = AnalysisService()

        self.df_gps = pd.DataFrame(
            {
                "TimeUS": [0, 1_000_000, 2_000_000],
                "Lat": [48.000000, 48.000100, 48.000200],
                "Lng": [2.000000, 2.000100, 2.000200],
                "Alt": [100.0, 102.0, 103.0],
                "Spd": [0.0, 5.0, 7.0],
                "VZ": [0.2, -0.3, 0.1],
                "Status": [2, 3, 4],
                "GWk": [0, 1, 1],
                "NSats": [5, 8, 10],
            }
        )

        self.df_imu = pd.DataFrame(
            {
                "TimeUS": [0, 500_000, 1_000_000],
                "I": [0, 0, 0],
                "AccX": [1.0, -3.0, 2.0],
                "AccY": [0.5, -1.5, 0.2],
                "AccZ": [9.8, 10.0, 9.6],
            }
        )

    def test_filter_gps_low_quality_samples(self) -> None:
        filtered = self.analyzer.filter_gps_low_quality_samples(
            self.df_gps,
            min_status=3,
            require_positive_gwk=True,
            min_sats=8,
        )

        self.assertEqual(len(filtered), 2)
        self.assertTrue((filtered["Status"] >= 3).all())
        self.assertTrue((filtered["GWk"] > 0).all())
        self.assertTrue((filtered["NSats"] >= 8).all())

    def test_metrics_are_computed(self) -> None:
        distance = self.analyzer.get_distance_traveled(self.df_gps)
        sample_rate = self.analyzer.get_sample_rate(self.df_gps)
        duration = self.analyzer.get_flight_duration(self.df_gps)

        self.assertGreater(distance, 0.0)
        self.assertAlmostEqual(sample_rate, 1.0, places=6)
        self.assertAlmostEqual(duration, 2.0, places=6)

    def test_max_acceleration(self) -> None:
        max_acc = self.analyzer.get_max_acceleration(self.df_imu)
        self.assertEqual(max_acc["AccX"], 3.0)
        self.assertEqual(max_acc["AccY"], 1.5)
        self.assertEqual(max_acc["AccZ"], 10.0)


class TestWgs84ToEnu(unittest.TestCase):
    def test_adds_enu_columns_with_new_naming(self) -> None:
        df = pd.DataFrame(
            {
                "Lat": [48.0, 48.0001],
                "Lng": [2.0, 2.0001],
                "Alt": [100.0, 101.0],
            }
        )

        converted = wgs84_to_enu(df)

        self.assertIn("East", converted.columns)
        self.assertIn("North", converted.columns)
        self.assertIn("Up", converted.columns)
        self.assertAlmostEqual(converted.iloc[0]["East"], 0.0, places=6)
        self.assertAlmostEqual(converted.iloc[0]["North"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
