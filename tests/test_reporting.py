import unittest

import pandas as pd

from app.services.reporting import export_full_report_pdf_bytes, export_metrics_csv_bytes


class TestReportingExports(unittest.TestCase):
    def test_export_metrics_csv_bytes_contains_headers(self) -> None:
        metrics = {"Flight Duration (s)": 12.5, "Max Altitude (m)": 101.2}
        csv_bytes = export_metrics_csv_bytes(metrics)
        csv_text = csv_bytes.decode("utf-8")

        self.assertIn("Metric,Value", csv_text)
        self.assertIn("Flight Duration (s)", csv_text)
        self.assertIn("Max Altitude (m)", csv_text)

    def test_export_full_report_pdf_bytes_has_pdf_signature(self) -> None:
        metrics = {"Flight Duration (s)": 12.5}
        df_gps = pd.DataFrame({"TimeUS": [0, 1_000_000], "Spd": [1.0, 2.0]})
        df_imu = pd.DataFrame({"TimeUS": [0, 1_000_000], "VelAccNorm": [1.0, 1.2]})

        try:
            pdf_bytes = export_full_report_pdf_bytes(metrics, df_gps, df_imu, source_label="test.BIN")
        except ImportError:
            self.skipTest("reportlab is not installed in current environment")
            return

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 500)


if __name__ == "__main__":
    unittest.main()
