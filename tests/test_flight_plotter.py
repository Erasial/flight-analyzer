import unittest

import pandas as pd

from visualization.flight_plotter import plot_flight_path_3d


class TestFlightPlotter(unittest.TestCase):
    def test_plot_raises_on_missing_columns(self) -> None:
        df = pd.DataFrame({"North": [0.0], "Up": [0.0], "Spd": [1.0], "VZ": [0.1]})

        with self.assertRaises(ValueError):
            plot_flight_path_3d(df, output_html=None, auto_open=False)

    def test_plot_returns_figure(self) -> None:
        df = pd.DataFrame(
            {
                "East": [0.0, 1.0, 2.0],
                "North": [0.0, 1.0, 2.0],
                "Up": [10.0, 11.0, 12.0],
                "Spd": [2.0, 3.0, 4.0],
                "VZ": [0.1, -0.2, 0.3],
            }
        )

        fig = plot_flight_path_3d(df, output_html=None, auto_open=False, color_by="combined")

        self.assertEqual(len(fig.data), 3)
        self.assertIn("Flight Trajectory (ENU)", fig.layout.title.text)


if __name__ == "__main__":
    unittest.main()
