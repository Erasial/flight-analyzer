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

        fig = plot_flight_path_3d(df, output_html=None, auto_open=False, color_by="combined", show_ground=False)

        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.data[0].name, "Trajectory")
        self.assertIn("Flight Trajectory (ENU)", fig.layout.title.text)

    def test_plot_can_disable_ground_surface(self) -> None:
        df = pd.DataFrame(
            {
                "East": [0.0, 1.0, 2.0],
                "North": [0.0, 1.0, 2.0],
                "Up": [10.0, 11.0, 12.0],
                "Spd": [2.0, 3.0, 4.0],
                "VZ": [0.1, -0.2, 0.3],
            }
        )

        fig = plot_flight_path_3d(df, output_html=None, auto_open=False, show_ground=False)

        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.data[0].name, "Trajectory")

    def test_plot_keeps_square_xy_ranges_without_ground(self) -> None:
        df = pd.DataFrame(
            {
                "East": [0.0, 100.0, 200.0],
                "North": [0.0, 10.0, 20.0],
                "Up": [10.0, 15.0, 12.0],
                "Spd": [2.0, 3.0, 4.0],
                "VZ": [0.1, -0.2, 0.3],
            }
        )

        fig = plot_flight_path_3d(df, output_html=None, auto_open=False, show_ground=False)

        x_range = fig.layout.scene.xaxis.range
        y_range = fig.layout.scene.yaxis.range
        self.assertIsNotNone(x_range)
        self.assertIsNotNone(y_range)
        x_span = float(x_range[1] - x_range[0])
        y_span = float(y_range[1] - y_range[0])
        self.assertAlmostEqual(x_span, y_span, places=6)

    def test_plot_supports_kmh_speed_unit(self) -> None:
        df = pd.DataFrame(
            {
                "East": [0.0, 1.0, 2.0],
                "North": [0.0, 1.0, 2.0],
                "Up": [10.0, 11.0, 12.0],
                "Spd": [2.0, 3.0, 4.0],
                "VZ": [0.1, -0.2, 0.3],
            }
        )

        fig = plot_flight_path_3d(
            df,
            output_html=None,
            auto_open=False,
            color_by="ground",
            speed_unit="km/h",
            show_ground=False,
        )

        trajectory_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "Trajectory")
        colorbar_title = trajectory_trace.line.colorbar.title.text
        self.assertEqual(colorbar_title, "Ground Speed (км/год)")

    def test_plot_supports_time_coloring(self) -> None:
        df = pd.DataFrame(
            {
                "East": [0.0, 1.0, 2.0],
                "North": [0.0, 1.0, 2.0],
                "Up": [10.0, 11.0, 12.0],
                "Spd": [2.0, 3.0, 4.0],
                "VZ": [0.1, -0.2, 0.3],
                "TimeUS": [1_000_000, 2_000_000, 3_000_000],
            }
        )

        fig = plot_flight_path_3d(df, output_html=None, auto_open=False, color_by="time", show_ground=False)

        trajectory_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "Trajectory")
        colorbar_title = trajectory_trace.line.colorbar.title.text
        self.assertEqual(colorbar_title, "Flight Time (s)")


if __name__ == "__main__":
    unittest.main()
