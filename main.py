import argparse
import logging

from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from app.services.pipeline import collect_metrics, parse_data_from_path, prepare_telemetry_frames
from visualization.flight_plotter import plot_flight_path_3d


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(description="Analyze ArduPilot BIN telemetry and export metrics/trajectory plot.")
    cli.add_argument("file_path", help="Path to .BIN log file")
    cli.add_argument("--imu-index", type=int, default=0, help="IMU module index to analyze")
    cli.add_argument(
        "--output-html",
        default="flight_trajectory_enu.html",
        help="Output HTML path for 3D trajectory",
    )
    cli.add_argument("--no-plot", action="store_true", help="Skip trajectory HTML generation")
    cli.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity",
    )
    return cli.parse_args()


def run() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    parser = BinaryDataParser()
    analyzer = AnalysisService()

    try:
        dataframes = parse_data_from_path(parser, args.file_path)
        telemetry = prepare_telemetry_frames(analyzer, dataframes, imu_index=args.imu_index)

        if telemetry.df_gps.empty:
            raise ValueError("GPS data is missing or empty in this log file")

        metrics = collect_metrics(analyzer, telemetry.df_gps, telemetry.df_imu)
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")

        if not args.no_plot:
            plot_flight_path_3d(telemetry.df_gps, output_html=args.output_html, auto_open=False)
            logger.info("Saved trajectory HTML to %s", args.output_html)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        logger.error("Analysis failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
