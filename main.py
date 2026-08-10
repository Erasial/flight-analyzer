import argparse
import logging

from app.parsers.base import ParseStatus
from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from app.services.data_quality import QualityStatus, assess_metric_quality
from app.services.pipeline import collect_metrics, prepare_telemetry_frames
from visualization.flight_plotter import plot_flight_path_3d


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Analyze ArduPilot BIN telemetry and export metrics/trajectory plot."
    )
    cli.add_argument("file_path", help="Path to .BIN log file")
    cli.add_argument("--imu-index", type=int, default=0, help="IMU module index to analyze")
    cli.add_argument(
        "--output-html",
        default="flight_trajectory_enu.html",
        help="Output HTML path for 3D trajectory",
    )
    cli.add_argument(
        "--no-ground", action="store_true", help="Disable ground surface on the 3D plot"
    )
    cli.add_argument("--no-plot", action="store_true", help="Skip trajectory HTML generation")
    cli.add_argument(
        "--expected-size",
        type=int,
        help="Trusted original file size in bytes for truncation detection",
    )
    cli.add_argument(
        "--expected-sha256",
        help="Trusted original SHA-256 for provenance verification",
    )
    cli.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity",
    )
    cli.add_argument(
        "--event-limit",
        type=int,
        default=50,
        help="Maximum detected non-mode events to print (default: 50)",
    )
    cli.add_argument(
        "--incident-limit",
        type=int,
        default=50,
        help="Maximum incident narratives to print (default: 50)",
    )
    cli.add_argument(
        "--segment-limit",
        type=int,
        default=50,
        help="Maximum flight/phase segment summaries to print (default: 50)",
    )
    return cli.parse_args()


def run() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    parser = BinaryDataParser()
    analyzer = AnalysisService()

    try:
        parse_result = parser.parse_with_diagnostics(
            args.file_path,
            expected_size_bytes=args.expected_size,
            expected_sha256=args.expected_sha256,
        )
        logger.info(
            "Data integrity: %s; decoded messages: %d",
            parse_result.status.value.upper(),
            parse_result.decoded_message_count,
        )
        for warning in parse_result.warnings:
            logger.warning("%s", warning)
        for diagnostic in parse_result.diagnostics.captured_lines[:5]:
            logger.debug("Decoder: %s", diagnostic)

        if parse_result.status is ParseStatus.REJECTED:
            logger.error("Analysis rejected: %s", parse_result.error or "invalid BIN file")
            return 3

        telemetry = prepare_telemetry_frames(
            analyzer,
            parse_result.dataframes,
            imu_index=args.imu_index,
        )

        if telemetry.df_gps.empty:
            if parse_result.status is ParseStatus.PARTIAL:
                logger.error("Analysis rejected: the damaged log contains no usable GPS data")
                return 3
            raise ValueError("GPS data is missing or empty in this log file")

        logger.info("Telemetry quality: %s", telemetry.quality_report.status.value.upper())
        for stream_name, stream_report in telemetry.quality_report.streams.items():
            logger.info(
                "%s quality: %s; valid=%d/%d; rejected=%d",
                stream_name,
                stream_report.status.value.upper(),
                stream_report.valid_records,
                stream_report.total_records,
                stream_report.rejected_records,
            )
            for warning in stream_report.warnings:
                logger.warning("%s", warning)

        event_report = telemetry.event_report
        logger.info(
            "Vehicle profile: %s; firmware: %s",
            event_report.vehicle_profile.value.upper(),
            event_report.firmware or "unknown",
        )
        logger.info(
            "Detected flight events: %d; segments: %d; critical: %d; warnings: %d",
            len(event_report.events),
            len(event_report.segments),
            event_report.critical_count,
            event_report.warning_count,
        )
        visible_events = [
            event
            for event in event_report.events
            if event.kind.value != "mode_change" or event.severity.value != "info"
        ]
        if args.event_limit > 0:
            print("\nDetected events:")
            for event in visible_events[: args.event_limit]:
                print(
                    f"{event.time_us}: {event.kind.value} "
                    f"[{event.severity.value.upper()}] {event.title}"
                )
            if len(visible_events) > args.event_limit:
                print(
                    f"... {len(visible_events) - args.event_limit} more event(s); "
                    "increase --event-limit to display them."
                )

        incident_report = telemetry.incident_report
        logger.info(
            "Incident report: %d incident(s); unresolved: %d",
            len(incident_report.incidents),
            incident_report.unresolved_count,
        )
        if incident_report.incidents and args.incident_limit > 0:
            print("\nIncident report:")
            for incident in incident_report.incidents[: args.incident_limit]:
                print(
                    f"#{incident.index} {incident.incident_type.value} "
                    f"[{incident.confidence.value.upper()}/{incident.status.value.upper()}]: "
                    f"{incident.narrative}"
                )
            if len(incident_report.incidents) > args.incident_limit:
                print(
                    f"... {len(incident_report.incidents) - args.incident_limit} "
                    "more incident(s); increase --incident-limit to display them."
                )

        segment_report = telemetry.segment_report
        logger.info("Segment analysis: %d segment(s)", len(segment_report.segments))
        if segment_report.segments and args.segment_limit > 0:
            print("\nSegment analysis:")
            for segment in segment_report.segments[: args.segment_limit]:
                distance = segment.metrics.get("Distance Traveled (m)")
                max_altitude = segment.metrics.get("Max Altitude (m)")
                details = [f"duration={segment.duration_s:.2f}s"]
                if distance is not None:
                    details.append(f"distance={distance:.2f}m")
                if max_altitude is not None:
                    details.append(f"max_alt={max_altitude:.2f}m")
                if segment.incident_indices:
                    details.append("incidents=" + ",".join(map(str, segment.incident_indices)))
                print(
                    f"#{segment.index} {segment.segment_type}:{segment.label} "
                    f"[{'COMPLETE' if segment.complete else 'PARTIAL'}] " + " ".join(details)
                )

        metrics = collect_metrics(analyzer, telemetry.df_gps, telemetry.df_imu)
        metric_quality = assess_metric_quality(metrics, telemetry.quality_report)
        for key, value in metrics.items():
            confidence = metric_quality[key].confidence.value.upper()
            print(f"{key}: {value:.2f} [{confidence}]")

        if not args.no_plot:
            plot_flight_path_3d(
                telemetry.df_gps,
                output_html=args.output_html,
                auto_open=False,
                show_ground=not args.no_ground,
            )
            logger.info("Saved trajectory HTML to %s", args.output_html)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        logger.error("Analysis failed: %s", exc)
        return 1

    if (
        parse_result.status is ParseStatus.PARTIAL
        or telemetry.quality_report.status is not QualityStatus.GOOD
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
