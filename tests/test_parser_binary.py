import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.parsers.base import ParseStatus
from app.parsers.binary import BinaryDataParser, BinaryParseError


class FakeMsg:
    def __init__(self, msg_type: str, payload: dict):
        self._msg_type = msg_type
        self._payload = payload

    def get_type(self) -> str:
        return self._msg_type

    def to_dict(self) -> dict:
        return dict(self._payload)


class FakeLog:
    def __init__(self, messages, terminal_error=None):
        self._messages = iter(messages)
        self._terminal_error = terminal_error

    def recv_match(self, **_kwargs):
        message = next(self._messages, None)
        if message is None and self._terminal_error is not None:
            error = self._terminal_error
            self._terminal_error = None
            raise error
        return message

    def close(self) -> None:
        return None


class TestBinaryDataParser(unittest.TestCase):
    def test_parse_groups_messages_and_skips_fmt(self) -> None:
        messages = [
            FakeMsg("FMT", {"mavpackettype": "FMT", "x": 1}),
            FakeMsg("GPS", {"mavpackettype": "GPS", "Lat": 1.0, "Lng": 2.0}),
            FakeMsg("GPS", {"mavpackettype": "GPS", "Lat": 1.1, "Lng": 2.1}),
            FakeMsg("IMU", {"mavpackettype": "IMU", "AccX": 0.5}),
        ]

        with patch("app.parsers.binary.mavutil.mavlink_connection", return_value=FakeLog(messages)):
            parser = BinaryDataParser()
            data = parser.parse(__file__)

        self.assertIn("GPS", data)
        self.assertIn("IMU", data)
        self.assertNotIn("FMT", data)
        self.assertIsInstance(data["GPS"], pd.DataFrame)
        self.assertEqual(len(data["GPS"]), 2)
        self.assertNotIn("mavpackettype", data["GPS"].columns)

    def test_default_parser_retains_only_analysis_message_types(self) -> None:
        messages = [
            FakeMsg("GPS", {"Lat": 1.0}),
            FakeMsg("BAT", {"Volt": 12.0}),
        ]

        with patch(
            "app.parsers.binary.mavutil.mavlink_connection",
            return_value=FakeLog(messages),
        ):
            data = BinaryDataParser().parse(__file__)

        self.assertIn("GPS", data)
        self.assertNotIn("BAT", data)

    def test_parser_can_opt_in_to_all_message_types(self) -> None:
        messages = [FakeMsg("BAT", {"Volt": 12.0})]

        with patch(
            "app.parsers.binary.mavutil.mavlink_connection",
            return_value=FakeLog(messages),
        ):
            data = BinaryDataParser(message_types=None).parse(__file__)

        self.assertIn("BAT", data)

    def test_parse_result_marks_unclosed_armed_log_partial(self) -> None:
        messages = [
            FakeMsg("MSG", {"Message": "Arming motors"}),
            FakeMsg("GPS", {"Lat": 1.0, "Lng": 2.0}),
        ]

        with patch("app.parsers.binary.mavutil.mavlink_connection", return_value=FakeLog(messages)):
            result = BinaryDataParser().parse_with_diagnostics(__file__)

        self.assertEqual(result.status, ParseStatus.PARTIAL)
        self.assertEqual(result.decoded_message_count, 2)
        self.assertTrue(any("without a later DISARM" in item for item in result.warnings))

    def test_parse_result_bounds_decoder_diagnostics(self) -> None:
        messages = [FakeMsg("GPS", {"Lat": 1.0, "Lng": 2.0})]

        def noisy_connection(_file_path: str) -> FakeLog:
            for index in range(5):
                print(f"bad header at {index}", file=sys.stderr)
            return FakeLog(messages)

        with patch(
            "app.parsers.binary.mavutil.mavlink_connection", side_effect=noisy_connection
        ):
            result = BinaryDataParser(max_diagnostic_lines=2).parse_with_diagnostics(
                __file__
            )

        self.assertEqual(result.status, ParseStatus.PARTIAL)
        self.assertEqual(result.diagnostics.total_lines, 5)
        self.assertEqual(len(result.diagnostics.captured_lines), 2)
        self.assertEqual(result.diagnostics.suppressed_lines, 3)

    def test_parse_result_rejects_file_when_decoder_fails_before_messages(self) -> None:
        with patch(
            "app.parsers.binary.mavutil.mavlink_connection",
            side_effect=ValueError("Invalid FMT length"),
        ):
            parser = BinaryDataParser()
            result = parser.parse_with_diagnostics(__file__)

            self.assertEqual(result.status, ParseStatus.REJECTED)
            self.assertEqual(result.error, "Invalid FMT length")
            with self.assertRaises(BinaryParseError):
                parser.parse(__file__)

    def test_parse_result_marks_trusted_size_mismatch_partial(self) -> None:
        messages = [FakeMsg("GPS", {"Lat": 1.0, "Lng": 2.0})]
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "flight.BIN"
            file_path.write_bytes(b"test-data")

            with patch(
                "app.parsers.binary.mavutil.mavlink_connection",
                return_value=FakeLog(messages),
            ):
                result = BinaryDataParser().parse_with_diagnostics(
                    str(file_path),
                    expected_size_bytes=100,
                )

        self.assertEqual(result.status, ParseStatus.PARTIAL)
        self.assertEqual(result.artifact_size_bytes, 9)
        self.assertTrue(any("File size mismatch" in item for item in result.warnings))


if __name__ == "__main__":
    unittest.main()
