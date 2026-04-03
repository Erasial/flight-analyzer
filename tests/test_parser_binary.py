import unittest
from unittest.mock import patch

import pandas as pd

from app.parsers.binary import BinaryDataParser


class FakeMsg:
    def __init__(self, msg_type: str, payload: dict):
        self._msg_type = msg_type
        self._payload = payload

    def get_type(self) -> str:
        return self._msg_type

    def to_dict(self) -> dict:
        return dict(self._payload)


class FakeLog:
    def __init__(self, messages):
        self._messages = iter(messages)

    def recv_match(self):
        return next(self._messages, None)


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
            data = parser.parse("dummy.BIN")

        self.assertIn("GPS", data)
        self.assertIn("IMU", data)
        self.assertNotIn("FMT", data)
        self.assertIsInstance(data["GPS"], pd.DataFrame)
        self.assertEqual(len(data["GPS"]), 2)
        self.assertNotIn("mavpackettype", data["GPS"].columns)


if __name__ == "__main__":
    unittest.main()
