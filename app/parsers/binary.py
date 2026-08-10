import contextlib
import hashlib
import io
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from pymavlink import mavutil

from app.parsers.base import DataParser, ParseDiagnostics, ParseResult, ParseStatus


class BinaryParseError(ValueError):
    """Raised by the legacy parse API when a BIN file cannot be decoded."""


class _BoundedDiagnosticStream(io.TextIOBase):
    """Collect decoder stderr without allowing unbounded output amplification."""

    def __init__(self, max_lines: int) -> None:
        super().__init__()
        self.max_lines = max_lines
        self.total_lines = 0
        self._captured: list[str] = []
        self._pending = ""

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not value:
            return 0

        self._pending += value
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._record(line)
        return len(value)

    def flush(self) -> None:
        return None

    def finish(self) -> None:
        if self._pending:
            self._record(self._pending)
            self._pending = ""

    def _record(self, line: str) -> None:
        self.total_lines += 1
        if len(self._captured) < self.max_lines:
            self._captured.append(line.rstrip("\r"))

    def as_diagnostics(self) -> ParseDiagnostics:
        self.finish()
        return ParseDiagnostics(
            total_lines=self.total_lines,
            captured_lines=tuple(self._captured),
            suppressed_lines=max(0, self.total_lines - len(self._captured)),
        )


# redirect_stderr changes process-global state. Serializing this small critical section
# prevents concurrent web requests from capturing each other's decoder diagnostics.
_STDERR_CAPTURE_LOCK = threading.Lock()

ANALYSIS_MESSAGE_TYPES = frozenset({"GPS", "IMU", "ATT", "MSG", "EV", "MODE", "ERR"})


class BinaryDataParser(DataParser):
    def __init__(
        self,
        max_diagnostic_lines: int = 100,
        message_types: Iterable[str] | None = ANALYSIS_MESSAGE_TYPES,
    ) -> None:
        if max_diagnostic_lines < 0:
            raise ValueError("max_diagnostic_lines must be non-negative")
        self.max_diagnostic_lines = max_diagnostic_lines
        self.message_types = (
            None if message_types is None else tuple(sorted(set(message_types)))
        )

    def parse(self, file_path: str) -> dict[str, pd.DataFrame]:
        result = self.parse_with_diagnostics(file_path)
        if result.status is ParseStatus.REJECTED:
            raise BinaryParseError(result.error or "BIN file could not be decoded")
        return result.dataframes

    def parse_with_diagnostics(
        self,
        file_path: str,
        *,
        expected_size_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> ParseResult:
        capture = _BoundedDiagnosticStream(self.max_diagnostic_lines)
        data: dict[str, list[dict[str, Any]]] = {}
        decoded_message_count = 0
        saw_arm = False
        armed_at_end = False
        parse_error: str | None = None
        artifact_size_bytes: int | None = None
        artifact_sha256: str | None = None

        try:
            artifact_size_bytes, artifact_sha256 = self._fingerprint(file_path)
            with _STDERR_CAPTURE_LOCK:
                previous_fast_index = os.environ.get("PYMAVLINK_FAST_INDEX")
                os.environ["PYMAVLINK_FAST_INDEX"] = "0"
                try:
                    with contextlib.redirect_stderr(capture):
                        mlog = None
                        try:
                            mlog = mavutil.mavlink_connection(file_path)

                            while True:
                                if self.message_types is None:
                                    msg = mlog.recv_match()
                                else:
                                    msg = mlog.recv_match(
                                        type=list(self.message_types),
                                        strict=True,
                                    )
                                if msg is None:
                                    break

                                msg_type = msg.get_type()
                                if msg_type == "FMT":
                                    continue
                                if (
                                    self.message_types is not None
                                    and msg_type not in self.message_types
                                ):
                                    continue

                                msg_dict = msg.to_dict()
                                msg_dict.pop("mavpackettype", None)
                                data.setdefault(msg_type, []).append(msg_dict)
                                decoded_message_count += 1

                                arm_state = self._arming_state(msg_type, msg_dict)
                                if arm_state is True:
                                    saw_arm = True
                                    armed_at_end = True
                                elif arm_state is False:
                                    armed_at_end = False
                        finally:
                            if mlog is not None:
                                mlog.close()
                finally:
                    if previous_fast_index is None:
                        os.environ.pop("PYMAVLINK_FAST_INDEX", None)
                    else:
                        os.environ["PYMAVLINK_FAST_INDEX"] = previous_fast_index
        except Exception as exc:
            parse_error = str(exc) or type(exc).__name__

        diagnostics = capture.as_diagnostics()
        dataframes = {
            msg_type: pd.DataFrame(messages) for msg_type, messages in data.items()
        }

        warnings: list[str] = []
        if diagnostics.total_lines:
            warnings.append(
                f"Decoder reported {diagnostics.total_lines} diagnostic line(s); "
                f"captured {len(diagnostics.captured_lines)}, "
                f"suppressed {diagnostics.suppressed_lines}."
            )
        if saw_arm and armed_at_end:
            warnings.append(
                "The log contains an ARM event without a later DISARM event; "
                "the recording may be truncated or incomplete."
            )
        if parse_error:
            warnings.append(f"Decoder stopped early: {parse_error}")

        provenance_mismatch = False
        if expected_size_bytes is not None and artifact_size_bytes != expected_size_bytes:
            provenance_mismatch = True
            warnings.append(
                f"File size mismatch: expected {expected_size_bytes} bytes, "
                f"got {artifact_size_bytes} bytes."
            )
        if expected_sha256 is not None:
            normalized_expected_sha256 = expected_sha256.casefold()
            if artifact_sha256 != normalized_expected_sha256:
                provenance_mismatch = True
                warnings.append(
                    "SHA-256 mismatch: the file does not match the trusted artifact."
                )

        if parse_error and decoded_message_count == 0:
            status = ParseStatus.REJECTED
        elif (
            diagnostics.total_lines
            or parse_error
            or provenance_mismatch
            or (saw_arm and armed_at_end)
        ):
            status = ParseStatus.PARTIAL
        elif decoded_message_count == 0:
            status = ParseStatus.REJECTED
            parse_error = "No decodable messages were found"
            warnings.append(parse_error)
        else:
            status = ParseStatus.COMPLETE

        return ParseResult(
            dataframes=dataframes,
            status=status,
            warnings=tuple(warnings),
            diagnostics=diagnostics,
            decoded_message_count=decoded_message_count,
            error=parse_error,
            artifact_size_bytes=artifact_size_bytes,
            artifact_sha256=artifact_sha256,
        )

    @staticmethod
    def _fingerprint(file_path: str) -> tuple[int, str]:
        path = Path(file_path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return path.stat().st_size, digest.hexdigest()

    @staticmethod
    def _arming_state(msg_type: str, payload: dict[str, Any]) -> bool | None:
        if msg_type == "MSG":
            message = str(payload.get("Message", "")).casefold()
            if "disarming motors" in message:
                return False
            if "arming motors" in message:
                return True

        if msg_type == "EV":
            event_id = payload.get("Id", payload.get("ID"))
            if event_id == 10:
                return True
            if event_id == 11:
                return False

        return None
