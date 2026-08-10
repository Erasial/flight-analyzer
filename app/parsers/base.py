from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd


class ParseStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ParseDiagnostics:
    total_lines: int = 0
    captured_lines: tuple[str, ...] = ()
    suppressed_lines: int = 0


@dataclass(frozen=True)
class ParseResult:
    dataframes: dict[str, pd.DataFrame]
    status: ParseStatus
    warnings: tuple[str, ...] = ()
    diagnostics: ParseDiagnostics = field(default_factory=ParseDiagnostics)
    decoded_message_count: int = 0
    error: str | None = None
    artifact_size_bytes: int | None = None
    artifact_sha256: str | None = None


class DataParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> dict[str, pd.DataFrame]:
        ...
