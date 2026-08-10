import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class EventSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EventCategory(StrEnum):
    FLIGHT = "flight"
    MODE = "mode"
    FAILSAFE = "failsafe"
    NAVIGATION = "navigation"
    POWER = "power"
    SYSTEM = "system"


class VehicleProfile(StrEnum):
    COPTER = "copter"
    ROCKET = "rocket"
    UNKNOWN = "unknown"


class EventKind(StrEnum):
    ARM = "arm"
    DISARM = "disarm"
    TAKEOFF = "takeoff"
    GROUND_CONTACT = "ground_contact"
    MODE_CHANGE = "mode_change"
    RC_FAILSAFE = "rc_failsafe"
    RC_FAILSAFE_CLEARED = "rc_failsafe_cleared"
    GCS_FAILSAFE = "gcs_failsafe"
    GCS_FAILSAFE_CLEARED = "gcs_failsafe_cleared"
    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    BATTERY_FAILSAFE = "battery_failsafe"
    BATTERY_FAILSAFE_CLEARED = "battery_failsafe_cleared"
    EKF_FAILSAFE = "ekf_failsafe"
    EKF_FAILSAFE_CLEARED = "ekf_failsafe_cleared"
    TERRAIN_FAILSAFE = "terrain_failsafe"
    TERRAIN_FAILSAFE_CLEARED = "terrain_failsafe_cleared"
    VIBRATION_FAILSAFE = "vibration_failsafe"
    VIBRATION_FAILSAFE_CLEARED = "vibration_failsafe_cleared"
    DEAD_RECKONING_FAILSAFE = "dead_reckoning_failsafe"
    DEAD_RECKONING_FAILSAFE_CLEARED = "dead_reckoning_failsafe_cleared"
    FENCE_BREACH = "fence_breach"
    FENCE_CLEARED = "fence_cleared"
    GPS_FIX_LOST = "gps_fix_lost"
    AUTOPILOT_ERROR = "autopilot_error"
    AUTOPILOT_ERROR_CLEARED = "autopilot_error_cleared"
    ROCKET_STAGE = "rocket_stage"
    PARACHUTE_RELEASE = "parachute_release"


COPTER_MODE_NAMES = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    9: "LAND",
    11: "DRIFT",
    13: "SPORT",
    14: "FLIP",
    15: "AUTOTUNE",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    19: "AVOID_ADSB",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL",
    22: "FLOWHOLD",
    23: "FOLLOW",
    24: "ZIGZAG",
    25: "SYSTEMID",
    26: "AUTOROTATE",
    27: "AUTO_RTL",
    28: "TURTLE",
}

MODE_REASON_NAMES = {
    1: "initialization",
    2: "operator_or_gcs_command",
    3: "radio_failsafe",
    4: "battery_failsafe",
    5: "gcs_failsafe",
    6: "ekf_failsafe",
    10: "fence_breach",
}

ERROR_SUBSYSTEM_NAMES = {
    1: "MAIN",
    2: "RADIO",
    3: "COMPASS",
    4: "OPTFLOW",
    5: "FAILSAFE_RADIO",
    6: "FAILSAFE_BATT",
    7: "FAILSAFE_GPS",
    8: "FAILSAFE_GCS",
    9: "FAILSAFE_FENCE",
    10: "FLIGHT_MODE",
    11: "GPS",
    12: "CRASH_CHECK",
    15: "PARACHUTES",
    16: "EKFCHECK",
    17: "FAILSAFE_EKFINAV",
    18: "BARO",
    19: "CPU",
    21: "TERRAIN",
    23: "FAILSAFE_TERRAIN",
    25: "THRUST_LOSS_CHECK",
    29: "FAILSAFE_VIBE",
    30: "INTERNAL_ERROR",
    31: "FAILSAFE_DEADRECKON",
}

FAILSAFE_ERROR_KINDS = {
    5: (EventKind.RC_FAILSAFE, EventKind.RC_FAILSAFE_CLEARED, "RC failsafe"),
    6: (
        EventKind.BATTERY_FAILSAFE,
        EventKind.BATTERY_FAILSAFE_CLEARED,
        "Battery failsafe",
    ),
    8: (EventKind.GCS_FAILSAFE, EventKind.GCS_FAILSAFE_CLEARED, "GCS failsafe"),
    9: (EventKind.FENCE_BREACH, EventKind.FENCE_CLEARED, "Fence breach"),
    17: (EventKind.EKF_FAILSAFE, EventKind.EKF_FAILSAFE_CLEARED, "EKF failsafe"),
    23: (
        EventKind.TERRAIN_FAILSAFE,
        EventKind.TERRAIN_FAILSAFE_CLEARED,
        "Terrain failsafe",
    ),
    29: (
        EventKind.VIBRATION_FAILSAFE,
        EventKind.VIBRATION_FAILSAFE_CLEARED,
        "Vibration failsafe",
    ),
    31: (
        EventKind.DEAD_RECKONING_FAILSAFE,
        EventKind.DEAD_RECKONING_FAILSAFE_CLEARED,
        "Dead-reckoning failsafe",
    ),
}


@dataclass(frozen=True)
class FlightEvent:
    time_us: int
    relative_time_s: float
    kind: EventKind
    category: EventCategory
    severity: EventSeverity
    title: str
    source: str
    details: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_us": self.time_us,
            "relative_time_s": self.relative_time_s,
            "kind": self.kind.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "source": self.source,
            "details": self.details,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class FlightSegment:
    index: int
    arm_time_us: int
    disarm_time_us: int | None
    duration_s: float | None
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "arm_time_us": self.arm_time_us,
            "disarm_time_us": self.disarm_time_us,
            "duration_s": self.duration_s,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class FlightEventReport:
    events: tuple[FlightEvent, ...] = ()
    segments: tuple[FlightSegment, ...] = ()
    warnings: tuple[str, ...] = ()
    vehicle_profile: VehicleProfile = VehicleProfile.UNKNOWN
    firmware: str | None = None

    @property
    def critical_count(self) -> int:
        return sum(event.severity is EventSeverity.CRITICAL for event in self.events)

    @property
    def warning_count(self) -> int:
        return sum(event.severity is EventSeverity.WARNING for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "vehicle_profile": self.vehicle_profile.value,
                "firmware": self.firmware,
                "event_count": len(self.events),
                "flight_segment_count": len(self.segments),
                "complete_segment_count": sum(segment.complete for segment in self.segments),
                "critical_count": self.critical_count,
                "warning_count": self.warning_count,
            },
            "events": [event.to_dict() for event in self.events],
            "segments": [segment.to_dict() for segment in self.segments],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _PendingEvent:
    time_us: int
    kind: EventKind
    category: EventCategory
    severity: EventSeverity
    title: str
    source: str
    details: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def _numeric_time(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric < 0:
        return None
    return int(numeric)


def detect_vehicle_profile(messages: pd.DataFrame) -> tuple[VehicleProfile, str | None]:
    if messages.empty or "Message" not in messages.columns:
        return VehicleProfile.UNKNOWN, None
    for raw_message in messages["Message"].astype(str):
        message = raw_message.strip()
        normalized = message.casefold()
        if normalized.startswith("therocket "):
            return VehicleProfile.ROCKET, message
        if normalized.startswith("arducopter "):
            return VehicleProfile.COPTER, message
    return VehicleProfile.UNKNOWN, None


def _message_event(time_us: int, message: str) -> _PendingEvent | None:
    normalized = message.casefold().strip()
    if normalized.startswith(("prearm:", "gcs:", "src=")):
        return None
    evidence = {"message": message}

    stage_match = re.search(r"fstg:\s*([a-z_]+)\s*->\s*([a-z_]+)", normalized)
    if stage_match:
        source_stage = stage_match.group(1).upper()
        target_stage = stage_match.group(2).upper()
        return _PendingEvent(
            time_us,
            EventKind.ROCKET_STAGE,
            EventCategory.FLIGHT,
            EventSeverity.INFO,
            f"Rocket stage changed to {target_stage}",
            "MSG",
            evidence={
                **evidence,
                "from_stage": source_stage,
                "to_stage": target_stage,
            },
        )
    if normalized == "parachute: released":
        return _PendingEvent(
            time_us,
            EventKind.PARACHUTE_RELEASE,
            EventCategory.FLIGHT,
            EventSeverity.INFO,
            "Parachute released",
            "MSG",
            evidence=evidence,
        )

    if "disarming motors" in normalized:
        return _PendingEvent(
            time_us,
            EventKind.DISARM,
            EventCategory.FLIGHT,
            EventSeverity.INFO,
            "Motors disarmed",
            "MSG",
            evidence=evidence,
        )
    if "arming motors" in normalized:
        return _PendingEvent(
            time_us,
            EventKind.ARM,
            EventCategory.FLIGHT,
            EventSeverity.INFO,
            "Motors armed",
            "MSG",
            evidence=evidence,
        )

    ground_match = re.search(r"sim hit ground at ([0-9.]+)\s*m/s", normalized)
    if ground_match:
        speed = float(ground_match.group(1))
        return _PendingEvent(
            time_us,
            EventKind.GROUND_CONTACT,
            EventCategory.FLIGHT,
            EventSeverity.INFO,
            "SITL ground contact",
            "MSG",
            "Simulation-only ground-contact evidence; not a real-flight impact detector.",
            {**evidence, "vertical_speed_m_s": speed, "simulation_only": True},
        )

    cleared = "cleared" in normalized
    if normalized.startswith("radio failsafe"):
        return _failsafe_message(
            time_us,
            EventKind.RC_FAILSAFE_CLEARED if cleared else EventKind.RC_FAILSAFE,
            "RC failsafe cleared" if cleared else "RC failsafe activated",
            message,
            cleared,
        )
    if normalized.startswith("gcs failsafe"):
        return _failsafe_message(
            time_us,
            EventKind.GCS_FAILSAFE_CLEARED if cleared else EventKind.GCS_FAILSAFE,
            "GCS failsafe cleared" if cleared else "GCS failsafe activated",
            message,
            cleared,
        )
    if normalized.startswith("ekf failsafe"):
        return _failsafe_message(
            time_us,
            EventKind.EKF_FAILSAFE_CLEARED if cleared else EventKind.EKF_FAILSAFE,
            "EKF failsafe cleared" if cleared else "EKF failsafe activated",
            message,
            cleared,
        )
    if normalized.startswith("terrain failsafe"):
        return _failsafe_message(
            time_us,
            EventKind.TERRAIN_FAILSAFE_CLEARED if cleared else EventKind.TERRAIN_FAILSAFE,
            "Terrain failsafe cleared" if cleared else "Terrain failsafe activated",
            message,
            cleared,
        )
    if normalized.startswith("vibration failsafe"):
        return _failsafe_message(
            time_us,
            EventKind.VIBRATION_FAILSAFE_CLEARED if cleared else EventKind.VIBRATION_FAILSAFE,
            "Vibration failsafe cleared" if cleared else "Vibration failsafe activated",
            message,
            cleared,
        )
    if normalized.startswith("battery failsafe"):
        return _failsafe_message(
            time_us,
            EventKind.BATTERY_FAILSAFE_CLEARED if cleared else EventKind.BATTERY_FAILSAFE,
            "Battery failsafe cleared" if cleared else "Battery failsafe activated",
            message,
            cleared,
        )
    if "battery" in normalized and "critical" in normalized:
        return _PendingEvent(
            time_us,
            EventKind.BATTERY_CRITICAL,
            EventCategory.POWER,
            EventSeverity.CRITICAL,
            "Critical battery condition",
            "MSG",
            evidence=evidence,
        )
    if "battery" in normalized and "low" in normalized:
        return _PendingEvent(
            time_us,
            EventKind.BATTERY_LOW,
            EventCategory.POWER,
            EventSeverity.WARNING,
            "Low battery condition",
            "MSG",
            evidence=evidence,
        )
    if normalized.startswith("fence breach cleared"):
        return _PendingEvent(
            time_us,
            EventKind.FENCE_CLEARED,
            EventCategory.NAVIGATION,
            EventSeverity.INFO,
            "Geofence breach cleared",
            "MSG",
            evidence=evidence,
        )
    if "fence" in normalized and normalized.endswith("breached"):
        return _PendingEvent(
            time_us,
            EventKind.FENCE_BREACH,
            EventCategory.NAVIGATION,
            EventSeverity.CRITICAL,
            "Geofence breached",
            "MSG",
            evidence=evidence,
        )
    return None


def _failsafe_message(
    time_us: int,
    kind: EventKind,
    title: str,
    message: str,
    cleared: bool,
) -> _PendingEvent:
    return _PendingEvent(
        time_us,
        kind,
        EventCategory.FAILSAFE,
        EventSeverity.INFO if cleared else EventSeverity.CRITICAL,
        title,
        "MSG",
        evidence={"message": message},
    )


def _deduplicate(events: list[_PendingEvent], tolerance_us: int = 1_000) -> list[_PendingEvent]:
    result: list[_PendingEvent] = []
    for event in sorted(events, key=lambda item: item.time_us):
        duplicate = any(
            previous.kind is event.kind and abs(previous.time_us - event.time_us) <= tolerance_us
            for previous in result[-5:]
        )
        if not duplicate:
            result.append(event)
    return result


def _segments(events: list[_PendingEvent]) -> tuple[FlightSegment, ...]:
    segments: list[FlightSegment] = []
    active_arm: int | None = None
    for event in events:
        if event.kind is EventKind.ARM and active_arm is None:
            active_arm = event.time_us
        elif event.kind is EventKind.DISARM and active_arm is not None:
            segments.append(
                FlightSegment(
                    index=len(segments) + 1,
                    arm_time_us=active_arm,
                    disarm_time_us=event.time_us,
                    duration_s=(event.time_us - active_arm) / 1e6,
                    complete=True,
                )
            )
            active_arm = None
    if active_arm is not None:
        segments.append(
            FlightSegment(
                index=len(segments) + 1,
                arm_time_us=active_arm,
                disarm_time_us=None,
                duration_s=None,
                complete=False,
            )
        )
    return tuple(segments)


def detect_flight_events(dataframes: dict[str, pd.DataFrame]) -> FlightEventReport:
    pending: list[_PendingEvent] = []
    warnings: list[str] = []

    messages = dataframes.get("MSG", pd.DataFrame())
    vehicle_profile, firmware = detect_vehicle_profile(messages)
    if not messages.empty and {"TimeUS", "Message"}.issubset(messages.columns):
        for record in messages.to_dict(orient="records"):
            time_us = _numeric_time(record.get("TimeUS"))
            if time_us is None:
                continue
            event = _message_event(time_us, str(record.get("Message", "")))
            if event is not None:
                pending.append(event)

    modes = dataframes.get("MODE", pd.DataFrame())
    if not modes.empty and "TimeUS" in modes.columns:
        for record in modes.to_dict(orient="records"):
            time_us = _numeric_time(record.get("TimeUS"))
            mode_value = _numeric_time(record.get("ModeNum", record.get("Mode")))
            reason = _numeric_time(record.get("Rsn"))
            if time_us is None or mode_value is None:
                continue
            if vehicle_profile is VehicleProfile.ROCKET:
                mode_name = f"CUSTOM_MODE_{mode_value}"
                reason_name = f"custom_reason_{reason}"
            elif vehicle_profile is VehicleProfile.COPTER:
                mode_name = COPTER_MODE_NAMES.get(mode_value, f"MODE_{mode_value}")
                reason_name = MODE_REASON_NAMES.get(reason or -1, f"reason_{reason}")
            else:
                mode_name = f"MODE_{mode_value}"
                reason_name = MODE_REASON_NAMES.get(reason or -1, f"reason_{reason}")
            pending.append(
                _PendingEvent(
                    time_us,
                    EventKind.MODE_CHANGE,
                    EventCategory.MODE,
                    EventSeverity.WARNING if reason in {3, 4, 5, 6, 10} else EventSeverity.INFO,
                    f"Mode changed to {mode_name}",
                    "MODE",
                    f"Mode reason: {reason_name}.",
                    {
                        "mode": mode_value,
                        "mode_name": mode_name,
                        "reason": reason,
                        "reason_name": reason_name,
                    },
                )
            )

    ev_records = dataframes.get("EV", pd.DataFrame())
    if not ev_records.empty and "TimeUS" in ev_records.columns:
        event_map = {
            10: (EventKind.ARM, "Motors armed"),
            11: (EventKind.DISARM, "Motors disarmed"),
            15: (EventKind.TAKEOFF, "Takeoff detected"),
        }
        if vehicle_profile is VehicleProfile.ROCKET:
            event_map[51] = (EventKind.PARACHUTE_RELEASE, "Parachute released")
        for record in ev_records.to_dict(orient="records"):
            time_us = _numeric_time(record.get("TimeUS"))
            event_id = _numeric_time(record.get("Id", record.get("ID")))
            if time_us is None or event_id not in event_map:
                continue
            kind, title = event_map[event_id]
            pending.append(
                _PendingEvent(
                    time_us,
                    kind,
                    EventCategory.FLIGHT,
                    EventSeverity.INFO,
                    title,
                    "EV",
                    evidence={"event_id": event_id},
                )
            )

    errors = dataframes.get("ERR", pd.DataFrame())
    if not errors.empty and "TimeUS" in errors.columns:
        active_error_subsystems: set[int] = set()
        for record in errors.to_dict(orient="records"):
            time_us = _numeric_time(record.get("TimeUS"))
            subsystem = _numeric_time(record.get("Subsys"))
            error_code = _numeric_time(record.get("ECode"))
            if time_us is None or subsystem is None or error_code is None:
                continue
            if error_code == 0 and subsystem not in active_error_subsystems:
                continue
            if error_code == 0:
                active_error_subsystems.discard(subsystem)
            else:
                active_error_subsystems.add(subsystem)
            subsystem_name = ERROR_SUBSYSTEM_NAMES.get(subsystem, f"SUBSYSTEM_{subsystem}")
            if subsystem in FAILSAFE_ERROR_KINDS:
                active_kind, cleared_kind, title = FAILSAFE_ERROR_KINDS[subsystem]
                cleared = error_code == 0
                pending.append(
                    _PendingEvent(
                        time_us,
                        cleared_kind if cleared else active_kind,
                        EventCategory.FAILSAFE,
                        EventSeverity.INFO if cleared else EventSeverity.CRITICAL,
                        f"{title} cleared" if cleared else f"{title} activated",
                        "ERR",
                        evidence={
                            "subsystem": subsystem,
                            "subsystem_name": subsystem_name,
                            "error_code": error_code,
                        },
                    )
                )
            else:
                cleared = error_code == 0
                pending.append(
                    _PendingEvent(
                        time_us,
                        EventKind.AUTOPILOT_ERROR_CLEARED if cleared else EventKind.AUTOPILOT_ERROR,
                        EventCategory.SYSTEM,
                        EventSeverity.INFO if cleared else EventSeverity.WARNING,
                        f"{subsystem_name} error cleared" if cleared else f"{subsystem_name} error",
                        "ERR",
                        evidence={
                            "subsystem": subsystem,
                            "subsystem_name": subsystem_name,
                            "error_code": error_code,
                        },
                    )
                )

    gps = dataframes.get("GPS", pd.DataFrame())
    if not gps.empty and {"TimeUS", "Status"}.issubset(gps.columns):
        gps_status = gps[["TimeUS", "Status"]].copy()
        gps_status["TimeUS"] = pd.to_numeric(gps_status["TimeUS"], errors="coerce")
        gps_status["Status"] = pd.to_numeric(gps_status["Status"], errors="coerce")
        gps_status = gps_status.dropna().sort_values("TimeUS")
        healthy_seen = False
        loss_reported = False
        previous_status: float | None = None
        for record in gps_status.to_dict(orient="records"):
            time_us = _numeric_time(record["TimeUS"])
            status = float(record["Status"])
            if time_us is None:
                continue
            if status >= 3:
                healthy_seen = True
                loss_reported = False
            elif healthy_seen and not loss_reported:
                pending.append(
                    _PendingEvent(
                        time_us,
                        EventKind.GPS_FIX_LOST,
                        EventCategory.NAVIGATION,
                        EventSeverity.CRITICAL,
                        "GPS fix degraded",
                        "GPS",
                        evidence={"previous_status": previous_status, "status": status},
                    )
                )
                loss_reported = True
            previous_status = status

    deduplicated = _deduplicate(pending)
    segments = _segments(deduplicated)
    if any(not segment.complete for segment in segments):
        warnings.append("At least one armed flight segment has no matching disarm event.")

    first_time_us = min((event.time_us for event in deduplicated), default=0)
    events = tuple(
        FlightEvent(
            time_us=event.time_us,
            relative_time_s=(event.time_us - first_time_us) / 1e6,
            kind=event.kind,
            category=event.category,
            severity=event.severity,
            title=event.title,
            source=event.source,
            details=event.details,
            evidence=event.evidence,
        )
        for event in deduplicated
    )
    return FlightEventReport(
        events=events,
        segments=segments,
        warnings=tuple(warnings),
        vehicle_profile=vehicle_profile,
        firmware=firmware,
    )
