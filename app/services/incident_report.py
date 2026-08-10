from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.services.event_detector import (
    EventKind,
    FlightEvent,
    FlightEventReport,
    FlightSegment,
)


class IncidentConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(StrEnum):
    RESOLVED = "resolved"
    COMPLETED_BY_DISARM = "completed_by_disarm"
    ONGOING = "ongoing"
    OBSERVED = "observed"


ACTIVATION_KINDS = {
    EventKind.RC_FAILSAFE,
    EventKind.GCS_FAILSAFE,
    EventKind.BATTERY_FAILSAFE,
    EventKind.EKF_FAILSAFE,
    EventKind.TERRAIN_FAILSAFE,
    EventKind.VIBRATION_FAILSAFE,
    EventKind.DEAD_RECKONING_FAILSAFE,
    EventKind.FENCE_BREACH,
}

CLEAR_KIND_BY_ACTIVATION = {
    EventKind.RC_FAILSAFE: EventKind.RC_FAILSAFE_CLEARED,
    EventKind.GCS_FAILSAFE: EventKind.GCS_FAILSAFE_CLEARED,
    EventKind.BATTERY_FAILSAFE: EventKind.BATTERY_FAILSAFE_CLEARED,
    EventKind.EKF_FAILSAFE: EventKind.EKF_FAILSAFE_CLEARED,
    EventKind.TERRAIN_FAILSAFE: EventKind.TERRAIN_FAILSAFE_CLEARED,
    EventKind.VIBRATION_FAILSAFE: EventKind.VIBRATION_FAILSAFE_CLEARED,
    EventKind.DEAD_RECKONING_FAILSAFE: EventKind.DEAD_RECKONING_FAILSAFE_CLEARED,
    EventKind.FENCE_BREACH: EventKind.FENCE_CLEARED,
}

EXPECTED_MODE_REASON = {
    EventKind.RC_FAILSAFE: 3,
    EventKind.BATTERY_FAILSAFE: 4,
    EventKind.GCS_FAILSAFE: 5,
    EventKind.EKF_FAILSAFE: 6,
    EventKind.FENCE_BREACH: 10,
}

PRECURSOR_KINDS = {
    EventKind.BATTERY_FAILSAFE: {EventKind.BATTERY_LOW, EventKind.BATTERY_CRITICAL},
    EventKind.EKF_FAILSAFE: {EventKind.GPS_FIX_LOST},
}

OBSERVATION_KINDS = {
    EventKind.GPS_FIX_LOST,
    EventKind.BATTERY_LOW,
    EventKind.BATTERY_CRITICAL,
}


@dataclass(frozen=True)
class FlightIncident:
    index: int
    incident_type: EventKind
    trigger_event: FlightEvent
    failsafe_event: FlightEvent | None
    action_event: FlightEvent | None
    clear_event: FlightEvent | None
    ground_contact_event: FlightEvent | None
    disarm_event: FlightEvent | None
    segment_index: int | None
    response_latency_s: float | None
    time_to_ground_s: float | None
    time_to_disarm_s: float | None
    status: IncidentStatus
    confidence: IncidentConfidence
    narrative: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "incident_type": self.incident_type.value,
            "segment_index": self.segment_index,
            "trigger": _event_reference(self.trigger_event),
            "failsafe": _event_reference(self.failsafe_event),
            "action": _event_reference(self.action_event),
            "clear": _event_reference(self.clear_event),
            "ground_contact": _event_reference(self.ground_contact_event),
            "disarm": _event_reference(self.disarm_event),
            "response_latency_s": self.response_latency_s,
            "time_to_ground_s": self.time_to_ground_s,
            "time_to_disarm_s": self.time_to_disarm_s,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "narrative": self.narrative,
        }


@dataclass(frozen=True)
class IncidentReport:
    incidents: tuple[FlightIncident, ...] = ()

    @property
    def unresolved_count(self) -> int:
        return sum(incident.status is IncidentStatus.ONGOING for incident in self.incidents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "incident_count": len(self.incidents),
                "unresolved_count": self.unresolved_count,
                "high_confidence_count": sum(
                    incident.confidence is IncidentConfidence.HIGH for incident in self.incidents
                ),
            },
            "incidents": [incident.to_dict() for incident in self.incidents],
        }


def _event_reference(event: FlightEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "time_us": event.time_us,
        "kind": event.kind.value,
        "title": event.title,
        "source": event.source,
        "evidence": event.evidence,
    }


def _segment_for_event(
    event: FlightEvent,
    segments: tuple[FlightSegment, ...],
) -> FlightSegment | None:
    for segment in segments:
        upper_bound = segment.disarm_time_us if segment.disarm_time_us is not None else float("inf")
        if segment.arm_time_us <= event.time_us <= upper_bound:
            return segment
    return None


def _first_event(
    events: tuple[FlightEvent, ...],
    *,
    kinds: set[EventKind],
    start_us: int,
    end_us: float,
) -> FlightEvent | None:
    return next(
        (event for event in events if event.kind in kinds and start_us <= event.time_us <= end_us),
        None,
    )


def _previous_event(
    events: tuple[FlightEvent, ...],
    *,
    kinds: set[EventKind],
    before_us: int,
    lower_bound_us: int,
) -> FlightEvent | None:
    matches = [
        event
        for event in events
        if event.kind in kinds and lower_bound_us <= event.time_us <= before_us
    ]
    return matches[-1] if matches else None


def _latency(start: FlightEvent, end: FlightEvent | None) -> float | None:
    if end is None:
        return None
    return (end.time_us - start.time_us) / 1e6


def _confidence(
    activation: FlightEvent,
    action: FlightEvent | None,
    clear: FlightEvent | None,
    disarm: FlightEvent | None,
) -> IncidentConfidence:
    explicit_activation = activation.source in {"MSG", "ERR"}
    if explicit_activation and (action is not None or clear is not None or disarm is not None):
        return IncidentConfidence.HIGH
    if explicit_activation:
        return IncidentConfidence.MEDIUM
    return IncidentConfidence.LOW


def _narrative(
    trigger: FlightEvent,
    activation: FlightEvent,
    action: FlightEvent | None,
    ground: FlightEvent | None,
    disarm: FlightEvent | None,
    clear: FlightEvent | None,
) -> str:
    parts: list[str] = []
    if trigger is not activation:
        precursor_s = (activation.time_us - trigger.time_us) / 1e6
        parts.append(f"{trigger.title} preceded {activation.title} by {precursor_s:.2f} s.")
    else:
        parts.append(f"{activation.title} at TimeUS {activation.time_us}.")
    if action is not None:
        latency_s = (action.time_us - activation.time_us) / 1e6
        mode_name = action.evidence.get("mode_name", "unknown mode")
        reason_name = action.evidence.get("reason_name", "unknown reason")
        parts.append(f"The autopilot selected {mode_name} after {latency_s:.2f} s ({reason_name}).")
    else:
        parts.append("No matching automatic mode transition was found.")
    if ground is not None:
        parts.append(
            f"SITL ground contact followed after "
            f"{(ground.time_us - activation.time_us) / 1e6:.2f} s."
        )
    if disarm is not None:
        parts.append(f"Disarm followed after {(disarm.time_us - activation.time_us) / 1e6:.2f} s.")
    if clear is not None:
        parts.append(
            f"The condition cleared after {(clear.time_us - activation.time_us) / 1e6:.2f} s."
        )
    return " ".join(parts)


def build_incident_report(event_report: FlightEventReport) -> IncidentReport:
    events = event_report.events
    incidents: list[FlightIncident] = []
    used_precursor_ids: set[int] = set()

    activations = [event for event in events if event.kind in ACTIVATION_KINDS]
    for activation_index, activation in enumerate(activations):
        segment = _segment_for_event(activation, event_report.segments)
        segment_start = (
            segment.arm_time_us if segment is not None else max(0, activation.time_us - 60_000_000)
        )
        segment_end = (
            segment.disarm_time_us
            if segment is not None and segment.disarm_time_us is not None
            else float("inf")
        )
        precursor = _previous_event(
            events,
            kinds=PRECURSOR_KINDS.get(activation.kind, set()),
            before_us=activation.time_us,
            lower_bound_us=max(segment_start, activation.time_us - 60_000_000),
        )
        trigger = precursor or activation
        if precursor is not None:
            used_precursor_ids.add(id(precursor))

        expected_reason = EXPECTED_MODE_REASON.get(activation.kind)
        action = next(
            (
                event
                for event in events
                if event.kind is EventKind.MODE_CHANGE
                and activation.time_us <= event.time_us <= activation.time_us + 2_000_000
                and (expected_reason is None or event.evidence.get("reason") == expected_reason)
            ),
            None,
        )
        clear_kind = CLEAR_KIND_BY_ACTIVATION[activation.kind]
        next_same_activation = next(
            (
                event
                for event in activations[activation_index + 1 :]
                if event.kind is activation.kind
            ),
            None,
        )
        clear_search_end = (
            next_same_activation.time_us - 1 if next_same_activation is not None else float("inf")
        )
        clear = _first_event(
            events,
            kinds={clear_kind},
            start_us=activation.time_us,
            end_us=clear_search_end,
        )
        outcome_end = segment_end
        next_activation_in_same_segment = (
            next_same_activation is not None and next_same_activation.time_us <= segment_end
        )
        if clear is not None and (next_activation_in_same_segment or action is None):
            outcome_end = min(outcome_end, clear.time_us)
        ground = _first_event(
            events,
            kinds={EventKind.GROUND_CONTACT},
            start_us=activation.time_us,
            end_us=outcome_end,
        )
        disarm = _first_event(
            events,
            kinds={EventKind.DISARM},
            start_us=activation.time_us,
            end_us=outcome_end,
        )

        if clear is not None:
            status = IncidentStatus.RESOLVED
        elif disarm is not None:
            status = IncidentStatus.COMPLETED_BY_DISARM
        else:
            status = IncidentStatus.ONGOING

        incidents.append(
            FlightIncident(
                index=len(incidents) + 1,
                incident_type=activation.kind,
                trigger_event=trigger,
                failsafe_event=activation,
                action_event=action,
                clear_event=clear,
                ground_contact_event=ground,
                disarm_event=disarm,
                segment_index=segment.index if segment is not None else None,
                response_latency_s=_latency(activation, action),
                time_to_ground_s=_latency(activation, ground),
                time_to_disarm_s=_latency(activation, disarm),
                status=status,
                confidence=_confidence(activation, action, clear, disarm),
                narrative=_narrative(trigger, activation, action, ground, disarm, clear),
            )
        )

    for observation in events:
        if observation.kind not in OBSERVATION_KINDS or id(observation) in used_precursor_ids:
            continue
        segment = _segment_for_event(observation, event_report.segments)
        incidents.append(
            FlightIncident(
                index=len(incidents) + 1,
                incident_type=observation.kind,
                trigger_event=observation,
                failsafe_event=None,
                action_event=None,
                clear_event=None,
                ground_contact_event=None,
                disarm_event=None,
                segment_index=segment.index if segment is not None else None,
                response_latency_s=None,
                time_to_ground_s=None,
                time_to_disarm_s=None,
                status=IncidentStatus.OBSERVED,
                confidence=IncidentConfidence.MEDIUM,
                narrative=(
                    f"{observation.title} at TimeUS {observation.time_us}; "
                    "no matching supported failsafe activation was found."
                ),
            )
        )

    ordered = sorted(incidents, key=lambda incident: incident.trigger_event.time_us)
    reindexed = tuple(
        FlightIncident(
            index=index,
            incident_type=incident.incident_type,
            trigger_event=incident.trigger_event,
            failsafe_event=incident.failsafe_event,
            action_event=incident.action_event,
            clear_event=incident.clear_event,
            ground_contact_event=incident.ground_contact_event,
            disarm_event=incident.disarm_event,
            segment_index=incident.segment_index,
            response_latency_s=incident.response_latency_s,
            time_to_ground_s=incident.time_to_ground_s,
            time_to_disarm_s=incident.time_to_disarm_s,
            status=incident.status,
            confidence=incident.confidence,
            narrative=incident.narrative,
        )
        for index, incident in enumerate(ordered, start=1)
    )
    return IncidentReport(incidents=reindexed)
