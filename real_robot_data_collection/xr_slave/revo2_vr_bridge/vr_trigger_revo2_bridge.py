#!/usr/bin/env python3
"""Map XR triggers or absolute six-joint targets to BrainCo Revo2 hands.

The node subscribes on the robot (slave) to the two Float64 gripper-control
topics already transported by XR teleoperation.  The index trigger moves the
thumb tip, index, and middle finger into a calibrated three-digit grasp.  The
controller grip trigger arrives over a small, source-restricted UDP bridge and
independently rotates the thumb root.  Ring and pinky remain at zero.
For recorded trajectories it also accepts six normalized absolute joints on
``/revo2/absolute/{left,right}``.  The same process remains the sole Modbus
owner and publishes motor, tactile, and successfully submitted command samples
as versioned JSON ROS messages.

Safety properties:

* both devices are identified by their Revo2 serial number and handedness;
* no movement is sent before the first valid trigger or absolute sample;
* stale/missing input holds that control axis at its last position;
* simultaneous fresh trigger and absolute inputs stop the bridge;
* commands are rate-limited to at most 10 Hz and ignore tiny input changes;
* ``--dry-run`` validates ROS input without opening either serial device.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import math
import os
import re
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable


LOGGER = logging.getLogger("revo2_vr_bridge")
SDK_TRANSPORT_LOGGER_NAME = "bc_stark_sdk.modbus.device_ctx"
BROKEN_PIPE_LOG_MARKER = "Transport(Kind(BrokenPipe))"

ACTUATOR_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
TACTILE_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
ALL_TACTILE_FINGERS_ENABLED = 0x1F
STATE_SCHEMA = "revo2_joint_state_v1"
TACTILE_SCHEMA = "revo2_tactile_v1"
COMMAND_SCHEMA = "revo2_joint_command_v1"
ABSOLUTE_COMMAND_SCHEMA = "revo2_absolute_joint_command_v1"
GRIP_UDP_SCHEMA = "revo2_vr_grip_v1"
ABSOLUTE_COMMAND_TOPICS = {
    "left": "/revo2/absolute/left",
    "right": "/revo2/absolute/right",
}
PUBLISH_TOPIC_TEMPLATES = {
    "state": "/revo2/state/{side}",
    "tactile": "/revo2/tactile/{side}",
    "command": "/revo2/command/{side}",
    "absolute_command": "/revo2/absolute_ack/{side}",
}

OPEN_POSITION = (0, 0, 0, 0, 0, 0)
# Revo2 SDK motor order: thumb tip/flex, thumb root/opposition, index, middle,
# ring, pinky.  The index trigger owns only indices 0, 2, and 3.  The grip
# trigger independently owns index 1; index 5 is pinky and must not be confused
# with thumb rotation.
GRASP_PREPOSITION = {
    "left": (400, 0, 0, 0, 0, 0),
    "right": (400, 0, 0, 0, 0, 0),
}
GRASP_CLOSE_POSITION = {
    "left": (500, 0, 593, 593, 0, 0),
    "right": (517, 0, 593, 593, 0, 0),
}
# 814 is the already-tested lowered-thumb opposition endpoint from the prior
# mapping.  It is now reached continuously by the independent grip trigger.
THUMB_ROTATION_CLOSE_POSITION = {
    "left": 814,
    "right": 814,
}
GRASP_PREPOSITION_END = 0.40

DEFAULT_PORT_GLOB = (
    "/dev/serial/by-path/"
    "pci-0000:c[57]:00.3-usb-*:1.0-port0"
)


@dataclass(frozen=True)
class HandSpec:
    side: str
    topic: str
    slave_id: int
    serial_number: str
    hand_type: str


HAND_SPECS = (
    HandSpec(
        side="left",
        topic="/teleoperation/slave/left_gripper_control",
        slave_id=126,
        serial_number=os.environ.get("REVO2_LEFT_SERIAL", "").strip(),
        hand_type="left",
    ),
    HandSpec(
        side="right",
        topic="/teleoperation/slave/right_gripper_control",
        slave_id=127,
        serial_number=os.environ.get("REVO2_RIGHT_SERIAL", "").strip(),
        hand_type="right",
    ),
)


@dataclass
class TriggerSample:
    value: float
    received_at: float


@dataclass
class AbsoluteJointSample:
    positions: tuple[int, int, int, int, int, int]
    received_at: float


class TriggerStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, TriggerSample] = {}
        self._grip_samples: dict[str, TriggerSample] = {}
        self._absolute_samples: dict[str, AbsoluteJointSample] = {}

    def update(self, side: str, value: float) -> None:
        if not math.isfinite(value):
            LOGGER.warning("Ignoring non-finite %s trigger value: %r", side, value)
            return
        with self._lock:
            self._samples[side] = TriggerSample(value=value, received_at=time.monotonic())

    def snapshot(self) -> dict[str, TriggerSample]:
        with self._lock:
            return {
                side: TriggerSample(sample.value, sample.received_at)
                for side, sample in self._samples.items()
            }

    def update_grip(self, side: str, value: float) -> None:
        if side not in {"left", "right"}:
            LOGGER.warning("Ignoring grip sample for unknown side: %r", side)
            return
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            LOGGER.warning("Ignoring invalid %s grip value: %r", side, value)
            return
        with self._lock:
            self._grip_samples[side] = TriggerSample(
                value=value,
                received_at=time.monotonic(),
            )

    def snapshot_grip(self) -> dict[str, TriggerSample]:
        with self._lock:
            return {
                side: TriggerSample(sample.value, sample.received_at)
                for side, sample in self._grip_samples.items()
            }

    def update_absolute(self, side: str, values: Iterable[Any]) -> None:
        try:
            raw = _require_vector("absolute positions", values, len(ACTUATOR_NAMES))
            positions = []
            for value in raw:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("absolute positions must be finite integers")
                rounded = round(numeric)
                if abs(numeric - rounded) > 1e-6:
                    raise ValueError("absolute positions must be finite integers")
                if not 0 <= rounded <= 1000:
                    raise ValueError("absolute positions must be in 0..1000")
                positions.append(rounded)
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Ignoring invalid %s absolute command: %s", side, exc)
            return
        with self._lock:
            self._absolute_samples[side] = AbsoluteJointSample(
                tuple(positions),  # type: ignore[arg-type]
                time.monotonic(),
            )

    def snapshot_absolute(self) -> dict[str, AbsoluteJointSample]:
        with self._lock:
            return {
                side: AbsoluteJointSample(sample.positions, sample.received_at)
                for side, sample in self._absolute_samples.items()
            }

    def snapshot_all(
        self,
    ) -> tuple[
        dict[str, TriggerSample],
        dict[str, TriggerSample],
        dict[str, AbsoluteJointSample],
    ]:
        """Return all input families from one lock acquisition."""
        with self._lock:
            triggers = {
                side: TriggerSample(sample.value, sample.received_at)
                for side, sample in self._samples.items()
            }
            grips = {
                side: TriggerSample(sample.value, sample.received_at)
                for side, sample in self._grip_samples.items()
            }
            absolute = {
                side: AbsoluteJointSample(sample.positions, sample.received_at)
                for side, sample in self._absolute_samples.items()
            }
        return triggers, grips, absolute


def parse_grip_udp_packet(data: bytes) -> tuple[str, float, int, str]:
    """Validate one versioned grip datagram without trusting its sender."""
    if len(data) > 512:
        raise ValueError("grip datagram is too large")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("grip datagram is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != GRIP_UDP_SCHEMA:
        raise ValueError("unsupported grip datagram schema")
    side = payload.get("side")
    if side not in {"left", "right"}:
        raise ValueError("grip datagram side must be left or right")
    try:
        value = float(payload["grip"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("grip datagram value is invalid") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("grip datagram value must be finite and in 0..1")
    sequence = payload.get("seq")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("grip datagram sequence must be a non-negative integer")
    session = payload.get("session")
    if not isinstance(session, str) or not re.fullmatch(r"[0-9a-f]{16}", session):
        raise ValueError("grip datagram session is invalid")
    return side, value, sequence, session


@dataclass
class GripUdpRuntime:
    sock: socket.socket
    stop_event: threading.Event
    thread: threading.Thread

    def close(self) -> None:
        self.stop_event.set()
        self.sock.close()
        self.thread.join(timeout=2.0)


def start_grip_udp_runtime(
    store: TriggerStore,
    bind_address: str,
    port: int,
    source_ip: str,
    allowed_sides: set[str],
) -> GripUdpRuntime:
    """Receive controller grip values from the explicitly allowed VR host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_address, port))
    sock.settimeout(0.2)
    receiver_stop = threading.Event()

    def receive() -> None:
        last_sequence: dict[str, tuple[str, int]] = {}
        announced_sides: set[str] = set()
        last_warning_at = 0.0
        while not receiver_stop.is_set():
            try:
                data, address = sock.recvfrom(513)
            except socket.timeout:
                continue
            except OSError:
                if receiver_stop.is_set():
                    break
                raise
            try:
                if address[0] != source_ip:
                    raise ValueError(f"unexpected source IP {address[0]}")
                side, value, sequence, session = parse_grip_udp_packet(data)
                if side not in allowed_sides:
                    continue
                previous = last_sequence.get(side)
                if (
                    previous is not None
                    and previous[0] == session
                    and sequence <= previous[1]
                ):
                    continue
                last_sequence[side] = (session, sequence)
                store.update_grip(side, value)
                if side not in announced_sides:
                    LOGGER.info("Received first valid %s grip UDP packet", side)
                    announced_sides.add(side)
            except ValueError as exc:
                now = time.monotonic()
                if now - last_warning_at >= 2.0:
                    LOGGER.warning("Ignoring grip UDP packet: %s", exc)
                    last_warning_at = now

    thread = threading.Thread(
        target=receive,
        name="revo2-grip-udp-receiver",
        daemon=True,
    )
    thread.start()
    LOGGER.info(
        "Listening for VR grip controls on udp://%s:%d from %s",
        bind_address,
        port,
        source_ip,
    )
    return GripUdpRuntime(sock=sock, stop_event=receiver_stop, thread=thread)


class BrokenPipeLogHandler(logging.Handler):
    """Turn the SDK's log-only BrokenPipe failure into a process failure."""

    def __init__(self, failure_event: threading.Event) -> None:
        super().__init__(level=logging.WARNING)
        self.failure_event = failure_event

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive logging path
            return
        if BROKEN_PIPE_LOG_MARKER in message:
            self.failure_event.set()


def raise_if_transport_failed(failure_event: threading.Event) -> None:
    if failure_event.is_set():
        raise RuntimeError(
            "BrainCo SDK transport failed with BrokenPipe; exiting for restart"
        )


@dataclass
class RosRuntime:
    rclpy: Any
    node: Any
    executor: Any
    thread: threading.Thread
    publishers: dict[tuple[str, str], Any]
    string_message_type: Any

    def publish(self, kind: str, side: str, payload: dict[str, Any]) -> None:
        message = self.string_message_type()
        message.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.publishers[(kind, side)].publish(message)

    def close(self) -> None:
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(timeout=2.0)
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


@dataclass
class ConnectedHand:
    spec: HandSpec
    port: str
    client: Any
    lock: asyncio.Lock


@dataclass
class PendingCommand:
    side: str
    source: str
    target: list[int]
    operation: Any
    raw_trigger: float | None = None
    normalized_trigger: float | None = None
    raw_grip: float | None = None
    normalized_grip: float | None = None


def _require_side(side: str) -> None:
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown hand side: {side!r}")


def _require_vector(name: str, values: Iterable[Any], length: int) -> list[Any]:
    result = list(values)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def _normalized(values: Iterable[int | float]) -> list[float]:
    return [float(value) / 1000.0 for value in values]


def decode_touch_status(item: Any) -> str:
    """Decode the stable status name exposed in the SDK description string."""
    description = str(getattr(item, "description", ""))
    match = re.search(r"status:\s*([A-Za-z0-9_]+)", description)
    return match.group(1) if match else "Unknown"


def state_payload(
    side: str,
    status: Any,
    timestamp: float,
    sdk_version: str,
) -> dict[str, Any]:
    """Serialize one six-actuator SDK motor-status sample."""
    _require_side(side)
    positions = [int(value) for value in _require_vector(
        "positions", status.positions, len(ACTUATOR_NAMES)
    )]
    speeds = [int(value) for value in _require_vector(
        "speeds", status.speeds, len(ACTUATOR_NAMES)
    )]
    currents = [int(value) for value in _require_vector(
        "currents", status.currents, len(ACTUATOR_NAMES)
    )]
    states = [enum_name(value) for value in _require_vector(
        "states", status.states, len(ACTUATOR_NAMES)
    )]
    return {
        "schema": STATE_SCHEMA,
        "side": side,
        "timestamp": float(timestamp),
        "sdk_version": str(sdk_version),
        "actuator_names": list(ACTUATOR_NAMES),
        "positions": positions,
        "speeds": speeds,
        "currents": currents,
        "states": states,
        "units": {
            "positions": "normalized_0_1000",
            "speeds": "normalized_-1000_1000",
            "currents": "normalized_-1000_1000",
            "states": "sdk_motor_state",
            "normalized_positions": "unitless_0_1",
            "normalized_speeds": "unitless_-1_1",
            "normalized_currents": "unitless_-1_1",
        },
        "normalized": {
            "positions": _normalized(positions),
            "speeds": _normalized(speeds),
            "currents": _normalized(currents),
        },
    }


def tactile_payload(
    side: str,
    items: Iterable[Any],
    timestamp: float,
    sdk_version: str,
) -> dict[str, Any]:
    """Serialize one five-finger capacitive tactile sample."""
    _require_side(side)
    tactile_items = _require_vector(
        "tactile items", items, len(TACTILE_FINGER_NAMES)
    )
    fingers = []
    for name, item in zip(TACTILE_FINGER_NAMES, tactile_items):
        description = str(getattr(item, "description", ""))
        fingers.append(
            {
                "name": name,
                "status": decode_touch_status(item),
                "description": description,
                "normal_force": [
                    int(item.normal_force1),
                    int(item.normal_force2),
                    int(item.normal_force3),
                ],
                "tangential_force": [
                    int(item.tangential_force1),
                    int(item.tangential_force2),
                    int(item.tangential_force3),
                ],
                "tangential_direction": [
                    int(item.tangential_direction1),
                    int(item.tangential_direction2),
                    int(item.tangential_direction3),
                ],
                "proximity": [
                    int(item.self_proximity1),
                    int(item.self_proximity2),
                    int(item.mutual_proximity),
                ],
            }
        )
    return {
        "schema": TACTILE_SCHEMA,
        "side": side,
        "timestamp": float(timestamp),
        "sdk_version": str(sdk_version),
        "finger_names": list(TACTILE_FINGER_NAMES),
        "fingers": fingers,
    }


def command_payload(
    side: str,
    target: Iterable[int],
    raw_trigger: float,
    normalized_trigger: float,
    duration_ms: int,
    timestamp: float,
    sdk_version: str,
    raw_grip: float | None = None,
    normalized_grip: float | None = None,
) -> dict[str, Any]:
    """Serialize a command only after the SDK reports successful submission."""
    _require_side(side)
    target_values = [int(value) for value in _require_vector(
        "target", target, len(ACTUATOR_NAMES)
    )]
    payload = {
        "schema": COMMAND_SCHEMA,
        "side": side,
        "timestamp": float(timestamp),
        "sdk_version": str(sdk_version),
        "actuator_names": list(ACTUATOR_NAMES),
        "positions": target_values,
        "position_unit": "normalized_0_1000",
        "raw_trigger": float(raw_trigger),
        "normalized_trigger": float(normalized_trigger),
        "trigger_unit": "unitless_0_1",
        "duration_ms": int(duration_ms),
    }
    if raw_grip is not None and normalized_grip is not None:
        payload["raw_grip"] = float(raw_grip)
        payload["normalized_grip"] = float(normalized_grip)
        payload["grip_unit"] = "unitless_0_1"
    return payload


def absolute_command_payload(
    side: str,
    target: Iterable[int],
    duration_ms: int,
    timestamp: float,
    sdk_version: str,
    status: str = "submitted",
    requested_target: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Serialize an accepted six-axis absolute command."""
    _require_side(side)
    target_values = [int(value) for value in _require_vector(
        "target", target, len(ACTUATOR_NAMES)
    )]
    payload = {
        "schema": ABSOLUTE_COMMAND_SCHEMA,
        "side": side,
        "timestamp": float(timestamp),
        "sdk_version": str(sdk_version),
        "actuator_names": list(ACTUATOR_NAMES),
        "positions": target_values,
        "position_unit": "normalized_0_1000",
        "duration_ms": int(duration_ms),
        "status": str(status),
    }
    if requested_target is not None:
        payload["requested_positions"] = [
            int(value)
            for value in _require_vector(
                "requested_target", requested_target, len(ACTUATOR_NAMES)
            )
        ]
    return payload


def all_tactile_fingers_enabled(enabled_bits: int) -> bool:
    return int(enabled_bits) & ALL_TACTILE_FINGERS_ENABLED == ALL_TACTILE_FINGERS_ENABLED


def normalize_trigger(value: float, low_deadzone: float, high_deadzone: float) -> float:
    """Clamp and rescale a raw 0..1 trigger with small endpoint deadzones."""
    value = min(1.0, max(0.0, value))
    if value <= low_deadzone:
        return 0.0
    if value >= high_deadzone:
        return 1.0
    return (value - low_deadzone) / (high_deadzone - low_deadzone)


def smoothstep_interval(value: float, start: float, end: float) -> float:
    """Return a clamped cubic smoothstep over ``start..end``."""
    if value <= start:
        return 0.0
    if value >= end:
        return 1.0
    progress = (value - start) / (end - start)
    return progress * progress * (3.0 - 2.0 * progress)


def interpolate_positions(
    start: tuple[int, ...],
    end: tuple[int, ...],
    progress: float,
) -> list[int]:
    return [
        round(start_value + progress * (end_value - start_value))
        for start_value, end_value in zip(start, end)
    ]


def target_for_trigger(value: float, side: str) -> list[int]:
    """Map the index trigger to thumb flex, index, and middle only."""
    if side not in GRASP_CLOSE_POSITION:
        raise ValueError(f"Unknown hand side: {side!r}")
    value = min(1.0, max(0.0, value))
    if value <= GRASP_PREPOSITION_END:
        progress = smoothstep_interval(value, 0.0, GRASP_PREPOSITION_END)
        return interpolate_positions(
            OPEN_POSITION,
            GRASP_PREPOSITION[side],
            progress,
        )

    progress = smoothstep_interval(value, GRASP_PREPOSITION_END, 1.0)
    return interpolate_positions(
        GRASP_PREPOSITION[side],
        GRASP_CLOSE_POSITION[side],
        progress,
    )


def target_for_controls(trigger_value: float, grip_value: float, side: str) -> list[int]:
    """Combine independent index-trigger and thumb-rotation grip controls."""
    if side not in THUMB_ROTATION_CLOSE_POSITION:
        raise ValueError(f"Unknown hand side: {side!r}")
    target = target_for_trigger(trigger_value, side)
    grip_value = min(1.0, max(0.0, grip_value))
    target[1] = round(
        smoothstep_interval(grip_value, 0.0, 1.0)
        * THUMB_ROTATION_CLOSE_POSITION[side]
    )
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously map XR VR triggers to left/right Revo2 hands."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Subscribe and print mapped targets without opening serial devices",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Verify selected Revo2 device identities without moving fingers",
    )
    parser.add_argument(
        "--hands",
        choices=("left", "right", "both"),
        default="both",
        help="Hands to control (default: both)",
    )
    parser.add_argument(
        "--input-source",
        choices=("trigger", "absolute", "both"),
        default="both",
        help="Accepted ROS command family (default: both, conflicts fail closed)",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Exit after this many seconds; 0 runs until stopped",
    )
    parser.add_argument(
        "--port-glob",
        default=DEFAULT_PORT_GLOB,
        help="Candidate Revo2 serial ports; devices are verified by Revo2 SN",
    )
    parser.add_argument("--left-port", help="Explicit left-hand port override")
    parser.add_argument("--right-port", help="Explicit right-hand port override")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--command-rate", type=float, default=10.0)
    parser.add_argument(
        "--feedback-rate",
        type=float,
        default=30.0,
        help="Motor and tactile feedback sampling rate (default: 30 Hz)",
    )
    parser.add_argument(
        "--feedback-timeout",
        type=float,
        default=0.25,
        help="Timeout for each feedback Modbus transaction (default: 0.25s)",
    )
    parser.add_argument(
        "--max-feedback-errors",
        type=int,
        default=5,
        help="Restart after this many consecutive feedback cycles fail per hand",
    )
    parser.add_argument("--min-trigger-delta", type=float, default=0.02)
    parser.add_argument(
        "--min-absolute-delta",
        type=int,
        default=1,
        help="Minimum normalized joint change before another absolute command",
    )
    parser.add_argument("--duration-ms", type=int, default=200)
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=1.0,
        help="Timeout for a Revo2 command transaction (default: 1s)",
    )
    parser.add_argument("--input-timeout", type=float, default=0.5)
    parser.add_argument(
        "--grip-udp-port",
        type=int,
        default=0,
        help="UDP port for independent VR grip values; 0 disables it",
    )
    parser.add_argument(
        "--grip-udp-bind",
        default="0.0.0.0",
        help="Local bind address for grip UDP packets",
    )
    parser.add_argument(
        "--grip-source-ip",
        default="",
        help="Only accept grip UDP packets from this source IP",
    )
    parser.add_argument("--low-deadzone", type=float, default=0.02)
    parser.add_argument("--high-deadzone", type=float, default=0.98)
    parser.add_argument(
        "--set-normalized-mode",
        action="store_true",
        help="Set a verified hand to normalized units when needed",
    )
    args = parser.parse_args(argv)

    if args.dry_run and args.probe_only:
        parser.error("--dry-run and --probe-only cannot be used together")
    if args.run_seconds < 0:
        parser.error("--run-seconds must be non-negative")
    if not 1.0 <= args.command_rate <= 10.0:
        parser.error("--command-rate must be in the range 1..10 Hz")
    if not 1.0 <= args.feedback_rate <= 100.0:
        parser.error("--feedback-rate must be in the range 1..100 Hz")
    if not 0.05 <= args.feedback_timeout <= 5.0:
        parser.error("--feedback-timeout must be in the range 0.05..5 seconds")
    if not 1 <= args.max_feedback_errors <= 100:
        parser.error("--max-feedback-errors must be in the range 1..100")
    if not 0.001 <= args.min_trigger_delta <= 0.25:
        parser.error("--min-trigger-delta must be in the range 0.001..0.25")
    if not 1 <= args.min_absolute_delta <= 100:
        parser.error("--min-absolute-delta must be in the range 1..100")
    if not 1 <= args.duration_ms <= 2_000:
        parser.error("--duration-ms must be in the range 1..2000")
    if not 0.05 <= args.command_timeout <= 5.0:
        parser.error("--command-timeout must be in the range 0.05..5 seconds")
    if not 0.1 <= args.input_timeout <= 10.0:
        parser.error("--input-timeout must be in the range 0.1..10 seconds")
    if args.grip_udp_port and not 1024 <= args.grip_udp_port <= 65535:
        parser.error("--grip-udp-port must be 0 or in the range 1024..65535")
    if args.grip_udp_port and not args.grip_source_ip:
        parser.error("--grip-source-ip is required when grip UDP is enabled")
    if not 0.0 <= args.low_deadzone < args.high_deadzone <= 1.0:
        parser.error("deadzone values must satisfy 0 <= low < high <= 1")
    return args


def selected_hand_specs(args: argparse.Namespace) -> tuple[HandSpec, ...]:
    if args.hands == "both":
        return HAND_SPECS
    return tuple(spec for spec in HAND_SPECS if spec.side == args.hands)


def start_ros_runtime(
    store: TriggerStore,
    specs: tuple[HandSpec, ...],
    input_source: str,
) -> RosRuntime:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Float64, Float64MultiArray, String

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("revo2_vr_trigger_bridge")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    feedback_qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    command_qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    publishers: dict[tuple[str, str], Any] = {}

    # Keep subscription objects owned by the node for its full lifetime.
    for spec in specs:
        if input_source in {"trigger", "both"}:
            node.create_subscription(
                Float64,
                spec.topic,
                lambda message, side=spec.side: store.update(
                    side, float(message.data)
                ),
                qos,
            )
        if input_source in {"absolute", "both"}:
            node.create_subscription(
                Float64MultiArray,
                ABSOLUTE_COMMAND_TOPICS[spec.side],
                lambda message, side=spec.side: store.update_absolute(
                    side, message.data
                ),
                command_qos,
            )
        for kind, template in PUBLISH_TOPIC_TEMPLATES.items():
            publishers[(kind, spec.side)] = node.create_publisher(
                String,
                template.format(side=spec.side),
                command_qos
                if kind in {"command", "absolute_command"}
                else feedback_qos,
            )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(
        target=executor.spin,
        name="revo2-ros-executor",
        daemon=True,
    )
    thread.start()
    LOGGER.info(
        "Subscribed to %s commands; "
        "advertising Revo2 feedback topics",
        input_source,
    )
    return RosRuntime(
        rclpy=rclpy,
        node=node,
        executor=executor,
        thread=thread,
        publishers=publishers,
        string_message_type=String,
    )


def candidate_ports(args: argparse.Namespace) -> list[str]:
    explicit = [port for port in (args.left_port, args.right_port) if port]
    ports = explicit + sorted(glob.glob(args.port_glob))
    unique: list[str] = []
    real_paths: set[str] = set()
    for port in ports:
        real_path = os.path.realpath(port)
        if real_path not in real_paths:
            unique.append(port)
            real_paths.add(real_path)
    return unique


def baudrate_enum(sdk: Any, baud: int) -> Any:
    if hasattr(sdk.Baudrate, "from_int"):
        return sdk.Baudrate.from_int(baud)
    names = {
        19200: "Baud19200",
        57600: "Baud57600",
        115200: "Baud115200",
        460800: "Baud460800",
        1_000_000: "Baud1Mbps",
        2_000_000: "Baud2Mbps",
        3_000_000: "Baud3Mbps",
        5_000_000: "Baud5Mbps",
    }
    try:
        return getattr(sdk.Baudrate, names[baud])
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unsupported baud rate: {baud}") from exc


def enum_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1].lower()


async def close_client(sdk: Any, client: Any) -> None:
    try:
        sdk.modbus_close(client)
    except Exception as exc:  # pragma: no cover - vendor cleanup path
        LOGGER.warning("Failed to close Revo2 client cleanly: %s", exc)


async def connect_hands(
    args: argparse.Namespace,
    sdk: Any,
    specs: tuple[HandSpec, ...],
) -> dict[str, ConnectedHand]:
    ports = candidate_ports(args)
    if not ports:
        raise FileNotFoundError(f"No candidate serial ports matched {args.port_glob!r}")

    LOGGER.info("Candidate serial ports: %s", ", ".join(ports))
    connected: dict[str, ConnectedHand] = {}
    used_real_paths: set[str] = set()

    for spec in specs:
        override = args.left_port if spec.side == "left" else args.right_port
        ports_for_hand: Iterable[str] = [override] if override else ports
        observed: list[str] = []

        for port in ports_for_hand:
            if not port or os.path.realpath(port) in used_real_paths:
                continue
            client = None
            try:
                client = await asyncio.wait_for(
                    sdk.modbus_open(port, baudrate_enum(sdk, args.baud)),
                    timeout=3.0,
                )
                info = await asyncio.wait_for(
                    client.get_device_info(spec.slave_id),
                    timeout=2.0,
                )
                serial_number = str(info.serial_number)
                hand_type = enum_name(info.hand_type)
                observed.append(f"{port}:SN={serial_number},type={hand_type}")
                if (
                    serial_number != spec.serial_number
                    or hand_type != spec.hand_type
                ):
                    await close_client(sdk, client)
                    continue

                unit_mode = await asyncio.wait_for(
                    client.get_finger_unit_mode(spec.slave_id),
                    timeout=2.0,
                )
                if unit_mode != sdk.FingerUnitMode.Normalized:
                    if not args.set_normalized_mode:
                        raise RuntimeError(
                            f"{spec.side} hand is not in Normalized unit mode; "
                            "restart with --set-normalized-mode after checking safety"
                        )
                    await asyncio.wait_for(
                        client.set_finger_unit_mode(
                            spec.slave_id,
                            sdk.FingerUnitMode.Normalized,
                        ),
                        timeout=2.0,
                    )

                touch_enabled = int(
                    await asyncio.wait_for(
                        client.get_touch_sensor_enabled(spec.slave_id),
                        timeout=2.0,
                    )
                )
                if not all_tactile_fingers_enabled(touch_enabled):
                    raise RuntimeError(
                        f"{spec.side} tactile sensors are not all enabled "
                        f"(mask=0x{touch_enabled:02x}, required=0x1f); "
                        "refusing to enable them implicitly"
                    )

                connected[spec.side] = ConnectedHand(
                    spec,
                    port,
                    client,
                    asyncio.Lock(),
                )
                used_real_paths.add(os.path.realpath(port))
                LOGGER.info(
                    "Verified %s Revo2: port=%s id=%d SN=%s touch=0x%02x",
                    spec.side,
                    port,
                    spec.slave_id,
                    serial_number,
                    touch_enabled,
                )
                break
            except Exception as exc:
                observed.append(f"{port}:{type(exc).__name__}")
                if client is not None:
                    await close_client(sdk, client)

        if spec.side not in connected:
            details = "; ".join(observed) if observed else "no usable ports"
            for hand in connected.values():
                await close_client(sdk, hand.client)
            raise RuntimeError(
                f"Could not verify {spec.side} Revo2 hand ({details})"
            )

    return connected


def should_send(previous: float | None, current: float, min_delta: float) -> bool:
    if previous is None:
        return True
    if current in (0.0, 1.0) and current != previous:
        return True
    return abs(current - previous) >= min_delta


def should_send_controls(
    previous_trigger: float | None,
    current_trigger: float,
    previous_grip: float | None,
    current_grip: float,
    min_delta: float,
) -> bool:
    return should_send(previous_trigger, current_trigger, min_delta) or should_send(
        previous_grip,
        current_grip,
        min_delta,
    )


def should_send_absolute(
    previous: tuple[int, ...] | None,
    current: tuple[int, ...],
    min_delta: int,
) -> bool:
    if previous is None:
        return True
    return max(abs(a - b) for a, b in zip(previous, current)) >= min_delta


def sdk_version_string(sdk: Any) -> str:
    try:
        return str(sdk.get_sdk_version())
    except Exception:  # pragma: no cover - vendor metadata path
        return "unknown"


async def send_hand_command(
    hand: ConnectedHand,
    target: list[int],
    duration_ms: int,
) -> Any:
    """Submit one command while exclusively owning that hand's Modbus client."""
    async with hand.lock:
        return await hand.client.set_finger_positions_and_durations(
            hand.spec.slave_id,
            target,
            [duration_ms] * len(ACTUATOR_NAMES),
        )


async def sample_hand_feedback(
    hand: ConnectedHand,
    feedback_timeout: float,
    sdk_version: str,
    ros_runtime: RosRuntime,
) -> None:
    """Read motor and tactile feedback serially on one hand's Modbus client."""
    async with hand.lock:
        motor_status = await asyncio.wait_for(
            hand.client.get_motor_status(hand.spec.slave_id),
            timeout=feedback_timeout,
        )
    ros_runtime.publish(
        "state",
        hand.spec.side,
        state_payload(hand.spec.side, motor_status, time.time(), sdk_version),
    )

    async with hand.lock:
        tactile_items = await asyncio.wait_for(
            hand.client.get_touch_sensor_status(hand.spec.slave_id),
            timeout=feedback_timeout,
        )
    ros_runtime.publish(
        "tactile",
        hand.spec.side,
        tactile_payload(hand.spec.side, tactile_items, time.time(), sdk_version),
    )


async def run_feedback_loop(
    args: argparse.Namespace,
    connected: dict[str, ConnectedHand],
    stop_event: threading.Event,
    transport_failure_event: threading.Event,
    sdk_version: str,
    ros_runtime: RosRuntime,
) -> None:
    """Publish both hands at a bounded rate and fail after repeated read errors."""
    consecutive_errors = {side: 0 for side in connected}
    interval = 1.0 / args.feedback_rate
    next_cycle = time.monotonic()

    while not stop_event.is_set():
        raise_if_transport_failed(transport_failure_event)
        hands = list(connected.items())
        results = await asyncio.gather(
            *(
                sample_hand_feedback(
                    hand,
                    args.feedback_timeout,
                    sdk_version,
                    ros_runtime,
                )
                for _, hand in hands
            ),
            return_exceptions=True,
        )
        raise_if_transport_failed(transport_failure_event)

        for (side, _), result in zip(hands, results):
            if isinstance(result, BaseException):
                consecutive_errors[side] += 1
                LOGGER.warning(
                    "%s feedback cycle failed (%d/%d): %s: %s",
                    side,
                    consecutive_errors[side],
                    args.max_feedback_errors,
                    type(result).__name__,
                    result,
                )
                if consecutive_errors[side] >= args.max_feedback_errors:
                    raise RuntimeError(
                        f"{side} Revo2 feedback failed "
                        f"{consecutive_errors[side]} consecutive times; "
                        "exiting for Docker restart"
                    ) from result
            else:
                if consecutive_errors[side]:
                    LOGGER.info("%s feedback stream recovered", side)
                consecutive_errors[side] = 0

        next_cycle += interval
        now = time.monotonic()
        if next_cycle < now - interval:
            next_cycle = now
        await asyncio.sleep(max(0.0, next_cycle - now))


async def run_mapping_loop(
    args: argparse.Namespace,
    store: TriggerStore,
    stop_event: threading.Event,
    transport_failure_event: threading.Event,
    sdk: Any | None,
    specs: tuple[HandSpec, ...],
    ros_runtime: RosRuntime | None,
) -> None:
    connected = {} if args.dry_run else await connect_hands(args, sdk, specs)
    if args.probe_only:
        LOGGER.info("Probe complete; no movement command was sent")
        for hand in connected.values():
            await close_client(sdk, hand.client)
        return

    sdk_version = "unavailable" if sdk is None else sdk_version_string(sdk)
    feedback_task = None
    if connected:
        if ros_runtime is None:
            for hand in connected.values():
                await close_client(sdk, hand.client)
            raise RuntimeError("ROS runtime is required for Revo2 feedback")
        feedback_task = asyncio.create_task(
            run_feedback_loop(
                args,
                connected,
                stop_event,
                transport_failure_event,
                sdk_version,
                ros_runtime,
            ),
            name="revo2-feedback",
        )

    last_trigger: dict[str, float | None] = {spec.side: None for spec in specs}
    last_grip: dict[str, float | None] = {spec.side: None for spec in specs}
    last_absolute: dict[str, tuple[int, ...] | None] = {
        spec.side: None for spec in specs
    }
    last_source: dict[str, str | None] = {spec.side: None for spec in specs}
    stale_trigger_logged: dict[str, bool] = {
        spec.side: False for spec in specs
    }
    stale_grip_logged: dict[str, bool] = {spec.side: False for spec in specs}
    stale_absolute_logged: dict[str, bool] = {
        spec.side: False for spec in specs
    }
    last_progress_log: dict[str, float] = {spec.side: 0.0 for spec in specs}
    started = time.monotonic()
    interval = 1.0 / args.command_rate

    try:
        while not stop_event.is_set():
            raise_if_transport_failed(transport_failure_event)
            if feedback_task is not None and feedback_task.done():
                feedback_error = feedback_task.exception()
                if feedback_error is not None:
                    raise feedback_error
                break
            loop_started = time.monotonic()
            if args.run_seconds and loop_started - started >= args.run_seconds:
                break

            samples, grip_samples, absolute_samples = store.snapshot_all()
            for spec in specs:
                trigger = samples.get(spec.side)
                grip = grip_samples.get(spec.side)
                absolute = absolute_samples.get(spec.side)
                trigger_fresh = (
                    trigger is not None
                    and loop_started - trigger.received_at <= args.input_timeout
                )
                grip_fresh = (
                    grip is not None
                    and loop_started - grip.received_at <= args.input_timeout
                )
                absolute_fresh = (
                    absolute is not None
                    and loop_started - absolute.received_at <= args.input_timeout
                )
                if (trigger_fresh or grip_fresh) and absolute_fresh:
                    raise RuntimeError(
                        f"conflicting fresh analog and absolute command for "
                        f"{spec.side}"
                    )

            pending: list[PendingCommand] = []
            for spec in specs:
                sample = samples.get(spec.side)
                grip_sample = grip_samples.get(spec.side)
                absolute = absolute_samples.get(spec.side)
                trigger_fresh = (
                    sample is not None
                    and loop_started - sample.received_at <= args.input_timeout
                )
                grip_fresh = (
                    grip_sample is not None
                    and loop_started - grip_sample.received_at <= args.input_timeout
                )
                absolute_fresh = (
                    absolute is not None
                    and loop_started - absolute.received_at <= args.input_timeout
                )
                if absolute_fresh:
                    if stale_absolute_logged[spec.side]:
                        LOGGER.info("%s absolute stream resumed", spec.side)
                        stale_absolute_logged[spec.side] = False
                    target_tuple = absolute.positions
                    if (
                        last_source[spec.side] == "absolute"
                        and not should_send_absolute(
                            last_absolute[spec.side],
                            target_tuple,
                            args.min_absolute_delta,
                        )
                    ):
                        if ros_runtime is not None:
                            applied = last_absolute[spec.side]
                            assert applied is not None
                            unchanged = target_tuple == applied
                            ros_runtime.publish(
                                "absolute_command",
                                spec.side,
                                absolute_command_payload(
                                    spec.side,
                                    applied,
                                    args.duration_ms,
                                    time.time(),
                                    sdk_version,
                                    status=(
                                        "unchanged" if unchanged else "suppressed"
                                    ),
                                    requested_target=(
                                        None if unchanged else target_tuple
                                    ),
                                ),
                            )
                        continue
                    target = list(target_tuple)
                    if args.dry_run:
                        LOGGER.info("DRY-RUN %s absolute target=%s", spec.side, target)
                        last_absolute[spec.side] = target_tuple
                        last_source[spec.side] = "absolute"
                        continue
                    pending.append(
                        PendingCommand(
                            side=spec.side,
                            source="absolute",
                            target=target,
                            operation=send_hand_command(
                                connected[spec.side], target, args.duration_ms
                            ),
                        )
                    )
                    continue

                if absolute is not None and not stale_absolute_logged[spec.side]:
                    LOGGER.warning(
                        "%s absolute command is stale; holding last hand position",
                        spec.side,
                    )
                    stale_absolute_logged[spec.side] = True

                if not trigger_fresh:
                    if sample is not None and not stale_trigger_logged[spec.side]:
                        LOGGER.warning(
                            "%s trigger is stale (%.3fs); holding its last axes",
                            spec.side,
                            loop_started - sample.received_at,
                        )
                        stale_trigger_logged[spec.side] = True
                elif stale_trigger_logged[spec.side]:
                    LOGGER.info("%s trigger stream resumed", spec.side)
                    stale_trigger_logged[spec.side] = False

                if not grip_fresh:
                    if (
                        grip_sample is not None
                        and not stale_grip_logged[spec.side]
                    ):
                        LOGGER.warning(
                            "%s grip is stale (%.3fs); holding thumb rotation",
                            spec.side,
                            loop_started - grip_sample.received_at,
                        )
                        stale_grip_logged[spec.side] = True
                elif stale_grip_logged[spec.side]:
                    LOGGER.info("%s grip stream resumed", spec.side)
                    stale_grip_logged[spec.side] = False

                if not trigger_fresh and not grip_fresh:
                    continue

                if sample is None:
                    normalized = last_trigger[spec.side] or 0.0
                    raw_trigger = normalized
                else:
                    raw_trigger = sample.value
                    normalized = normalize_trigger(
                        raw_trigger, args.low_deadzone, args.high_deadzone
                    )
                if grip_sample is None:
                    normalized_grip = last_grip[spec.side] or 0.0
                    raw_grip = normalized_grip
                else:
                    raw_grip = grip_sample.value
                    normalized_grip = normalize_trigger(
                        raw_grip, args.low_deadzone, args.high_deadzone
                    )
                if (
                    last_source[spec.side] == "trigger"
                    and not should_send_controls(
                        last_trigger[spec.side],
                        normalized,
                        last_grip[spec.side],
                        normalized_grip,
                        args.min_trigger_delta,
                    )
                ):
                    continue

                target = target_for_controls(
                    normalized,
                    normalized_grip,
                    spec.side,
                )
                if args.dry_run:
                    LOGGER.info(
                        "DRY-RUN %s trigger=%.4f grip=%.4f target=%s",
                        spec.side,
                        normalized,
                        normalized_grip,
                        target,
                    )
                    last_trigger[spec.side] = normalized
                    last_grip[spec.side] = normalized_grip
                    last_source[spec.side] = "trigger"
                    continue

                pending.append(
                    PendingCommand(
                        side=spec.side,
                        source="trigger",
                        target=target,
                        operation=send_hand_command(
                            connected[spec.side], target, args.duration_ms
                        ),
                        raw_trigger=raw_trigger,
                        normalized_trigger=normalized,
                        raw_grip=raw_grip,
                        normalized_grip=normalized_grip,
                    )
                )

            if pending:
                results = await asyncio.gather(
                    *(
                        asyncio.wait_for(
                            command.operation,
                            timeout=args.command_timeout,
                        )
                        for command in pending
                    ),
                    return_exceptions=True,
                )
                # The BrainCo SDK currently logs a broken transport and returns
                # normally. Convert that condition into an exception before a
                # failed command can be recorded as successfully sent.
                raise_if_transport_failed(transport_failure_event)
                for command, result in zip(pending, results):
                    if isinstance(result, BaseException):
                        raise RuntimeError(
                            f"{command.side} Revo2 command failed"
                        ) from result
                    last_source[command.side] = command.source
                    if command.source == "absolute":
                        last_absolute[command.side] = tuple(command.target)
                    else:
                        normalized = command.normalized_trigger
                        raw_trigger = command.raw_trigger
                        assert normalized is not None and raw_trigger is not None
                        last_trigger[command.side] = normalized
                        normalized_grip = command.normalized_grip
                        assert normalized_grip is not None
                        last_grip[command.side] = normalized_grip
                    if ros_runtime is not None:
                        if command.source == "absolute":
                            ros_runtime.publish(
                                "absolute_command",
                                command.side,
                                absolute_command_payload(
                                    command.side,
                                    command.target,
                                    args.duration_ms,
                                    time.time(),
                                    sdk_version,
                                    command.raw_grip,
                                    command.normalized_grip,
                                ),
                            )
                        else:
                            ros_runtime.publish(
                                "command",
                                command.side,
                                command_payload(
                                    command.side,
                                    command.target,
                                    raw_trigger,
                                    normalized,
                                    args.duration_ms,
                                    time.time(),
                                    sdk_version,
                                ),
                            )
                    now = time.monotonic()
                    if now - last_progress_log[command.side] >= 1.0:
                        if command.source == "absolute":
                            LOGGER.info(
                                "%s absolute target=%s",
                                command.side,
                                command.target,
                            )
                        else:
                            LOGGER.info(
                                "%s trigger=%.3f grip=%.3f target=%s",
                                command.side,
                                command.normalized_trigger,
                                command.normalized_grip,
                                command.target,
                            )
                        last_progress_log[command.side] = now

            elapsed = time.monotonic() - loop_started
            await asyncio.sleep(max(0.0, interval - elapsed))
    finally:
        if feedback_task is not None:
            if not feedback_task.done():
                feedback_task.cancel()
            await asyncio.gather(feedback_task, return_exceptions=True)
        for hand in connected.values():
            try:
                await close_client(sdk, hand.client)
            except Exception as exc:  # pragma: no cover - vendor cleanup path
                LOGGER.warning(
                    "Failed to close %s Revo2 client: %s",
                    hand.spec.side,
                    exc,
                )


def install_signal_handlers(stop_event: threading.Event) -> None:
    def request_stop(signum: int, _frame: Any) -> None:
        LOGGER.info("Received signal %d; stopping bridge", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stop_event = threading.Event()
    transport_failure_event = threading.Event()
    transport_failure_handler = BrokenPipeLogHandler(transport_failure_event)
    sdk_transport_logger = logging.getLogger(SDK_TRANSPORT_LOGGER_NAME)
    sdk_transport_logger.addHandler(transport_failure_handler)
    install_signal_handlers(stop_event)
    store = TriggerStore()
    ros_runtime = None
    grip_udp_runtime = None
    sdk = None
    specs = selected_hand_specs(args)

    missing_serials = [
        f"REVO2_{spec.side.upper()}_SERIAL"
        for spec in specs
        if not spec.serial_number
    ]
    if missing_serials:
        LOGGER.error(
            "Bridge configuration is missing required hand serial(s): %s",
            ", ".join(missing_serials),
        )
        return 2

    try:
        if not args.dry_run:
            from bc_stark_sdk import main_mod as sdk_module

            sdk = sdk_module
        if not args.probe_only:
            ros_runtime = start_ros_runtime(store, specs, args.input_source)
            if (
                args.grip_udp_port
                and args.input_source in {"trigger", "both"}
            ):
                grip_udp_runtime = start_grip_udp_runtime(
                    store,
                    args.grip_udp_bind,
                    args.grip_udp_port,
                    args.grip_source_ip,
                    {spec.side for spec in specs},
                )
        asyncio.run(
            run_mapping_loop(
                args,
                store,
                stop_event,
                transport_failure_event,
                sdk,
                specs,
                ros_runtime,
            )
        )
    except Exception as exc:
        LOGGER.error("Bridge failed: %s: %s", type(exc).__name__, exc)
        return 1
    finally:
        if grip_udp_runtime is not None:
            grip_udp_runtime.close()
        if ros_runtime is not None:
            ros_runtime.close()
        sdk_transport_logger.removeHandler(transport_failure_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
