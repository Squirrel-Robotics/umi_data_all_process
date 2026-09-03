#!/usr/bin/env python3
"""Forward XR controller grip values to the Revo2 bridge over UDP."""

from __future__ import annotations

import argparse
import json
import logging
import math
import secrets
import socket
import threading
import time
from typing import Any


LOGGER = logging.getLogger("revo2_vr_grip_sender")
GRIP_UDP_SCHEMA = "revo2_vr_grip_v1"
TOPICS = {
    "left": "/joy_trigger/state/left",
    "right": "/joy_trigger/state/right",
}
NO_PAUSE_TOPICS = {
    "left": "/revo2/joy_trigger_no_pause/left",
    "right": "/revo2/joy_trigger_no_pause/right",
}


def make_packet(side: str, value: float, sequence: int, session: str) -> bytes:
    if side not in TOPICS:
        raise ValueError(f"unknown side: {side!r}")
    if not math.isfinite(value):
        raise ValueError("grip value must be finite")
    value = min(1.0, max(0.0, value))
    return json.dumps(
        {
            "schema": GRIP_UDP_SCHEMA,
            "side": side,
            "grip": value,
            "seq": sequence,
            "session": session,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def side_grip_is_active(
    samples: dict[str, tuple[float, float]],
    now: float,
    threshold: float,
    max_age: float,
) -> bool:
    """Return whether either side grip is currently pressed and fresh."""
    return any(
        value > threshold and now - updated_at <= max_age
        for value, updated_at in samples.values()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward JoyTriggerState.grip to the Revo2 slave bridge"
    )
    parser.add_argument("--destination", required=True)
    parser.add_argument("--port", type=int, default=39157)
    parser.add_argument(
        "--hands",
        choices=("left", "right", "both"),
        default="both",
    )
    parser.add_argument(
        "--suppress-side-grip-pause",
        action="store_true",
        help="Ignore VR stop requests generated while either side grip is pressed",
    )
    parser.add_argument(
        "--publish-no-pause-inputs",
        action="store_true",
        help="Republish controller inputs with grip forced to zero for databridge",
    )
    parser.add_argument(
        "--control-service",
        default="/vr/meta_quest_control/trigger",
    )
    parser.add_argument(
        "--raw-control-service",
        default="/vr/meta_quest_control/trigger_raw",
    )
    parser.add_argument("--grip-active-threshold", type=float, default=0.05)
    parser.add_argument("--grip-sample-max-age", type=float, default=0.25)
    parser.add_argument("--pause-classify-delay", type=float, default=0.08)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in the range 1024..65535")
    if not 0.0 <= args.grip_active_threshold < 1.0:
        parser.error("--grip-active-threshold must be in the range 0..1")
    if not 0.05 <= args.grip_sample_max_age <= 2.0:
        parser.error("--grip-sample-max-age must be in the range 0.05..2")
    if not 0.0 <= args.pause_classify_delay <= 0.5:
        parser.error("--pause-classify-delay must be in the range 0..0.5")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import rclpy
    from protocol.msg import JoyTriggerState
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_srvs.srv import SetBool

    rclpy.init()
    node = Node("revo2_vr_grip_sender")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    no_pause_qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((args.destination, args.port))
    sequence = {"left": 0, "right": 0}
    session = secrets.token_hex(8)
    packet_count = {"left": 0, "right": 0}
    grip_samples: dict[str, tuple[float, float]] = {}
    grip_lock = threading.Lock()
    no_pause_publishers = {
        side: node.create_publisher(
            JoyTriggerState,
            NO_PAUSE_TOPICS[side],
            no_pause_qos,
        )
        for side in TOPICS
        if args.publish_no_pause_inputs
    }

    def publish_no_pause_input(message: Any, side: str) -> None:
        publisher = no_pause_publishers.get(side)
        if publisher is None:
            return
        sanitized = JoyTriggerState()
        sanitized.joystick_x = message.joystick_x
        sanitized.joystick_y = message.joystick_y
        sanitized.trigger = message.trigger
        sanitized.grip = 0.0
        sanitized.joy_button = message.joy_button
        sanitized.sw1 = message.sw1
        sanitized.sw2 = message.sw2
        publisher.publish(sanitized)

    def forward(message: Any, side: str) -> None:
        try:
            value = float(message.grip)
            with grip_lock:
                grip_samples[side] = (value, time.monotonic())
            publish_no_pause_input(message, side)
            packet = make_packet(side, value, sequence[side], session)
            sock.send(packet)
            sequence[side] += 1
            packet_count[side] += 1
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.warning("Failed to forward %s grip: %s", side, exc)

    selected = TOPICS if args.hands == "both" else {args.hands: TOPICS[args.hands]}
    for side, topic in selected.items():
        node.create_subscription(
            JoyTriggerState,
            topic,
            lambda message, selected_side=side: forward(message, selected_side),
            qos,
        )

    if args.suppress_side_grip_pause:
        callback_group = ReentrantCallbackGroup()
        raw_control_client = node.create_client(
            SetBool,
            args.raw_control_service,
            callback_group=callback_group,
        )

        def filter_control(request: Any, response: Any) -> Any:
            if not request.data and args.pause_classify_delay:
                # The request and grip topic originate from the same VR input
                # frame but can be scheduled a few milliseconds apart.
                time.sleep(args.pause_classify_delay)

            with grip_lock:
                suppress = not request.data and side_grip_is_active(
                    grip_samples,
                    time.monotonic(),
                    args.grip_active_threshold,
                    args.grip_sample_max_age,
                )

            if suppress:
                response.success = True
                response.message = "side-grip pause suppressed"
                LOGGER.info("Suppressed VR stop request while side grip is active")
                return response

            if not raw_control_client.wait_for_service(timeout_sec=1.0):
                response.success = False
                response.message = "raw VR control service unavailable"
                LOGGER.error("Raw VR control service is unavailable")
                return response

            future = raw_control_client.call_async(request)
            completed = threading.Event()
            future.add_done_callback(lambda _: completed.set())
            if not completed.wait(timeout=3.0):
                response.success = False
                response.message = "raw VR control service timed out"
                LOGGER.error("Raw VR control service timed out")
                return response

            try:
                raw_response = future.result()
                response.success = raw_response.success
                response.message = raw_response.message
            except Exception as exc:  # noqa: BLE001 - ROS surfaces transport errors
                response.success = False
                response.message = f"raw VR control service failed: {exc}"
                LOGGER.error("Raw VR control service failed: %s", exc)
            return response

        node.create_service(
            SetBool,
            args.control_service,
            filter_control,
            callback_group=callback_group,
        )
        LOGGER.info(
            "Suppressing side-grip pause on %s; forwarding other requests to %s",
            args.control_service,
            args.raw_control_service,
        )

    LOGGER.info(
        "Forwarding %s grip topic(s) to udp://%s:%d",
        ",".join(selected),
        args.destination,
        args.port,
    )
    if no_pause_publishers:
        LOGGER.info(
            "Publishing grip-zeroed controller inputs for databridge: %s",
            ",".join(NO_PAUSE_TOPICS[side] for side in no_pause_publishers),
        )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        LOGGER.info("Forwarded packets: %s", packet_count)
        executor.shutdown()
        sock.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
