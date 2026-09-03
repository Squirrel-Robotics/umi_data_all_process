#!/usr/bin/env python3
"""Forward XR controller grip values to the Revo2 bridge over UDP."""

from __future__ import annotations

import argparse
import json
import logging
import math
import secrets
import socket
from typing import Any


LOGGER = logging.getLogger("revo2_vr_grip_sender")
GRIP_UDP_SCHEMA = "revo2_vr_grip_v1"
TOPICS = {
    "left": "/joy_trigger/state/left",
    "right": "/joy_trigger/state/right",
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
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in the range 1024..65535")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import rclpy
    from protocol.msg import JoyTriggerState
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    rclpy.init()
    node = Node("revo2_vr_grip_sender")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((args.destination, args.port))
    sequence = {"left": 0, "right": 0}
    session = secrets.token_hex(8)
    packet_count = {"left": 0, "right": 0}

    def forward(message: Any, side: str) -> None:
        try:
            value = float(message.grip)
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

    LOGGER.info(
        "Forwarding %s grip topic(s) to udp://%s:%d",
        ",".join(selected),
        args.destination,
        args.port,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        LOGGER.info("Forwarded packets: %s", packet_count)
        sock.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
