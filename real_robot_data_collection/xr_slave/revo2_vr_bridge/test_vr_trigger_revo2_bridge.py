import argparse
import asyncio
import contextlib
import io
import json
import logging
import threading
import unittest
from unittest import mock
from types import SimpleNamespace

from vr_trigger_revo2_bridge import (
    ACTUATOR_NAMES,
    ABSOLUTE_COMMAND_SCHEMA,
    BROKEN_PIPE_LOG_MARKER,
    BrokenPipeLogHandler,
    COMMAND_SCHEMA,
    ConnectedHand,
    DEFAULT_PORT_GLOB,
    GRIP_UDP_SCHEMA,
    GRASP_CLOSE_POSITION,
    GRASP_PREPOSITION,
    HAND_SPECS,
    OPEN_POSITION,
    STATE_SCHEMA,
    TACTILE_FINGER_NAMES,
    TACTILE_SCHEMA,
    THUMB_ROTATION_CLOSE_POSITION,
    TriggerStore,
    all_tactile_fingers_enabled,
    absolute_command_payload,
    command_payload,
    connect_hands,
    decode_touch_status,
    normalize_trigger,
    parse_grip_udp_packet,
    parse_args,
    raise_if_transport_failed,
    run_feedback_loop,
    run_mapping_loop,
    sample_hand_feedback,
    send_hand_command,
    should_send,
    should_send_controls,
    should_send_absolute,
    smoothstep_interval,
    state_payload,
    tactile_payload,
    target_for_trigger,
    target_for_controls,
)
from vr_grip_udp_sender import make_packet


class TriggerMappingTests(unittest.TestCase):
    def test_endpoint_deadzones(self) -> None:
        self.assertEqual(normalize_trigger(-1.0, 0.02, 0.98), 0.0)
        self.assertEqual(normalize_trigger(0.02, 0.02, 0.98), 0.0)
        self.assertEqual(normalize_trigger(0.98, 0.02, 0.98), 1.0)
        self.assertEqual(normalize_trigger(2.0, 0.02, 0.98), 1.0)

    def test_midpoint_follows_power_grasp_closing_stage(self) -> None:
        self.assertAlmostEqual(normalize_trigger(0.5, 0.02, 0.98), 0.5)
        target = target_for_trigger(0.5, "left")
        self.assertEqual(target, [407, 0, 44, 44, 0, 0])

    def test_thumb_prepositions_before_index_and_middle_close(self) -> None:
        target = target_for_trigger(0.4, "left")
        self.assertEqual(target, list(GRASP_PREPOSITION["left"]))
        self.assertEqual(target[2:], [0, 0, 0, 0])

    def test_released_trigger_is_exact_all_zero_target(self) -> None:
        self.assertEqual(OPEN_POSITION, (0, 0, 0, 0, 0, 0))
        for side in ("left", "right"):
            self.assertEqual(target_for_trigger(0.0, side), [0, 0, 0, 0, 0, 0])

    def test_index_and_middle_close_while_ring_and_pinky_stay_zero(self) -> None:
        target = target_for_trigger(0.8, "left")
        self.assertGreater(target[2], 0)
        self.assertEqual(target[3], target[2])
        self.assertEqual(target[4:], [0, 0])

    def test_smoothstep_interval_endpoints(self) -> None:
        self.assertEqual(smoothstep_interval(0.0, 0.1, 0.9), 0.0)
        self.assertAlmostEqual(smoothstep_interval(0.5, 0.1, 0.9), 0.5)
        self.assertEqual(smoothstep_interval(1.0, 0.1, 0.9), 1.0)

    def test_exact_open_and_close_targets(self) -> None:
        for side in ("left", "right"):
            self.assertEqual(target_for_trigger(0.0, side), list(OPEN_POSITION))
            self.assertEqual(
                target_for_trigger(1.0, side),
                list(GRASP_CLOSE_POSITION[side]),
            )

    def test_left_and_right_calibrated_thumb_targets_differ(self) -> None:
        self.assertNotEqual(
            target_for_trigger(1.0, "left"),
            target_for_trigger(1.0, "right"),
        )

    def test_closed_pose_uses_only_index_and_middle_fingers(self) -> None:
        for side in ("left", "right"):
            target = target_for_trigger(1.0, side)
            self.assertEqual(target[2:], [593, 593, 0, 0])

    def test_grip_independently_controls_actual_thumb_rotation_axis(self) -> None:
        for side in ("left", "right"):
            grip_only = target_for_controls(0.0, 1.0, side)
            grasp_only = target_for_controls(1.0, 0.0, side)
            combined = target_for_controls(1.0, 1.0, side)

            self.assertEqual(
                grip_only,
                [0, THUMB_ROTATION_CLOSE_POSITION[side], 0, 0, 0, 0],
            )
            self.assertEqual(grasp_only[1], 0)
            self.assertEqual(grasp_only[2:], [593, 593, 0, 0])
            self.assertEqual(combined[0], grasp_only[0])
            self.assertEqual(combined[1], THUMB_ROTATION_CLOSE_POSITION[side])
            self.assertEqual(combined[2:], [593, 593, 0, 0])
            self.assertEqual(combined[5], 0)

    def test_rate_delta_logic_keeps_endpoints(self) -> None:
        self.assertTrue(should_send(None, 0.0, 0.01))
        self.assertFalse(should_send(0.50, 0.505, 0.01))
        self.assertTrue(should_send(0.50, 0.52, 0.01))
        self.assertTrue(should_send(0.995, 1.0, 0.01))

    def test_either_analog_control_can_trigger_a_command(self) -> None:
        self.assertFalse(should_send_controls(0.5, 0.505, 0.4, 0.405, 0.01))
        self.assertTrue(should_send_controls(0.5, 0.52, 0.4, 0.405, 0.01))
        self.assertTrue(should_send_controls(0.5, 0.505, 0.4, 0.42, 0.01))

    def test_absolute_rate_delta_uses_largest_joint_change(self) -> None:
        previous = (10, 20, 30, 40, 50, 60)
        self.assertTrue(should_send_absolute(None, previous, 2))
        self.assertFalse(
            should_send_absolute(previous, (11, 20, 30, 40, 50, 60), 2)
        )
        self.assertTrue(
            should_send_absolute(previous, (10, 20, 32, 40, 50, 60), 2)
        )


class AbsoluteInputTests(unittest.TestCase):
    def test_store_accepts_six_integer_normalized_positions(self) -> None:
        store = TriggerStore()
        store.update_absolute("left", [0.0, 100, 200, 300, 400, 1000])

        sample = store.snapshot_absolute()["left"]
        self.assertEqual(sample.positions, (0, 100, 200, 300, 400, 1000))

    def test_store_rejects_bad_absolute_positions(self) -> None:
        invalid = (
            [0, 1, 2],
            [0, 1, 2, 3, 4, 1001],
            [0, 1, 2, 3, 4, 5.5],
            [0, 1, 2, 3, 4, float("nan")],
            [0, 1, 2, 3, 4, float("inf")],
        )
        for values in invalid:
            with self.subTest(values=values):
                store = TriggerStore()
                with self.assertLogs("revo2_vr_bridge", level="WARNING"):
                    store.update_absolute("right", values)
                self.assertEqual(store.snapshot_absolute(), {})


class GripUdpTests(unittest.TestCase):
    def test_versioned_packet_is_accepted(self) -> None:
        packet = make_packet("right", 0.625, 42, "0123456789abcdef")
        self.assertEqual(
            parse_grip_udp_packet(packet),
            ("right", 0.625, 42, "0123456789abcdef"),
        )

    def test_invalid_packets_are_rejected(self) -> None:
        invalid = (
            b"not-json",
            b'{"schema":"wrong","side":"left","grip":0.5,"seq":1,"session":"0123456789abcdef"}',
            json.dumps(
                {
                    "schema": GRIP_UDP_SCHEMA,
                    "side": "middle",
                    "grip": 0.5,
                    "seq": 1,
                    "session": "0123456789abcdef",
                }
            ).encode(),
            json.dumps(
                {
                    "schema": GRIP_UDP_SCHEMA,
                    "side": "left",
                    "grip": 1.1,
                    "seq": 1,
                    "session": "0123456789abcdef",
                }
            ).encode(),
        )
        for packet in invalid:
            with self.subTest(packet=packet):
                with self.assertRaises(ValueError):
                    parse_grip_udp_packet(packet)

    def test_store_tracks_grip_separately_from_index_trigger(self) -> None:
        store = TriggerStore()
        store.update("left", 0.2)
        store.update_grip("left", 0.8)
        self.assertEqual(store.snapshot()["left"].value, 0.2)
        self.assertEqual(store.snapshot_grip()["left"].value, 0.8)


class BrokenPipeRecoveryTests(unittest.TestCase):
    @staticmethod
    def make_record(message: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="bc_stark_sdk.modbus.device_ctx",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )

    def test_sdk_broken_pipe_log_sets_failure_event(self) -> None:
        failure_event = threading.Event()
        handler = BrokenPipeLogHandler(failure_event)

        handler.handle(
            self.make_record(
                "write_multiple_registers failed: " + BROKEN_PIPE_LOG_MARKER
            )
        )

        self.assertTrue(failure_event.is_set())

    def test_unrelated_sdk_warning_does_not_set_failure_event(self) -> None:
        failure_event = threading.Event()
        handler = BrokenPipeLogHandler(failure_event)

        handler.handle(self.make_record("temporary response timeout"))

        self.assertFalse(failure_event.is_set())

    def test_failure_guard_raises_for_docker_restart(self) -> None:
        failure_event = threading.Event()
        failure_event.set()

        with self.assertRaisesRegex(RuntimeError, "exiting for restart"):
            raise_if_transport_failed(failure_event)


class PayloadSerializationTests(unittest.TestCase):
    @staticmethod
    def motor_status() -> SimpleNamespace:
        return SimpleNamespace(
            positions=[0, 100, 200, 300, 400, 1000],
            speeds=[-1000, -500, 0, 250, 500, 1000],
            currents=[-900, -100, 0, 100, 700, 900],
            states=[
                "MotorState.Idle",
                "MotorState.Running",
                "MotorState.Stalled",
                "MotorState.Idle",
                "MotorState.Idle",
                "MotorState.Idle",
            ],
        )

    @staticmethod
    def touch_item(index: int, status: str = "Normal") -> SimpleNamespace:
        base = index * 100
        return SimpleNamespace(
            description=f"finger {index}, status: {status}, ready",
            normal_force1=base + 1,
            normal_force2=base + 2,
            normal_force3=base + 3,
            tangential_force1=base + 4,
            tangential_force2=base + 5,
            tangential_force3=base + 6,
            tangential_direction1=base + 7,
            tangential_direction2=base + 8,
            tangential_direction3=base + 9,
            self_proximity1=base + 10,
            self_proximity2=base + 11,
            mutual_proximity=base + 12,
        )

    def test_state_payload_has_flat_sdk_values_and_normalized_derivatives(self) -> None:
        payload = state_payload("left", self.motor_status(), 123.5, "2.0.2")

        self.assertEqual(payload["schema"], STATE_SCHEMA)
        self.assertEqual(payload["schema"], "revo2_joint_state_v1")
        self.assertEqual(payload["side"], "left")
        self.assertEqual(payload["timestamp"], 123.5)
        self.assertEqual(payload["sdk_version"], "2.0.2")
        self.assertEqual(payload["actuator_names"], list(ACTUATOR_NAMES))
        self.assertEqual(payload["positions"], [0, 100, 200, 300, 400, 1000])
        self.assertEqual(payload["speeds"], [-1000, -500, 0, 250, 500, 1000])
        self.assertEqual(payload["currents"], [-900, -100, 0, 100, 700, 900])
        self.assertEqual(payload["states"][0], "idle")
        self.assertEqual(payload["units"]["positions"], "normalized_0_1000")
        self.assertEqual(
            payload["normalized"]["positions"],
            [0.0, 0.1, 0.2, 0.3, 0.4, 1.0],
        )
        self.assertEqual(payload["normalized"]["speeds"][0], -1.0)
        self.assertEqual(payload["normalized"]["currents"][-1], 0.9)

    def test_state_payload_rejects_non_six_axis_data(self) -> None:
        status = self.motor_status()
        status.positions = [1, 2]
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            state_payload("left", status, 1.0, "2.0.2")

    def test_tactile_payload_has_five_named_fingers_and_triplets(self) -> None:
        items = [self.touch_item(index) for index in range(5)]
        payload = tactile_payload("right", items, 456.25, "2.0.2")

        self.assertEqual(payload["schema"], TACTILE_SCHEMA)
        self.assertEqual(payload["finger_names"], list(TACTILE_FINGER_NAMES))
        self.assertEqual(len(payload["fingers"]), 5)
        index = payload["fingers"][1]
        self.assertEqual(index["name"], "index")
        self.assertEqual(index["status"], "Normal")
        self.assertEqual(index["normal_force"], [101, 102, 103])
        self.assertEqual(index["tangential_force"], [104, 105, 106])
        self.assertEqual(index["tangential_direction"], [107, 108, 109])
        self.assertEqual(index["proximity"], [110, 111, 112])

    def test_tactile_status_is_parsed_from_description(self) -> None:
        item = self.touch_item(0, "CommunicationError")
        item.status = 987654
        self.assertEqual(decode_touch_status(item), "CommunicationError")
        item.description = "no decoded status in this description"
        self.assertEqual(decode_touch_status(item), "Unknown")

    def test_tactile_payload_rejects_non_five_finger_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            tactile_payload("left", [self.touch_item(0)], 1.0, "2.0.2")

    def test_command_payload_contains_successful_six_axis_target(self) -> None:
        payload = command_payload(
            "left",
            [400, 337, 10, 20, 30, 40],
            raw_trigger=0.52,
            normalized_trigger=0.5,
            duration_ms=200,
            timestamp=789.0,
            sdk_version="2.0.2",
        )

        self.assertEqual(payload["schema"], COMMAND_SCHEMA)
        self.assertEqual(payload["schema"], "revo2_joint_command_v1")
        self.assertEqual(payload["positions"], [400, 337, 10, 20, 30, 40])
        self.assertEqual(payload["position_unit"], "normalized_0_1000")
        self.assertNotIn("target", payload)
        self.assertEqual(payload["raw_trigger"], 0.52)
        self.assertEqual(payload["normalized_trigger"], 0.5)
        self.assertEqual(payload["duration_ms"], 200)

    def test_absolute_command_payload_is_distinct_and_six_axis(self) -> None:
        payload = absolute_command_payload(
            "right",
            [10, 20, 30, 40, 50, 60],
            duration_ms=200,
            timestamp=790.0,
            sdk_version="2.0.2",
        )

        self.assertEqual(payload["schema"], ABSOLUTE_COMMAND_SCHEMA)
        self.assertEqual(payload["positions"], [10, 20, 30, 40, 50, 60])
        self.assertEqual(payload["position_unit"], "normalized_0_1000")
        self.assertEqual(payload["status"], "submitted")
        self.assertNotIn("raw_trigger", payload)

    def test_suppressed_absolute_ack_distinguishes_requested_and_applied(self) -> None:
        payload = absolute_command_payload(
            "left",
            [10, 20, 30, 40, 50, 60],
            duration_ms=200,
            timestamp=791.0,
            sdk_version="2.0.2",
            status="suppressed",
            requested_target=[11, 20, 30, 40, 50, 60],
        )

        self.assertEqual(payload["status"], "suppressed")
        self.assertEqual(payload["positions"], [10, 20, 30, 40, 50, 60])
        self.assertEqual(
            payload["requested_positions"], [11, 20, 30, 40, 50, 60]
        )

    def test_tactile_enabled_mask_requires_all_low_five_bits(self) -> None:
        self.assertTrue(all_tactile_fingers_enabled(0x1F))
        self.assertTrue(all_tactile_fingers_enabled(0x3F))
        self.assertFalse(all_tactile_fingers_enabled(0x0F))
        self.assertFalse(all_tactile_fingers_enabled(0x00))


class ArgumentValidationTests(unittest.TestCase):
    def test_default_port_glob_covers_both_robot_usb_controllers(self) -> None:
        self.assertIn("c[57]:00.3", DEFAULT_PORT_GLOB)

    def test_existing_container_command_uses_feedback_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.command_rate, 10.0)
        self.assertEqual(args.feedback_rate, 30.0)
        self.assertEqual(args.feedback_timeout, 0.25)
        self.assertEqual(args.max_feedback_errors, 5)
        self.assertEqual(args.input_source, "both")
        self.assertEqual(args.min_absolute_delta, 1)
        self.assertEqual(args.command_timeout, 1.0)

    def test_explicit_feedback_arguments_are_accepted(self) -> None:
        args = parse_args(
            [
                "--feedback-rate",
                "60",
                "--feedback-timeout",
                "0.5",
                "--max-feedback-errors",
                "9",
            ]
        )
        self.assertEqual(args.feedback_rate, 60.0)
        self.assertEqual(args.feedback_timeout, 0.5)
        self.assertEqual(args.max_feedback_errors, 9)

    def test_invalid_feedback_arguments_are_rejected(self) -> None:
        invalid_arguments = (
            ["--command-rate", "10.1"],
            ["--feedback-rate", "0"],
            ["--feedback-rate", "101"],
            ["--feedback-timeout", "0.01"],
            ["--max-feedback-errors", "0"],
            ["--max-feedback-errors", "101"],
            ["--min-absolute-delta", "-1"],
            ["--min-absolute-delta", "0"],
            ["--min-absolute-delta", "101"],
            ["--duration-ms", "2001"],
            ["--command-timeout", "0.01"],
            ["--command-timeout", "6"],
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(arguments)


class FakeRosRuntime:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, dict]] = []

    def publish(self, kind: str, side: str, payload: dict) -> None:
        self.published.append((kind, side, payload))


class SerializingClient:
    def __init__(self, fail_feedback: bool = False) -> None:
        self.active_calls = 0
        self.overlap_detected = False
        self.fail_feedback = fail_feedback

    async def _enter(self) -> None:
        self.active_calls += 1
        if self.active_calls != 1:
            self.overlap_detected = True
        await asyncio.sleep(0)

    def _leave(self) -> None:
        self.active_calls -= 1

    async def get_motor_status(self, _slave_id: int) -> SimpleNamespace:
        await self._enter()
        try:
            if self.fail_feedback:
                raise TimeoutError("motor timeout")
            await asyncio.sleep(0.001)
            return PayloadSerializationTests.motor_status()
        finally:
            self._leave()

    async def get_touch_sensor_status(self, _slave_id: int) -> list[SimpleNamespace]:
        await self._enter()
        try:
            await asyncio.sleep(0.001)
            return [PayloadSerializationTests.touch_item(index) for index in range(5)]
        finally:
            self._leave()

    async def set_finger_positions_and_durations(
        self,
        _slave_id: int,
        _target: list[int],
        _durations: list[int],
    ) -> None:
        await self._enter()
        try:
            await asyncio.sleep(0.001)
        finally:
            self._leave()


class SerialOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_trigger_and_absolute_inputs_fail_closed(self) -> None:
        store = TriggerStore()
        store.update("left", 0.5)
        store.update_absolute("left", [10, 20, 30, 40, 50, 60])
        args = parse_args(["--dry-run", "--hands", "left"])

        with self.assertRaisesRegex(RuntimeError, "conflicting fresh"):
            await run_mapping_loop(
                args,
                store,
                threading.Event(),
                threading.Event(),
                None,
                (HAND_SPECS[0],),
                None,
            )

    async def test_connect_fails_when_any_tactile_finger_is_disabled(self) -> None:
        client = SimpleNamespace()

        async def get_device_info(_slave_id: int) -> SimpleNamespace:
            return SimpleNamespace(
                serial_number=HAND_SPECS[0].serial_number,
                hand_type="HandType.Left",
            )

        normalized_mode = object()

        async def get_finger_unit_mode(_slave_id: int) -> object:
            return normalized_mode

        async def get_touch_sensor_enabled(_slave_id: int) -> int:
            return 0x0F

        client.get_device_info = get_device_info
        client.get_finger_unit_mode = get_finger_unit_mode
        client.get_touch_sensor_enabled = get_touch_sensor_enabled

        class FakeSdk:
            class FingerUnitMode:
                Normalized = normalized_mode

            class Baudrate:
                @staticmethod
                def from_int(value: int) -> int:
                    return value

            closed = []

            @staticmethod
            async def modbus_open(_port: str, _baud: int) -> SimpleNamespace:
                return client

            @classmethod
            def modbus_close(cls, value: SimpleNamespace) -> None:
                cls.closed.append(value)

        args = parse_args(["--hands", "left", "--left-port", "/dev/fake"])
        with self.assertRaisesRegex(RuntimeError, "Could not verify left"):
            await connect_hands(args, FakeSdk, (HAND_SPECS[0],))
        self.assertEqual(FakeSdk.closed, [client])

    async def test_unit_mode_mismatch_closes_native_client_once(self) -> None:
        client = SimpleNamespace()

        async def get_device_info(_slave_id: int) -> SimpleNamespace:
            return SimpleNamespace(
                serial_number=HAND_SPECS[0].serial_number,
                hand_type="HandType.Left",
            )

        async def get_finger_unit_mode(_slave_id: int) -> object:
            return object()

        client.get_device_info = get_device_info
        client.get_finger_unit_mode = get_finger_unit_mode

        class FakeSdk:
            class FingerUnitMode:
                Normalized = object()

            class Baudrate:
                @staticmethod
                def from_int(value: int) -> int:
                    return value

            closed = []

            @staticmethod
            async def modbus_open(_port: str, _baud: int) -> SimpleNamespace:
                return client

            @classmethod
            def modbus_close(cls, value: SimpleNamespace) -> None:
                cls.closed.append(value)

        args = parse_args(["--hands", "left", "--left-port", "/dev/fake"])
        with self.assertRaisesRegex(RuntimeError, "Could not verify left"):
            await connect_hands(args, FakeSdk, (HAND_SPECS[0],))
        self.assertEqual(FakeSdk.closed, [client])

    async def test_feedback_and_command_share_one_per_hand_lock(self) -> None:
        client = SerializingClient()
        hand = ConnectedHand(HAND_SPECS[0], "/dev/fake", client, asyncio.Lock())
        ros_runtime = FakeRosRuntime()

        await asyncio.gather(
            sample_hand_feedback(hand, 0.1, "2.0.2", ros_runtime),
            send_hand_command(hand, [0, 0, 0, 0, 0, 0], 200),
        )

        self.assertFalse(client.overlap_detected)
        self.assertEqual(
            [(kind, side) for kind, side, _ in ros_runtime.published],
            [("state", "left"), ("tactile", "left")],
        )

    async def test_feedback_loop_fails_after_bounded_consecutive_errors(self) -> None:
        client = SerializingClient(fail_feedback=True)
        hand = ConnectedHand(HAND_SPECS[0], "/dev/fake", client, asyncio.Lock())
        args = argparse.Namespace(
            feedback_rate=100.0,
            feedback_timeout=0.1,
            max_feedback_errors=2,
        )

        with self.assertLogs("revo2_vr_bridge", level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "2 consecutive times"):
                await run_feedback_loop(
                    args,
                    {"left": hand},
                    threading.Event(),
                    threading.Event(),
                    "2.0.2",
                    FakeRosRuntime(),
                )

    async def test_mapping_loop_closes_client_after_feedback_failure(self) -> None:
        client = SerializingClient(fail_feedback=True)
        hand = ConnectedHand(HAND_SPECS[0], "/dev/fake", client, asyncio.Lock())
        args = parse_args(
            [
                "--hands",
                "left",
                "--feedback-rate",
                "100",
                "--max-feedback-errors",
                "1",
            ]
        )
        sdk = SimpleNamespace(get_sdk_version=lambda: "test")

        with mock.patch(
            "vr_trigger_revo2_bridge.connect_hands",
            new=mock.AsyncMock(return_value={"left": hand}),
        ), mock.patch(
            "vr_trigger_revo2_bridge.close_client",
            new=mock.AsyncMock(),
        ) as close:
            with self.assertLogs("revo2_vr_bridge", level="WARNING"):
                with self.assertRaisesRegex(RuntimeError, "feedback failed"):
                    await run_mapping_loop(
                        args,
                        TriggerStore(),
                        threading.Event(),
                        threading.Event(),
                        sdk,
                        (HAND_SPECS[0],),
                        FakeRosRuntime(),
                    )

        close.assert_awaited_once_with(sdk, client)


if __name__ == "__main__":
    unittest.main()
