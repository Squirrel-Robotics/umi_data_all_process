import tempfile
import unittest
from pathlib import Path

import server


class RecoverableDemoCollector(server.DemoCollector):
    def __init__(self, result=None, error=None):
        super().__init__()
        self.recovery_result = result or {
            "ok": True,
            "code": "recovered",
            "message": "ready",
        }
        self.recovery_error = error
        self.recovery_calls = 0

    def recover_e6_transport(self):
        self.recovery_calls += 1
        if self.recovery_error is not None:
            raise self.recovery_error
        return self.recovery_result


class E6RecoveryControllerTests(unittest.TestCase):
    def make_controller(self, collector):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        return server.CollectionController(
            collector=collector,
            cooldown_seconds=0,
            output_dir=Path(temporary_directory.name),
        )

    def test_recovery_runs_while_ready_and_is_reported_in_status(self):
        collector = RecoverableDemoCollector()
        controller = self.make_controller(collector)

        response = controller.recover_e6({})

        self.assertTrue(response["e6_recovery"]["ok"])
        self.assertEqual(collector.recovery_calls, 1)
        recovery_status = controller.status()["e6_recovery"]
        self.assertFalse(recovery_status["running"])
        self.assertEqual(
            recovery_status["last_result"]["code"],
            "recovered",
        )

    def test_recovery_is_rejected_during_episode_recording(self):
        collector = RecoverableDemoCollector()
        controller = self.make_controller(collector)
        controller.state = "recording"

        with self.assertRaises(server.ApiError) as raised:
            controller.recover_e6({})

        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(collector.recovery_calls, 0)

    def test_recovery_failure_does_not_put_collector_in_error_state(self):
        collector = RecoverableDemoCollector(error=RuntimeError("ADB failed"))
        controller = self.make_controller(collector)

        response = controller.recover_e6({})

        self.assertFalse(response["e6_recovery"]["ok"])
        self.assertEqual(response["e6_recovery"]["code"], "recovery_failed")
        self.assertEqual(controller.state, "ready")

    def test_start_is_blocked_while_recovery_is_running(self):
        collector = RecoverableDemoCollector()
        controller = self.make_controller(collector)
        controller._e6_recovery_status["running"] = True

        with self.assertRaises(server.ApiError) as raised:
            controller.start({"task": "test"})

        self.assertEqual(raised.exception.status, 409)


if __name__ == "__main__":
    unittest.main()
