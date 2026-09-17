"""Inference-only rotation; training videos already contain this transform."""
from dataclasses import dataclass
import numpy as np


def rotate180(frame):
    """Rotate RGB HWC/CHW exactly, with no resampling, swapping or color change."""
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected HWC/CHW RGB image, got {frame.shape}")
    if frame.shape[-1] == 3 and frame.shape[0] != 3:
        axes = (0, 1)
    elif frame.shape[0] == 3 and frame.shape[-1] != 3:
        axes = (1, 2)
    else:
        raise ValueError(f"Ambiguous or non-RGB image shape: {frame.shape}")
    return np.ascontiguousarray(np.flip(frame, axis=axes))


@dataclass(frozen=True)
class RotateRawWristImages:
    """Apply once before CX002Inputs to raw, unrotated camera observations."""

    def __call__(self, data):
        if data.get("_wrist180_applied", False):
            raise ValueError("Wrist rotation was already applied in this transform chain")
        result = dict(data)
        images = dict(data["images"])
        for key in ("cam_left_wrist", "cam_right_wrist"):
            images[key] = rotate180(images[key])
        result["images"] = images
        result["_wrist180_applied"] = True
        return result


def self_test():
    cases = 0
    for dtype in (np.uint8, np.float32):
        for chw in (False, True):
            left = np.arange(5 * 7 * 3).reshape(5, 7, 3).astype(dtype)
            right = (left + 43).astype(dtype)
            head = left.copy()
            if chw:
                left, right, head = (a.transpose(2, 0, 1) for a in (left, right, head))
            before_l, before_r = left.copy(), right.copy()
            state, actions = np.arange(30), np.arange(50 * 30).reshape(50, 30)
            data = {"images": {"cam_high": head, "cam_left_wrist": left,
                               "cam_right_wrist": right}, "state": state,
                    "actions": actions, "prompt": "Put the two objects into the box."}
            result = RotateRawWristImages()(data)
            assert "_wrist180_applied" not in data
            assert result["state"] is state and result["actions"] is actions
            assert result["images"]["cam_high"] is head
            for key, before in (("cam_left_wrist", before_l), ("cam_right_wrist", before_r)):
                actual = result["images"][key]
                expected = before[:, ::-1, ::-1] if chw else before[::-1, ::-1, :]
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(data["images"][key], before)
                np.testing.assert_array_equal(rotate180(actual), before)
                assert actual.flags.c_contiguous and actual.dtype == dtype
            try:
                RotateRawWristImages()(result)
            except ValueError:
                pass
            else:
                raise AssertionError("Duplicate transform not rejected")
            cases += 1
    return {"status": "passed", "layout_dtype_cases": cases, "head_state_action_unchanged": True,
            "caller_unmodified": True, "camera_swap": False, "double_application_rejected": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test()))
