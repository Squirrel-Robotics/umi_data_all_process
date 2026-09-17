"""CPU check: the raw inference adapter agrees with the encoded training images."""
import copy
import json
import numpy as np
from openpi import transforms
from openpi.training import config, data_loader
from serve_wrist180 import CONFIG
from wrist180_inference import RotateRawWristImages

cfg = config.get_config(CONFIG)
baseline = config.get_config(cfg.policy_metadata["rotation_baseline_config"])
current_data = cfg.data.create(cfg.assets_dirs, cfg.model)
previous_data = baseline.data.create(baseline.assets_dirs, baseline.model)
current_raw = data_loader.create_torch_dataset(current_data, 50, cfg.model)
previous_raw = data_loader.create_torch_dataset(previous_data, 50, baseline.model)
training = data_loader.transform_dataset(current_raw, current_data)
inference = transforms.compose([
    RotateRawWristImages(), transforms.InjectDefaultPrompt(cfg.data.default_prompt),
    *current_data.data_transforms.inputs,
    transforms.Normalize(current_data.norm_stats, use_quantiles=current_data.use_quantile_norm),
    *current_data.model_transforms.inputs,
])
rows = []
for row in (0, len(current_raw) // 2, len(current_raw) - 1):
    old = previous_raw[row]
    raw_obs = {"images": {
        "cam_high": np.asarray(old["observation.images.head_rgb"]),
        "cam_left_wrist": np.asarray(old["observation.images.left_wrist_rgb"]),
        "cam_right_wrist": np.asarray(old["observation.images.right_wrist_rgb"])},
        "state": np.asarray(old["observation.state"]), "prompt": cfg.data.default_prompt}
    online = inference(copy.deepcopy(raw_obs))
    offline = training[row]
    errors = {}
    for key in offline["image"]:
        actual, expected = np.asarray(online["image"][key]), np.asarray(offline["image"][key])
        error = np.abs(actual.astype(float) - expected.astype(float))
        # H264 YUV420->RGB chroma interpolation and float->uint8 rounding may differ
        # slightly after rotation. Orientation must agree within 2 RGB code values.
        assert error.max() <= 2, (row, key, float(error.max()))
        errors[key] = {"max_abs_rgb": float(error.max()), "mean_abs_rgb": float(error.mean())}
    for key in ("tokenized_prompt", "tokenized_prompt_mask"):
        np.testing.assert_array_equal(online[key], offline[key])
    rows.append({"row": row, "image_errors": errors})
print(json.dumps({"status": "passed", "training_inference_orientation": rows}, indent=2))
