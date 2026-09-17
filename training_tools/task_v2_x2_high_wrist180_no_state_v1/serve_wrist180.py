#!/usr/bin/env python3
"""Serve only this wrist180/no-state experiment; never commands robot hardware."""
import argparse
import dataclasses
from pathlib import Path

CONFIG = "pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b64_w32_8gpu_v1"
EXPERIMENT = "task_v2_x2_high_wrist180_no_state_10hz_h50_10k_b64_w32_8gpu_20260911_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-orientation", choices=("raw", "rotated180"), required=True,
                        help="raw: rotate each wrist once; rotated180: already rotated, do not rotate again")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from openpi import transforms
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config
    from wrist180_inference import RotateRawWristImages, self_test
    self_test()
    cfg = config.get_config(CONFIG)
    checkpoint = args.checkpoint.resolve(strict=True)
    root = (Path(cfg.checkpoint_base_dir) / CONFIG / EXPERIMENT).resolve()
    if checkpoint.parent != root or not checkpoint.name.isdigit():
        raise ValueError(f"Choose a finalized numeric checkpoint directly inside {root}")
    if not all((checkpoint / name).exists() for name in ("params", "assets", "train_state")):
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    if cfg.model.discrete_state_input is not False or cfg.policy_metadata.get("wrist_rotation_degrees") != {"left": 180, "right": 180}:
        raise ValueError("Unexpected state/camera policy configuration")
    cfg = dataclasses.replace(cfg, policy_metadata={**cfg.policy_metadata,
        "server_input_orientation": args.input_orientation,
        "server_applies_wrist_rotation": args.input_orientation == "raw"})
    # Training uses the encoded dataset. ONLY this raw inference entry adds rotation.
    repack = transforms.Group(inputs=[RotateRawWristImages()] if args.input_orientation == "raw" else [])
    policy = policy_config.create_trained_policy(cfg, str(checkpoint), repack_transforms=repack)
    websocket_policy_server.WebsocketPolicyServer(
        policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
