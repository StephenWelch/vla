"""Read the left SO-101 and run one SmolVLA inference without motor commands."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("E:/vla-smolvla/smolvla_base"))
    parser.add_argument("--task", default="Move the gripper toward the green cylinder.")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--camera-feature", help="Policy image key; inferred for single-camera checkpoints")
    args = parser.parse_args()

    config_path = args.model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing policy config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    image_features = [key for key, spec in config["input_features"].items() if spec["type"] == "VISUAL"]
    if config["input_features"]["observation.state"]["shape"] != [6] or config["output_features"]["action"]["shape"] != [6]:
        raise ValueError("Policy must have six-dimensional SO-101 state and action features")
    camera_feature = args.camera_feature or (image_features[0] if image_features else None)
    if camera_feature not in image_features:
        raise ValueError(f"Camera feature must be one of {image_features}")
    if len(image_features) > 1:
        print(f"Policy expects {len(image_features)} image features; supplying only {camera_feature}.")

    robot = SO101Follower(SO101FollowerConfig(port="COM4", id="left"))
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Camera {args.camera_index} did not open")
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame = None
        for _ in range(30):
            ok, image = camera.read()
            if ok:
                frame = image
        if frame is None:
            raise RuntimeError("Left camera returned no frame")
        output = Path("outputs/so101_left_shadow.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), frame)
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(luminance.mean())
        bright_pixel_level = float(np.percentile(luminance, 95))

        robot.bus.connect()
        positions = robot.bus.sync_read("Present_Position")
        names = list(robot.bus.motors)
        state = np.array([positions[name] for name in names], dtype=np.float32)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
        camera.release()

    policy = SmolVLAPolicy.from_pretrained(args.model).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy.config,
        str(args.model),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    observation = {
        "observation.state": state,
        camera_feature: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    }
    observation = prepare_observation_for_inference(
        observation, torch.device("cuda"), args.task, "so101_follower"
    )
    with torch.inference_mode():
        action = post(policy.select_action(pre(observation))).squeeze(0).cpu().numpy()

    print("Current joints:", dict(zip(names, state.tolist(), strict=True)))
    print("Proposed joints:", dict(zip(names, action.tolist(), strict=True)))
    print("Difference:", dict(zip(names, (action - state).tolist(), strict=True)))
    print(f"Camera brightness: mean {mean_brightness:.1f}/255, 95th percentile {bright_pixel_level:.1f}/255")
    print(f"Largest joint change: {float(np.abs(action - state).max()):.1f} degrees/units")
    print(f"Camera frame: {output}")
    print("No motor commands sent")


if __name__ == "__main__":
    main()
