"""Read the left SO-101 and run one base-model inference without motor commands."""

import argparse
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
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port="COM4", id="left"))
    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        raise RuntimeError("Left camera 1 did not open")
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
        "observation.images.camera1": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    }
    observation = prepare_observation_for_inference(
        observation, torch.device("cuda"), args.task, "so101_follower"
    )
    with torch.inference_mode():
        action = post(policy.select_action(pre(observation))).squeeze(0).cpu().numpy()

    print("Current joints:", dict(zip(names, state.tolist(), strict=True)))
    print("Proposed joints:", dict(zip(names, action.tolist(), strict=True)))
    print("Difference:", dict(zip(names, (action - state).tolist(), strict=True)))
    print(f"Camera frame: {output}")
    print("No motor commands sent")


if __name__ == "__main__":
    main()
