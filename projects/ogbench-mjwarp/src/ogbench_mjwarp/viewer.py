"""Interactive playback of recorded simulator states, without rerunning physics."""

import math
import threading
import time
from dataclasses import dataclass
from queue import SimpleQueue

from .io import load_sim_states, rollout_path
from .tasks import make_env, restore_cpu


@dataclass
class Playback:
    frames: int
    fps: float
    speed: float = 1.0
    paused: bool = False
    loop: bool = True
    frame: int = 0
    deadline: float = 0.0

    def update(self, now, keys=()):
        """Apply keyboard controls and advance against a monotonic clock."""
        for key in keys:
            if key == 32:  # Space
                self.paused = not self.paused
            elif key in (263, 262):  # Left/right: single frame and pause.
                self.frame = min(
                    self.frames - 1, max(0, self.frame + (1 if key == 262 else -1))
                )
                self.paused = True
            elif key in (268, 82):  # Home / R
                self.frame = 0
            elif key == 269:  # End
                self.frame, self.paused = self.frames - 1, True
            elif key in (61, 334, 45, 333):  # +/- including keypad
                self.speed = min(
                    16.0, max(0.0625, self.speed * (2 if key in (61, 334) else 0.5))
                )
            self.deadline = now + 1 / (self.fps * self.speed)
        if self.paused or now < self.deadline:
            return
        interval = 1 / (self.fps * self.speed)
        steps = 1 + int((now - self.deadline) / interval)
        frame = self.frame + steps
        if frame >= self.frames and not self.loop:
            self.frame, self.paused = self.frames - 1, True
        else:
            self.frame = frame % self.frames
        self.deadline += steps * interval


def view(
    root, episode=0, speed=1.0, paused=False, loop=True, camera="free", seconds=None
):
    import mujoco
    import mujoco.viewer

    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("seconds must be finite and positive")
    row, archive = rollout_path(root, episode)
    states = load_sim_states(archive)
    frames = len(states["qpos"])
    fps = float(row["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Recorded fps must be finite and positive")
    env = make_env(row["env_id"], row["seed"], row["task_id"], row["image_size"])
    base = env.unwrapped
    keys = SimpleQueue()
    playback = Playback(frames, fps, speed, paused, loop)
    print(
        f"Episode {episode}: {row['env_id']} / {row['outcome']} / {frames - 1} actions",
        flush=True,
    )
    print(row["instruction"], flush=True)
    print(
        "Space: pause | arrows: step | Home/R: restart | End: terminal | +/-: speed | Esc: close",
        flush=True,
    )
    viewer_threads = set()
    try:
        restore_cpu(env, {key: value[0] for key, value in states.items()})
        threads_before = set(threading.enumerate())
        with mujoco.viewer.launch_passive(
            base._model, base._data, key_callback=keys.put, show_right_ui=False
        ) as viewer:
            viewer_threads = set(threading.enumerate()) - threads_before
            with viewer.lock():
                if camera != "free":
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = base._model.camera(
                        "ur5e/wrist" if camera == "wrist" else camera
                    ).id
                else:
                    viewer.cam.lookat[:] = base._data.site_xpos[base._pinch_site_id]
                    viewer.cam.distance = 1.3
                    viewer.cam.azimuth, viewer.cam.elevation = 135, -25
            start = time.monotonic()
            playback.deadline = start + 1 / (fps * speed)
            viewer.sync()
            print("MuJoCo viewer ready", flush=True)
            previous_frame = playback.frame
            while viewer.is_running():
                now = time.monotonic()
                if seconds is not None and now - start >= seconds:
                    break
                pending = []
                while not keys.empty():
                    pending.append(keys.get())
                if 256 in pending:  # Esc
                    break
                playback.update(now, pending)
                if playback.frame != previous_frame or pending:
                    with viewer.lock():
                        restore_cpu(
                            env,
                            {
                                key: value[playback.frame]
                                for key, value in states.items()
                            },
                        )
                    previous_frame = playback.frame
                viewer.sync()
                time.sleep(max(0, 1 / 60 - (time.monotonic() - now)))
        return {"episode": episode, "frames": frames, "frame": playback.frame}
    finally:
        # MuJoCo 3.14 close() signals its daemon render thread without joining it.
        # Let GLFW teardown finish before releasing the environment or exiting Python.
        for thread in viewer_threads:
            thread.join()
        env.close()
