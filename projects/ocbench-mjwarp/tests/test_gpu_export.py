import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("graph,batch", [(False, 1), (True, 1), (False, 2), (False, 4)])
def test_device_replay_matches_host_with_unequal_lengths(graph, batch):
    from ocbench_mjwarp.config import FIELDS
    from ocbench_mjwarp.environment import Simulation
    from ocbench_mjwarp.gpu_video import DeviceReplay

    sim = Simulation([83000, 83001] * batch, audit=False)
    try:
        initial = sim.snapshot()
        arrays = []
        for world, length in enumerate((3, 2)):
            episode = {
                f"sim/{k}": np.repeat(initial[k][world][None], length, axis=0)
                for k in FIELDS
            }
            episode["action"] = np.zeros((length, 7), dtype=np.float32)
            episode["sim/qpos"][:, 0] += np.arange(length) * 0.1
            arrays.append(episode)
        replay = DeviceReplay(sim, arrays, graph=graph, render_batch_frames=batch)
        for tick in (2, 0, 1):
            sim.restore(
                {
                    k: np.stack(
                        [
                            a[f"sim/{k}"][min(tick + offset, len(a["action"]) - 1)]
                            for offset in range(batch)
                            for a in arrays
                        ]
                    )
                    for k in FIELDS
                },
                forward=False,
            )
            expected = sim.render()
            bgra = replay.render(tick)
            sim.torch_stream.synchronize()
            rgb = bgra.cpu().numpy()[..., :3][..., ::-1]
            for camera, key in enumerate(("front", "wrist")):
                np.testing.assert_array_equal(rgb[:, camera], expected[key])
    finally:
        sim.close()
