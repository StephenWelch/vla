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


@pytest.mark.gpu
@pytest.mark.parametrize("image_size", [(480, 640), (240, 320)])
def test_reused_context_matches_fresh_episode_states(image_size):
    from ocbench_mjwarp.config import FIELDS
    from ocbench_mjwarp.environment import Simulation
    from ocbench_mjwarp.gpu_video import DeviceReplay

    reused = Simulation([83000, 83001] * 4, audit=False, image_size=image_size)
    try:
        reused.render()
        context = reused.renderer.context
        for seeds in ([83002, 83003], [83004, 83005]):
            fresh = Simulation(seeds * 4, audit=False, image_size=image_size)
            try:
                reused.reset_render(seeds * 4)
                state = fresh.snapshot()
                arrays = []
                for i, n in enumerate((5, 3)):
                    a = {
                        f"sim/{k}": np.repeat(state[k][i][None], n, axis=0)
                        for k in FIELDS
                    }
                    a["sim/qpos"][:, 0] += np.arange(n) * 0.03
                    a["action"] = np.zeros((n, 7), np.float32)
                    arrays.append(a)
                reference = DeviceReplay(fresh, arrays, render_batch_frames=4)
                actual = DeviceReplay(reused, arrays, render_batch_frames=4)
                for tick in (0, 4):
                    expected_gpu = reference.render(tick)
                    fresh.torch_stream.synchronize()
                    expected = expected_gpu.cpu().numpy().copy()
                    result_gpu = actual.render(tick)
                    reused.torch_stream.synchronize()
                    result = result_gpu.cpu().numpy().copy()
                    np.testing.assert_array_equal(result, expected)
                assert reused.renderer.context is context
            finally:
                fresh.close()
    finally:
        reused.close()
