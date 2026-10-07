import numpy as np
import pytest
from ocbench_mjwarp.image_stats import ImageStats


@pytest.mark.parametrize("constant", [False, True])
def test_image_stats_matches_numpy_across_batches(constant):
    pixels = np.random.default_rng(42).integers(0, 256, (1004, 3), dtype=np.uint8)
    if constant:
        pixels[:] = [0, 127, 255]
    stats = ImageStats()
    for batch in np.array_split(pixels, 17):
        stats.update(batch)
    actual = stats.get_statistics()
    for key in ("min", "max", "mean", "std"):
        np.testing.assert_allclose(
            actual[key], getattr(pixels, key)(axis=0), atol=1e-10
        )
    np.testing.assert_array_equal(actual["count"], [len(pixels)])
    for q in (0.01, 0.10, 0.50, 0.90, 0.99):
        np.testing.assert_allclose(
            actual[f"q{int(q * 100):02d}"], np.quantile(pixels, q, axis=0)
        )


def test_image_stats_rejects_invalid_input():
    stats = ImageStats()
    with pytest.raises(ValueError):
        stats.get_statistics()
    with pytest.raises(ValueError):
        stats.update(np.zeros((10, 3), dtype=np.float32))
