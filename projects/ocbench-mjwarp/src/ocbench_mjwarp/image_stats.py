"""Exact, bounded-memory statistics for sampled uint8 RGB images."""

import numpy as np


class ImageStats:
    """Accumulate 256-bin channel counts; quantiles use NumPy's linear rule.

    Unlike LeRobot's adaptive histogram, discrete intensities need no rebins.
    The caller supplies the same spatial samples used by the native writer.
    """

    def __init__(self):
        self.histogram = np.zeros((3, 256), dtype=np.int64)

    def update(self, pixels):
        if pixels.dtype != np.uint8 or pixels.shape[-1] != 3:
            raise ValueError("Expected uint8 RGB samples")
        pixels = pixels.reshape(-1, 3)
        for channel in range(3):
            self.histogram[channel] += np.bincount(pixels[:, channel], minlength=256)

    def get_statistics(self):
        hist = self.histogram
        count = int(hist[0].sum())
        if count < 2:
            raise ValueError("At least two RGB samples required")
        values = np.arange(256, dtype=np.float64)
        mean = hist @ values / count
        result = {
            "min": (hist > 0).argmax(axis=1),
            "max": 255 - (hist[:, ::-1] > 0).argmax(axis=1),
            "mean": mean,
            "std": np.sqrt(np.maximum(0, hist @ (values**2) / count - mean**2)),
            "count": np.array([count]),
        }
        cdf = hist.cumsum(axis=1)
        for q in (0.01, 0.10, 0.50, 0.90, 0.99):
            rank = q * (count - 1)
            low, high = int(np.floor(rank)), int(np.ceil(rank))
            a = np.array([np.searchsorted(c, low, side="right") for c in cdf])
            b = np.array([np.searchsorted(c, high, side="right") for c in cdf])
            result[f"q{int(q * 100):02d}"] = a + (b - a) * (rank - low)
        return result
