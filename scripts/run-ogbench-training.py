"""Compatibility launcher; implementation lives in ogbench_mjwarp.pipeline."""

import runpy

if __name__ == "__main__":
    runpy.run_module("ogbench_mjwarp.pipeline", run_name="__main__")
