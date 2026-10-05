"""Compatibility launcher; implementation lives in ogbench_mjwarp.benchmark_pipeline."""

import runpy

if __name__ == "__main__":
    runpy.run_module("ogbench_mjwarp.benchmark_pipeline", run_name="__main__")
