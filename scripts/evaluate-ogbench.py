"""Compatibility launcher; implementation lives in ogbench_mjwarp.evaluate."""

import runpy

if __name__ == "__main__":
    runpy.run_module("ogbench_mjwarp.evaluate", run_name="__main__")
