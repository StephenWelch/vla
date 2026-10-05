"""Compatibility launcher; implementation lives in ogbench_mjwarp.prepare."""

import runpy

if __name__ == "__main__":
    runpy.run_module("ogbench_mjwarp.prepare", run_name="__main__")
