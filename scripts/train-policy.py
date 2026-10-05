"""Compatibility launcher; implementation lives in ogbench_mjwarp.train."""

import runpy

if __name__ == "__main__":
    runpy.run_module("ogbench_mjwarp.train", run_name="__main__")
