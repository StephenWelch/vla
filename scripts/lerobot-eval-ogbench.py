"""Compatibility launcher; implementation lives in ogbench_mjwarp.eval_native."""

import runpy

if __name__ == "__main__":
    runpy.run_module("ogbench_mjwarp.eval_native", run_name="__main__")
