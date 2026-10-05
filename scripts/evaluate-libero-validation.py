"""Compatibility launcher; implementation lives in vla_libero.evaluate_validation."""

import runpy

if __name__ == "__main__":
    runpy.run_module("vla_libero.evaluate_validation", run_name="__main__")
