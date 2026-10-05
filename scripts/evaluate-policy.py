"""Compatibility launcher; implementation lives in vla_tools.evaluate."""

import runpy

if __name__ == "__main__":
    runpy.run_module("vla_tools.evaluate", run_name="__main__")
