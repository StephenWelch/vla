"""Compatibility launcher; implementation lives in vla_libero.train."""

import runpy

if __name__ == "__main__":
    runpy.run_module("vla_libero.train", run_name="__main__")
