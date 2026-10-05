"""Compatibility entry point for the shared local-policy trainer."""

from importlib import import_module

if __name__ == "__main__":
    import_module("train-policy").main()
