"""Compatibility entrypoint for shared train."""
import sys
from vla_tools import train as _module

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
