"""Compatibility entrypoint for shared hooks."""
import sys
from vla_tools import hooks as _module

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
