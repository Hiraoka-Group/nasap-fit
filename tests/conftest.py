import sys

# Try to import the real C++ extension first.
# If it is not built, install a MagicMock so that pure-Python tests can still run.
try:
    import nasap_fit._core
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["nasap_fit._core"] = MagicMock()
