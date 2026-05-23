from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nasap-fit")
except PackageNotFoundError:
    __version__ = "unknown"

from .api import NasapFit, config_from_yaml, default_config

__all__ = ["NasapFit", "config_from_yaml", "default_config", "__version__"]