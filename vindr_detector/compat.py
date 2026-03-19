from __future__ import annotations

import sys
import types
from importlib import metadata


def ensure_pkg_resources() -> None:
    """Provide a minimal pkg_resources shim for MMEngine on newer setuptools."""
    try:
        import pkg_resources  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType('pkg_resources')

    class DistributionNotFound(Exception):
        """Raised when a distribution cannot be found."""

    def get_distribution(name: str):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise DistributionNotFound(name) from exc
        return types.SimpleNamespace(project_name=name, version=version)

    module.DistributionNotFound = DistributionNotFound
    module.get_distribution = get_distribution
    sys.modules['pkg_resources'] = module
