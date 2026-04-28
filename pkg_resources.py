"""Compatibility shim for `pkg_resources` on modern Python.

Some upstream dependencies (e.g. BookNLP) still import `pkg_resources` for
`resource_filename`, but newer environments may not ship it.

We provide the minimal API BookNLP needs: `resource_filename(package, resource)`.
"""

from __future__ import annotations

from importlib import resources


def resource_filename(package: str, resource: str) -> str:
    """Return an absolute path for a package data file."""
    return str(resources.files(package).joinpath(resource))

