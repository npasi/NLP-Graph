from __future__ import annotations

from importlib import resources


def resource_filename(package: str, resource: str) -> str:
    """Return an absolute path to a package data file.

    BookNLP imports `pkg_resources.resource_filename(...)` to locate bundled
    assets (e.g. tagsets). Newer Python setups may not ship `pkg_resources`
    by default; this lightweight shim is sufficient for BookNLP.
    """
    return str(resources.files(package).joinpath(resource))

