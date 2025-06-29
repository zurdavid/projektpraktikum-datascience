"""
version.py

Module for managing the version of the fraud detection REST API.
"""

from importlib.metadata import version, PackageNotFoundError
import pathlib
import tomllib

try:
    # TODO change name
    __version__ = version("SCO-fraud-REST-api")
except PackageNotFoundError:
    # Fallback: parse from pyproject.toml
    pyproject_file = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_file.open("rb") as f:
        pyproject_data = tomllib.load(f)
    __version__ = pyproject_data["project"]["version"]
except Exception:
    __version__ = "unknown"
