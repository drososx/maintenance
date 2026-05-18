"""maintenance package."""

from importlib.metadata import PackageNotFoundError, metadata

__version__ = "0.1.0"


def get_project_name() -> str:
    try:
        return metadata(__name__)["Name"]
    except PackageNotFoundError:
        return __name__
