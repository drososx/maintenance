"""Command line entry point for the maintenance package."""

from . import get_project_name


def main() -> None:
    print(get_project_name())


if __name__ == "__main__":
    main()
