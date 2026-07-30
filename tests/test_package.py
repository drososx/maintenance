import subprocess
import sys
import unittest

from maintenance import __version__, get_project_name


class MaintenancePackageTests(unittest.TestCase):
    def test_project_name(self) -> None:
        self.assertEqual(get_project_name(), "maintenance")

    def test_version(self) -> None:
        self.assertEqual(__version__, "0.1.0")

    def test_module_entry_point(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "maintenance"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "maintenance")


if __name__ == "__main__":
    unittest.main()
