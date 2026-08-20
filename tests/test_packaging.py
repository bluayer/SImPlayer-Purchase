from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingContractTests(unittest.TestCase):
    def test_package_discovery_excludes_build_artifacts(self) -> None:
        config = tomllib.loads((REPO_ROOT / "src" / "pyproject.toml").read_text())
        package_find = config["tool"]["setuptools"]["packages"]["find"]

        self.assertEqual(package_find["include"], ["purchase_behavior_simulator*"])
        self.assertIn("build*", package_find["exclude"])
        self.assertFalse(package_find["namespaces"])


if __name__ == "__main__":
    unittest.main()
