import pathlib
import unittest

from ladder.components.inference.models.vlams.env_resolver.env_registry import ENV_PACKAGES
from ladder.core.common.platform import REPO_ROOT

_REPO_ROOT = pathlib.Path(REPO_ROOT)


def _pyproject_for_label(label):
    return _REPO_ROOT / label.removeprefix("//") / "pyproject.toml"


class TestEnvRegistry(unittest.TestCase):
    def test_every_env_has_a_pyproject(self):
        for tag, label in ENV_PACKAGES.items():
            pyproject = _pyproject_for_label(label)
            self.assertTrue(pyproject.exists(), f"env '{tag}' pyproject missing at {pyproject}")


if __name__ == "__main__":
    unittest.main()
