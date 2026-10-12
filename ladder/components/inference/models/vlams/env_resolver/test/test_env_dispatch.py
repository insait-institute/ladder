import os
import pathlib
import unittest

from ladder.components.inference.models.vlams.env_resolver.target_launcher import TargetLauncher
from ladder.core.common.path_utils import create_tempdir


def _ckpt_override(session_dir):
    ckpt = pathlib.Path(session_dir) / "checkpoints" / "model.pt"
    return [f"--policy_config.inference_config.ckpt_path={ckpt}"]


def _write_metadata(session_dir, version):
    src = pathlib.Path(session_dir) / "hf_export" / "nick" / "src"
    src.mkdir(parents=True)
    (src / "env_metadata.yaml").write_text(f"env_version: {version}\n")


class TestEnvDispatch(unittest.TestCase):
    def setUp(self):
        self.session_dir = create_tempdir(self)

    def test_resolve_version_reads_descriptor(self):
        _write_metadata(self.session_dir, "v1")
        self.assertEqual(TargetLauncher().resolve_version(_ckpt_override(self.session_dir)), "v1")

    def test_resolve_version_defaults_without_descriptor(self):
        (pathlib.Path(self.session_dir) / "checkpoints").mkdir(parents=True)
        self.assertEqual(TargetLauncher().resolve_version(_ckpt_override(self.session_dir)), "v0")

    def test_resolve_version_defaults_without_a_checkpoint(self):
        # A batched-eval client config carries no ckpt_path at all: the shared batch_inference_server
        # holds the model, so resolution must not raise.
        self.assertEqual(TargetLauncher().resolve_version(["--port", "5555"]), "v0")

    def test_resolve_version_from_config_file_with_env_expansion(self):
        # Exercises the full databib path: --config yaml + os.path.expandvars + ckpt extraction.
        _write_metadata(self.session_dir, "v1")
        config = pathlib.Path(self.session_dir) / "config.yaml"
        config.write_text(
            "policy_config:\n"
            "  inference_config:\n"
            "    ckpt_path: ${_TEST_SESSION_DIR}/checkpoints/model.pt\n"
        )
        os.environ["_TEST_SESSION_DIR"] = str(self.session_dir)
        try:
            self.assertEqual(TargetLauncher().resolve_version(["--config", str(config)]), "v1")
        finally:
            del os.environ["_TEST_SESSION_DIR"]


if __name__ == "__main__":
    unittest.main()
