import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ladder.components.inference.models.vlams.env_resolver.env_registry import DEFAULT_ENV_VERSION
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, ConfigParser

LOGGER = get_logger(__name__)

_ENV_METADATA = "env_metadata.yaml"
_ENV_TARGET_FLAG = "--env-target"


class TargetLauncher:
    """
    Automatically determines the target inference environment for a model (based on env_metadata.yaml
    in the model bundle) and re-routes to a target py_binary for that environment by launching it via
    `os.execv`. If env_metadata.yaml is missing, falls back to DEFAULT_ENV_VERSION

    os.execv: Executes a new program, replacing the current process; does not return. On Unix, the
    new executable is loaded into the current process, and will have the same process id as the caller.
    Errors will be reported as OSError exceptions.
    """

    def __init__(self) -> None:
        from python.runfiles import Runfiles

        self._runfiles = Runfiles.Create()
        if self._runfiles is None:
            raise RuntimeError("no runfiles available; run this as a bazel target")

    def resolve_version(self, argv: List[str]) -> str:
        ckpt_path = _find_ckpt_path(ConfigParser(Config).parse_args_to_dict(argv))
        if not ckpt_path:
            # A batched-eval client config has no checkpoint of its own: the model lives in the shared
            # batch_inference_server, so there is no bundle here to read a descriptor from.
            LOGGER.info(f"no ckpt_path in config; using env version {DEFAULT_ENV_VERSION}")
            return DEFAULT_ENV_VERSION

        # ckpt_path is <session>/checkpoints/<ckpt>; the exported bundle lives under <session>.
        session_dir = Path(ckpt_path).parent.parent
        descriptors = sorted(session_dir.glob(f"hf_export/*/src/{_ENV_METADATA}"))
        if not descriptors:
            return DEFAULT_ENV_VERSION

        with descriptors[0].open() as file:
            return yaml.safe_load(file)["env_version"]

    def run(self, argv: List[str]) -> None:
        impl_rlocationpaths, forwarded = _split_env_impl_args(argv)
        version = self.resolve_version(forwarded)
        if version not in impl_rlocationpaths:
            raise SystemExit(
                f"[run_model] no implementation for env version '{version}' "
                f"(available: {sorted(impl_rlocationpaths)})"
            )
        impl = self._runfiles.Rlocation(impl_rlocationpaths[version])
        if impl is None:
            raise SystemExit(f"[run_model] impl binary for env '{version}' not found in runfiles")
        # Pass our runfiles env to the exec'd impl so it resolves its own files, whether the tree is
        # directory- or manifest-based.
        os.environ.update(self._runfiles.EnvVars())
        os.execv(impl, [impl, *forwarded])


def _split_env_impl_args(argv: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Split the `--env-target <version>=<rlocationpath>` options (injected from BUILD) out of `argv`,
    returning the version -> rlocationpath map and the remaining arguments to forward to the impl.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(_ENV_TARGET_FLAG, action="append", default=[], dest="env_impl")
    known, forwarded = parser.parse_known_args(argv)
    impl_rlocationpaths = dict(entry.split("=", 1) for entry in known.env_impl)
    return impl_rlocationpaths, forwarded


def _find_ckpt_path(node: Any) -> Optional[str]:
    """Depth-first search for a `ckpt_path` string in the merged config dict."""
    if isinstance(node, dict):
        ckpt_path = node.get("ckpt_path")
        if isinstance(ckpt_path, str):
            return ckpt_path
        for value in node.values():
            found = _find_ckpt_path(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_ckpt_path(item)
            if found is not None:
                return found
    return None


def main() -> None:
    TargetLauncher().run(sys.argv[1:])


if __name__ == "__main__":
    main()
