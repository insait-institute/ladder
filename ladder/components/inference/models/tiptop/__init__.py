import importlib
import os
from pathlib import Path

import ninja
import torch
from filelock import FileLock

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

# cuRobo and M2T2's pointnet2 compile CUDA extensions the first time they are imported, against a
# toolchain and an architecture-specific cache that have to be configured before the import, and
# without two lanes building into the same half-written cache. Doing it here means every module in
# the package can import them at the top of the file: Python runs this first.
_JIT_MODULES = ("curobo.wrap.reacher.motion_gen", "m2t2.m2t2")


def cache_dir() -> Path:
    path = Path(os.environ.get("TIPTOP_CACHE_DIR", str(Path.home() / ".cache" / "tiptop")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def runfile(*parts: str) -> Path:
    here = Path(__file__).absolute().parent
    for parent in [here, *here.parents]:
        candidate = parent.joinpath(*parts)
        if candidate.exists():
            return candidate
    raise RuntimeError(f"{'/'.join(parts)} is missing from the runfiles")


def _build_extensions() -> None:
    os.environ.setdefault("CUDA_HOME", str(runfile("cuda_home", "bin", "nvcc").parents[1]))
    # torch builds the extensions with `ninja`; the env ships it as a pip package, but under Bazel
    # its binary is not on PATH and ninja.BIN_DIR is empty, so find it next to the package
    ninja_bin = ninja.BIN_DIR or str(Path(ninja.__file__).resolve().parents[2] / "bin")
    os.environ["PATH"] = os.pathsep.join([ninja_bin, os.environ.get("PATH", "")])
    # torch derives the arch flags from the live devices, so without one it indexes an empty list
    # and the import dies before it reaches anything of ours.
    arch, arch_list = ("cpu", "8.0")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch, arch_list = (f"sm{major}{minor}", f"{major}.{minor}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache_dir() / "torch_extensions" / arch))
    extension_dir = Path(os.environ["TORCH_EXTENSIONS_DIR"])
    extension_dir.mkdir(parents=True, exist_ok=True)
    built = extension_dir / "built.ok"
    # Without a card there are no concurrent lanes to serialise, and warming here would make a
    # plain import pay for a full compile.
    if arch == "cpu" or built.exists():
        return
    with FileLock(str(extension_dir / "build.lock"), timeout=3600):
        for lock in extension_dir.glob("*/lock"):
            lock.unlink(missing_ok=True)
            LOGGER.info(f"cleared a torch extension lock left behind by a dead build at {lock}")
        for module in _JIT_MODULES:
            importlib.import_module(module)
        built.touch()


_build_extensions()
