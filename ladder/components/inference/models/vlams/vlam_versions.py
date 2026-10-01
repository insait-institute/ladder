from pathlib import Path
from typing import Union

import yaml

# Bundles exported before env_metadata.yaml existed carry no descriptor at all
DEFAULT_ENV_VERSION = "v0"

# First env whose exported processor carries a leading batch dimension through create_input and drops
# the inference_mode argument. Bundles from this version onwards use that ABI, everything older does not
BATCHED_IO_ENV_VERSION = "v3"

# First env whose exported processor takes observations WITH effector axis s.t. processor.create_input
# receives [1, num_past_scalars, num_effectors, D] instead of [1, num_past_scalars, D]. Also first env
# where reference frames are nested under processor_config.arms_io_config, and the first where
# create_input names the joint observation `joint_positions` rather than `joints`
EFFECTOR_AXIS_ENV_VERSION = "v4"

# First env whose exported processor takes a single `RoboticsDataInput` dataclass rather than a keyword
# list, and converts the observations from the frames they were read in to the frames the model is
# trained on. Bundles before this one never converted the observations at all
DATACLASS_IO_ENV_VERSION = "v5"


def env_version_ordinal(env_version: str) -> int:
    """Numeric rank of an env version tag, so that versions can be ordered"""
    return int(env_version.lstrip("v"))


def env_version_from_session(session_path: Union[str, Path]) -> str:
    """
    Read the env version a model bundle under `session_path` was exported with. Models exported
    before env_metadata.yaml existed carry no descriptor and fall back to DEFAULT_ENV_VERSION
    """
    descriptors = sorted(Path(session_path).glob(f"hf_export/*/src/env_metadata.yaml"))
    if not descriptors:
        return DEFAULT_ENV_VERSION

    with descriptors[0].open() as file:
        return yaml.safe_load(file)["env_version"]


def has_unbatched_processor_api(env_version: str) -> bool:
    """Whether a bundle exported with `env_version` predates BATCHED_IO_ENV_VERSION"""
    return env_version_ordinal(env_version) < env_version_ordinal(BATCHED_IO_ENV_VERSION)


def has_effector_axis_processor_api(env_version: str) -> bool:
    """Whether a bundle exported with `env_version` expects the effector axis on scalar observations"""
    return env_version_ordinal(env_version) >= env_version_ordinal(EFFECTOR_AXIS_ENV_VERSION)


def has_dataclass_processor_api(env_version: str) -> bool:
    """Whether a bundle exported with `env_version` takes a `RoboticsDataInput` through create_input"""
    return env_version_ordinal(env_version) >= env_version_ordinal(DATACLASS_IO_ENV_VERSION)
