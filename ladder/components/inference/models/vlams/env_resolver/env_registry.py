# Registry of inference environments. Each version maps to the bazel package providing that env's
# pip packages and corresponding py_binary target
#
# See the "Inference environments" section of the package README.
# The release ships one env: v0, the one the released Bridge policy runs in (it carries no
# env_metadata.yaml, so it resolves to DEFAULT_ENV_VERSION).
DEFAULT_ENV_VERSION = "v0"  # Models exported without env_metadata.yaml default to this version
ENV_PACKAGES = {
    "v0": "//ladder/components/inference/models/vlams/envs/v0",
}
