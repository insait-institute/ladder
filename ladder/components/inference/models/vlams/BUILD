load("//build_support/python:pip_defs.bzl", "py_binary", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

exports_files([
    "test/test_vlam_robot_inference.py",
])

PIP_DEPS = [
    "backports.strenum",
    "einops",
    "numpy",
    "pandas",
    "transformers",
    "torch",
    "torchvision",
    "timm",
    "roma",
    "sentencepiece",
]

# NOTE: The current env doesn't have its own pyproject.toml and requirements.txt.lock. The env
# default env is determined from paths in pip-inference.MODULE.bazel.
# This package only hosts the shared code and the version-neutral launchers; venv/ipython targets
# live in each envs/<tag> package, bound to that env's own hub.

# Env-adaptive version of the vlams py_library. Used for py_binary which automatically resolve target env
py_library(
    name = "vlams_adaptive",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    data = glob(["**/*.yaml"]),
    # By default we use the common inference-client env for the library such that it doesn't include
    # any specific ML pip deps. py_binary and py_test targets route to the right pip environment
    pip_env = "inference-client",
    deps = [
        "//ladder/components/inference/api/client",
        "//ladder/components/inference/api/core",
        "//ladder/core/compute:collate",
        "//ladder/core/databib",
    ],
)

# The current (latest) vlams py_library
py_library(
    name = "vlams",
    pip_deps = PIP_DEPS,
    deps = [":vlams_adaptive"],
)

make_py_checks(
    name = "vlams_tests",
    test_srcs = glob(
        ["test/*.py"],
        exclude = [
            "test/test_vlam_robot_inference.py",
        ],
    ),
    deps = [":vlams"],
)

py_binary(
    name = "run_model",
    srcs = ["//ladder/components/inference/models/vlams/env_resolver:target_launcher.py"],
    # We manually pass the bazel package locations of the different environments such that the
    # launcher can resolve the correct implementation based on env_metadata.yaml in the model bundle
    args = [
        "--env-target",
        "v0=$(rlocationpath //ladder/components/inference/models/vlams/envs/v0:run_model)",
    ],
    # The environment is a dependency of the launcher to the launcher so that they are built and available
    # in the runfiles tree
    data = [
        "//ladder/components/inference/models/vlams/envs/v0:run_model",
    ],
    main = "//ladder/components/inference/models/vlams/env_resolver:target_launcher.py",
    # By default we use the common inference-client env for the launcher, which doesn't include
    # any specific ML pip deps. The launcher routes to the right pip environment
    pip_env = "inference-client",
    deps = ["//ladder/components/inference/models/vlams/env_resolver"],
)

# Scores a planner's zero-shot decompositions. The source sits with the plan metrics it reports;
# the binary is here because the local HF backend needs this package's torch/transformers env
# (--backend gateway needs none of it, but one target is simpler than two).
py_binary(
    name = "score_decomposition",
    srcs = ["//ladder/components/inference/api/client:metrics/tools/score_decomposition.py"],
    main = "//ladder/components/inference/api/client:metrics/tools/score_decomposition.py",
    pip_deps = PIP_DEPS + [
        "pillow",
        "pyyaml",  # the gateway backend reads the eval's own vlm.yaml rather than restating the URL
    ],
    deps = [":vlams"],
)

# Scores a VLM on the subtask-state samples (subtask verification, per-goal state tracking,
# on-demand next-subtask generation), the frame-level companion of score_decomposition. Same
# placement rationale: the local HF backend needs this package's torch/transformers env.
py_binary(
    name = "score_subtask_states",
    srcs = ["//ladder/components/inference/api/client:metrics/tools/score_subtask_states.py"],
    main = "//ladder/components/inference/api/client:metrics/tools/score_subtask_states.py",
    pip_deps = PIP_DEPS + [
        "pillow",
        "pyyaml",
    ],
    deps = [":vlams"],
)

py_binary(
    name = "publish_orchestrator_benchmark",
    srcs = ["//ladder/components/inference/api/client:metrics/tools/publish_orchestrator_benchmark.py"],
    main = "//ladder/components/inference/api/client:metrics/tools/publish_orchestrator_benchmark.py",
    pip_deps = PIP_DEPS + [
        "huggingface-hub",
        "pyarrow",
    ],
    deps = [":vlams"],
)

py_binary(
    name = "batch_inference_server",
    srcs = ["//ladder/components/inference/models/vlams/env_resolver:target_launcher.py"],
    # We manually pass the bazel package locations of the different environments such that the
    # launcher can resolve the correct implementation based on env_metadata.yaml in the model bundle
    args = [
        "--env-target",
        "v0=$(rlocationpath //ladder/components/inference/models/vlams/envs/v0:batch_inference_server)",
    ],
    # The environment is a dependency of the launcher to the launcher so that they are built and available
    # in the runfiles tree
    data = [
        "//ladder/components/inference/models/vlams/envs/v0:batch_inference_server",
    ],
    main = "//ladder/components/inference/models/vlams/env_resolver:target_launcher.py",
    # By default we use the common inference-client env for the launcher, which doesn't include
    # any specific ML pip deps. The launcher routes to the right pip environment
    pip_env = "inference-client",
    deps = ["//ladder/components/inference/models/vlams/env_resolver"],
)
