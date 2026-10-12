load("@pip-inference-client//:requirements.bzl", all_pip_inference_client = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_binary", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks", "make_py_test")

package(default_visibility = ["//visibility:public"])

# The server entry point is a binary in each policy's own package (models/vlams envs, the own-env
# external models) so it runs with that policy's pip env. Its tests are not: the wire types are
# policy-agnostic, so they run once here, in this package's own env.
exports_files(["batch_inference/batch_policy_server.py"])

pip_compile(
    name = "pip_compile_inference_client",
    requirements_in = "//ladder/components/inference/api/client:pyproject.toml",
    requirements_txt = "//ladder/components/inference/api/client:requirements.txt.lock",
    tags = ["build"],  # Tag for the test target
)

make_interactive_tools(
    deps = [":client"] + all_pip_inference_client,
)

py_library(
    name = "client",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    # The hlc/configs yamls are data because an eval config imports one by relative path at load
    # time, so it has to be in the runfiles.
    data = glob([
        "hlc/configs/*.yaml",
        "batch_inference/configs/*.yaml",
    ]) + [
        "pyproject.toml",
        "requirements.txt.lock",
    ],
    pip_deps = [
        "filelock",
        "opencv-python-headless",
        "openai",
        "pillow",
        "PyYAML",
        "rerun-sdk",
    ],
    deps = [
        # Vendored upstream VoLo; hlc/volo.py drives its strategies in-process.
        "@volo_agent_repo//:vlm_orchestrator",
        "//ladder/components/inference/api/core",
        "//ladder/core/common:platform",
        "//ladder/core/compute:seeder",
        "//ladder/core/databib",
    ],
)

# Cuts subtask-state samples -- frames plus the per-goal ground truth at that instant -- from
# recorded eval episodes. Pure harvesting (episode logs + mp4s in, PNGs + subtask_states.json out),
# so it lives in this env; the scorer that drives a VLM over the samples is
# //ladder/components/inference/models/vlams:score_subtask_states.
py_binary(
    name = "export_subtask_states",
    srcs = ["metrics/tools/export_subtask_states.py"],
    main = "metrics/tools/export_subtask_states.py",
    deps = [":client"],
)

# Failure causes, grasp fates and event timelines from the object-level trace in episode logs.
py_binary(
    name = "event_report",
    srcs = ["metrics/tools/event_report.py"],
    main = "metrics/tools/event_report.py",
    deps = [":client"],
)

# Recomputes results.yaml from episode_log.yaml with the current metric registry.
py_binary(
    name = "rescore",
    srcs = ["metrics/tools/rescore.py"],
    main = "metrics/tools/rescore.py",
    deps = [":client"],
)

make_py_checks(
    name = "client_tests",
    test_srcs = glob(
        [
            "test/*.py",
            "batch_inference/test/*.py",
        ],
        exclude = ["test/test_simpler_episode_runner.py"],
    ),
    deps = [":client"],
)

filegroup(
    name = "test_configs",
    srcs = glob(["test/configs/**/*.yaml"]),
)

make_py_test(
    name = "test_simpler_episode_runner",
    srcs = ["test/test_simpler_episode_runner.py"],
    data = [":test_configs"],
    deps = [
        ":client",
        "//ladder/components/inference/scripts:mock_server_py",
        "//ladder/core/common:path_utils",
        "//ladder/core/compute:seeder",
    ],
)
