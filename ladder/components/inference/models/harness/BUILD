load("@pip-inference-client//:requirements.bzl", all_pip_inference_client = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "py_binary", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

# No segmentation model and no GPU, so no pip environment of its own: the env and its lock are
# the inference client's.
PIP_DEPS = [
    "numpy",
    "pillow",
]

make_interactive_tools(
    requirements_txt = "//ladder/components/inference/api/client:requirements.txt.lock",
    deps = [":harness"] + all_pip_inference_client,
)

py_library(
    name = "harness",
    srcs = glob(["**/*.py"]),
    pip_deps = PIP_DEPS,
    deps = [
        "//ladder/components/inference/api/client",
        "//ladder/components/inference/api/core",
        "//ladder/core/common/logger",
        "//ladder/core/databib",
    ],
)

py_binary(
    name = "run_model",
    srcs = ["//ladder/components/inference/api/client:run_model.py"],
    # A config imports the vlams chain by relative path at load time, so both trees must be here.
    data = glob(["configs/**/*.yaml"]) + ["//ladder/components/inference/models/vlams/configs"],
    main = "//ladder/components/inference/api/client:run_model.py",
    pip_deps = PIP_DEPS + ["ipdb"],
    deps = [":harness"],
)

make_py_checks(
    name = "harness_checks",
    deps = [":harness"],
)
