load("@pip-inference-client//:requirements.bzl", all_pip_inference_client = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "py_binary", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

PIP_DEPS = [
    "jsonschema",
    "numpy",
    "pillow",
]

CONFIGS = glob(["configs/**/*.yaml"]) + ["//ladder/components/inference/models/vlams/configs"]

make_interactive_tools(
    requirements_txt = "//ladder/components/inference/api/client:requirements.txt.lock",
    deps = [":gpt_policy"] + all_pip_inference_client,
)

py_library(
    name = "gpt_policy",
    srcs = glob(["*.py"]),
    pip_deps = PIP_DEPS,
    deps = [
        "//ladder/components/inference/api/client",
        "//ladder/components/inference/api/core",
        "//ladder/core/common/logger",
        "//ladder/core/databib",
        "@gpt_policy_repo//:gpt_policy",
    ],
)

py_binary(
    name = "run_model",
    srcs = ["//ladder/components/inference/api/client:run_model.py"],
    data = CONFIGS,
    main = "//ladder/components/inference/api/client:run_model.py",
    pip_deps = PIP_DEPS,
    deps = [":gpt_policy"],
)

make_py_checks(
    name = "gpt_policy_checks",
    deps = [":gpt_policy"],
)
