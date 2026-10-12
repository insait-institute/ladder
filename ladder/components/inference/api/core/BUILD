load("@pip-inference-core//:requirements.bzl", all_pip_inference_core = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

pip_compile(
    name = "pip_compile_inference_core",
    requirements_in = "//ladder/components/inference/api/core:pyproject.toml",
    requirements_txt = "//ladder/components/inference/api/core:requirements.txt.lock",
    tags = ["build"],  # Tag for the test target
)

make_interactive_tools(
    deps = [":core"] + all_pip_inference_core,
)

py_library(
    name = "core",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    data = [
        "requirements.txt.lock",
    ],
    pip_deps = [
        "msgpack",
        "msgpack-numpy",
        "opencv-python-headless",
        "pyzmq",
        "scipy",
        "typing-extensions",
    ],
    deps = [
        "//ladder/core/common/logger",
        "//ladder/core/databib",
    ],
)

make_py_checks(
    name = "core_tests",
    test_srcs = glob(["test/*.py"]),
    deps = [":core"],
)
