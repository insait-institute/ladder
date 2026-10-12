load("@pip-inference-server//:requirements.bzl", all_pip_inference_server = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

pip_compile(
    name = "pip_compile_inference_server",
    requirements_in = "//ladder/components/inference/api/server:pyproject.toml",
    requirements_txt = "//ladder/components/inference/api/server:requirements.txt.lock",
    tags = ["build"],  # Tag for the test target
)

make_interactive_tools(
    deps = [":server"] + all_pip_inference_server,
)

py_library(
    name = "server",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    data = [
        "pyproject.toml",
        "requirements.txt.lock",
    ],
    pip_deps = [
        "numpy",
        "opencv-python-headless",
        "pyrealsense2",
        "pyudev",
    ],
    deps = [
        "//ladder/components/inference/api/core",
        "//ladder/core/databib",
    ],
)

make_py_checks(
    name = "server_tests",
    test_srcs = glob(["test/*.py"]),
    deps = [":server"],
)
