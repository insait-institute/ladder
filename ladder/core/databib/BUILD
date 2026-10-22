load("@pip-databib//:requirements.bzl", all_pip_databib = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_library")

package(default_visibility = ["//visibility:public"])

pip_compile(
    name = "pip_compile_databib",
    extra_args = [
        # Include the optional (test-only) dependencies for this package when compiling
        "--extra=ladder-core-databib-optional",
    ],
    requirements_in = "//ladder/core/databib:pyproject.toml",
    requirements_txt = "//ladder/core/databib:requirements.txt.lock",
    tags = ["build"],  # Tag for the test target
)

py_library(
    name = "databib",
    srcs = ["__init__.py"],
    deps = [
        "//ladder/core/databib/config",
        "//ladder/core/databib/dataclasses",
        "//ladder/core/databib/utils:classproperty",
        "//ladder/core/databib/utils:typing_utils",
    ],
)

make_interactive_tools(
    deps = all_pip_databib + [":databib"],
)
