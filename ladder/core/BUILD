load("@pip-core//:requirements.bzl", all_pip_core = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile")

package(default_visibility = ["//visibility:public"])

pip_compile(
    name = "pip_compile_core",
    extra_args = [
        # Include the optional dependencies for this package when compiling requirements.txt
        "--extra=ladder-core-optional",
    ],
    requirements_in = "//ladder/core:pyproject.toml",
    requirements_txt = "//ladder/core:requirements.txt.lock",
    tags = ["build"],  # Tag for the test target
)

make_interactive_tools(
    deps = [
        "//ladder/core/common:git_utils",
        "//ladder/core/common:np_utils",
        "//ladder/core/common:path_utils",
        "//ladder/core/common:platform",
        "//ladder/core/common/logger",
        "//ladder/core/common/timing",
        "//ladder/core/common/video_utils",
        "//ladder/core/compute:seeder",
        "//ladder/core/compute:tensor_utils",
    ] + all_pip_core,
)
