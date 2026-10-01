load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "path_utils",
    srcs = ["path_utils.py"],
)

py_library(
    name = "platform",
    srcs = ["platform.py"],
    pip_deps = ["psutil"],
)

py_library(
    name = "np_utils",
    srcs = ["np_utils.py"],
    pip_deps = ["numpy"],
)

py_library(
    name = "git_utils",
    srcs = ["git_utils.py"],
    data = [":git_meta"],
    pip_deps = ["gitpython"],
    deps = [
        ":platform",
        "//ladder/core/common/logger",
    ],
)

filegroup(
    name = "git_meta",
    srcs = glob([
        "**/git.diff",
        "**/git.hash",
        "**/git.branch",
    ]),
)

make_py_checks(
    name = "path_utils_tests",
    deps = [":path_utils"],
)

make_py_checks(
    name = "platform_tests",
    deps = [":platform"],
)

make_py_checks(
    name = "np_utils_tests",
    test_srcs = ["test/test_np_utils.py"],
    deps = [":np_utils"],
)

make_py_checks(
    name = "git_utils_tests",
    test_srcs = ["test/test_git_utils.py"],
    deps = [
        ":git_utils",
        ":path_utils",
    ],
)
