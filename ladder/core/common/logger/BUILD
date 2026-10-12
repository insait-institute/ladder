load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "logger",
    srcs = [
        "__init__.py",
        "colorize.py",
        "logger.py",
        "stdout.py",
    ],
    deps = [],
)

make_py_checks(
    name = "logger_tests",
    test_srcs = ["test/test_logger.py"],
    deps = [":logger"],
)
