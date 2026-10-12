load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "classproperty",
    srcs = ["classproperty.py"],
)

py_library(
    name = "date_utils",
    srcs = ["date_utils.py"],
    pip_deps = ["python-dateutil"],
)

py_library(
    name = "json_utils",
    srcs = ["json_utils.py"],
    pip_deps = ["numpy"],
    deps = [
        ":date_utils",
        "//ladder/core/databib/dataclasses",
    ],
)

py_library(
    name = "typing_utils",
    srcs = ["typing_utils.py"],
    pip_deps = ["numpy"],
)

make_py_checks(
    name = "utils_tests",
    test_srcs = ["test/test_typing_utils.py"],
    deps = [
        ":typing_utils",
        "//ladder/core/databib/config",
        "//ladder/core/databib/dataclasses",
    ],
)
