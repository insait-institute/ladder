load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "timing",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    pip_deps = [
        "numpy",
    ],
    deps = [
        "//ladder/core/common/logger",
    ],
)

make_py_checks(
    name = "timing_tests",
    test_srcs = [
        "test/test_perf_counters.py",
        "test/test_timedscope.py",
    ],
    deps = [
        ":timing",
    ],
)
