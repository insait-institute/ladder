load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "template",
    srcs = [
        "__init__.py",
        "template.py",
        "template_dataclass.py",
        "template_utils.py",
    ],
    deps = [
        "//ladder/core/databib/utils:classproperty",
        "//ladder/core/databib/utils:typing_utils",
    ],
)

make_py_checks(
    name = "template_tests",
    min_coverage_pct = 90,
    test_srcs = [
        "test/test_template.py",
        "test/test_template_dataclass.py",
        "test/test_template_utils.py",
    ],
    with_coverage = True,
    deps = [":template"],
)
