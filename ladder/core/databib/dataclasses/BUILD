load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "dataclasses",
    srcs = [
        "__init__.py",
        "dataclass.py",
        "dataclass_dict.py",
        "dataclass_interface.py",
        "dataclass_list.py",
        "dataclass_utils.py",
    ],
    pip_deps = [
        "frozendict",
        "numpy",
    ],
    deps = [
        "//ladder/core/databib/template",
        "//ladder/core/databib/utils:classproperty",
        "//ladder/core/databib/utils:typing_utils",
    ],
)

make_py_checks(
    name = "dataclasses_tests",
    min_coverage_pct = 85,
    pip_deps = ["torch"],
    test_srcs = [
        "test/helper.py",
        "test/test_dataclass.py",
        "test/test_dataclass_dict.py",
        "test/test_dataclass_list.py",
        "test/test_dataclass_type_checks.py",
        "test/test_dataclass_utils.py",
    ],
    with_coverage = True,
    deps = [":dataclasses"],
)
