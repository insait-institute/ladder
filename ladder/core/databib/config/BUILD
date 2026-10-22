load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "config",
    srcs = [
        "__init__.py",
        "auto_config.py",
        "config.py",
        "config_iterables.py",
        "config_parser.py",
        "yaml_util.py",
    ],
    pip_deps = [
        "deepdiff",
        "frozendict",
        "PyYAML",
    ],
    deps = [
        "//ladder/core/databib/dataclasses",
        "//ladder/core/databib/template",
    ],
)

filegroup(
    name = "test_config_files",
    srcs = glob(["test/config_files/**/*.yaml"]),
)

make_py_checks(
    name = "config_tests",
    data = [
        ":test_config_files",
    ],
    min_coverage_pct = 85,
    # backports.strenum is test-only (no py3.12 wheel); keeping it off :config lets py3.12 depend on it.
    pip_deps = [
        "backports.strenum",
        "torch",
    ],
    test_srcs = glob(["test/*.py"]),
    with_coverage = True,
    deps = [
        ":config",
    ],
)
