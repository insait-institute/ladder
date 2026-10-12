load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_test")

package(default_visibility = ["//visibility:public"])

exports_files(["target_launcher.py"])

py_library(
    name = "env_resolver",
    srcs = glob(["*.py"]),
    pip_deps = ["PyYAML"],
    pip_env = "inference-client",
    deps = [
        "//ladder/core/common/logger",
        "//ladder/core/databib/config",
        "@rules_python//python/runfiles",
    ],
)

make_py_test(
    name = "test_env_dispatch",
    srcs = ["test/test_env_dispatch.py"],
    pip_env = "inference-client",
    deps = [
        ":env_resolver",
        "//ladder/core/common:path_utils",
    ],
)

make_py_test(
    name = "test_env_registry",
    srcs = ["test/test_env_registry.py"],
    pip_env = "inference-client",
    deps = [
        ":env_resolver",
        "//ladder/core/common:platform",
    ],
)
