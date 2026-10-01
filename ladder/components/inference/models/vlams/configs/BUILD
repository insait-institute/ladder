load("//build_support/python:py_tests.bzl", "make_py_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "configs",
    srcs = glob(["**/*.yaml"]),
)

make_py_test(
    name = "test_configs_parse",
    srcs = ["test/test_configs_parse.py"],
    data = [":configs"],
    deps = [
        "//ladder/components/inference/models/vlams",
        "//ladder/core/common/logger",
        "//ladder/core/compute:seeder",
    ],
)
