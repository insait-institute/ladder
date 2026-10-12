load("//build_support/python:pip_defs.bzl", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks", "make_py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "seeder",
    srcs = ["seeder.py"],
    # NOTE: In principle this code also requires torch. However, we strictly want to avoid having
    # torch as a dependency in ladder/core. Instead, the downstream package which uses this code
    # is expected to add torch as a dependency
    pip_deps = ["numpy"],
)

py_library(
    name = "checksum",
    testonly = True,
    srcs = ["checksum.py"],
    # NOTE: In principle this code also requires torch. However, we strictly want to avoid having
    # torch as a dependency in ladder/core. Instead, the downstream package which uses this code
    # is expected to add torch as a dependency
    pip_deps = [
        "numpy",
        "pillow",
        "xxhash",
    ],
    deps = ["//ladder/core/databib"],
)

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    # NOTE: In principle this code also requires torch. However, we strictly want to avoid having
    # torch as a dependency in ladder/core. Instead, the downstream package which uses this code
    # is expected to add torch as a dependency
    pip_deps = ["numpy"],
)

py_library(
    name = "collate",
    srcs = ["collate.py"],
    # NOTE: In principle this code also requires torch. However, we strictly want to avoid having
    # torch as a dependency in ladder/core. Instead, the downstream package which uses this code
    # is expected to add torch as a dependency
    pip_deps = [
        "numpy",
    ],
    deps = ["//ladder/core/databib"],
)

make_py_checks(
    name = "seeder_tests",
    deps = [":seeder"],
)

make_py_test(
    name = "test_seeder",
    srcs = ["test/test_seeder.py"],
    pip_deps = ["torch"],
    deps = [":seeder"],
)

make_py_test(
    name = "test_seeder_gpu",
    srcs = ["test/test_seeder_gpu.py"],
    pip_deps = ["torch"],
    tags = ["needs-gpu"],
    deps = [":seeder"],
)

make_py_checks(
    name = "checksum_tests",
    deps = [":checksum"],
)

make_py_checks(
    name = "tensor_utils_tests",
    pip_deps = [
        "numpy",
        "torch",
    ],
    test_srcs = ["test/test_tensor_utils.py"],
    deps = [":tensor_utils"],
)
