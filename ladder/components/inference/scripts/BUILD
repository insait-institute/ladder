load("//build_support/python:pip_defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])





py_library(
    name = "example_client_py",
    srcs = ["example_client.py"],
    deps = ["//ladder/components/inference/api/core"],
)


py_library(
    name = "mock_server_py",
    srcs = ["mock_server.py"],
    pip_deps = [
        "numpy",
    ],
    pip_env = "inference-core",
    deps = [
        "//ladder/components/inference/api/core",
        "//ladder/core/databib",
    ],
)

py_binary(
    name = "example_client",
    srcs = ["example_client.py"],
    main = "example_client.py",
    pip_deps = [
        "numpy",
    ],
    pip_env = "inference-client",
    deps = [
        ":example_client_py",
        "//ladder/core/databib",
    ],
)
