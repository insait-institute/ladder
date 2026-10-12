load("@pip-dashboard//:requirements.bzl", all_pip_dashboard = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_binary", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

pip_compile(
    name = "pip_compile_dashboard",
    requirements_in = "//ladder/components/dashboard:pyproject.toml",
    requirements_txt = "//ladder/components/dashboard:requirements.txt.lock",
    tags = ["build"],
)

make_interactive_tools(
    deps = [":dashboard"] + all_pip_dashboard,
)

py_library(
    name = "dashboard",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "app.py",
            "fetch.py",
        ],
    ),
    data = [
        "pyproject.toml",
        "requirements.txt.lock",
    ],
    pip_deps = [
        "PyYAML",
    ],
    deps = ["//ladder/components/inference/api/client"],
)

py_binary(
    name = "serve",
    srcs = ["app.py"],
    data = glob(["static/*"]),
    main = "app.py",
    pip_deps = [
        "fastapi",
        "uvicorn",
    ],
    deps = [":dashboard"],
)

py_binary(
    name = "fetch",
    srcs = ["fetch.py"],
    main = "fetch.py",
    pip_deps = ["huggingface-hub"],
    deps = [":dashboard"],
)

make_py_checks(
    name = "dashboard_checks",
    srcs = [
        "app.py",
        "fetch.py",
    ],
    deps = [":dashboard"],
)
