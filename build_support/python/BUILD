load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@pip-build//:requirements.bzl", all_pip_build = "all_requirements")
load("@pip_env_config//:defs.bzl", "ALL_ENVS")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "pip_compile_all", "py_library")

package(default_visibility = ["//visibility:public"])

string_flag(
    name = "pip_env",
    build_setting_default = "build",
    values = ALL_ENVS,
)

# -----------------------------------------------------------------------------
# Configuration for dynamic pip dependency resolution
# -----------------------------------------------------------------------------

# Generate config settings
[
    config_setting(
        name = "env_" + env.replace("-", "_"),
        flag_values = {":pip_env": env},
    )
    for env in ALL_ENVS
]

pip_compile(
    name = "pip_compile_build_support",
    requirements_in = "//build_support/python:pyproject.toml",
    requirements_txt = "//build_support/python:requirements.txt.lock",
    tags = ["build"],  # Tag for the test target
)

# Target for compiling all pip environments in the correct order
pip_compile_all(name = "run_pip_compile_all")

exports_files([
    "pytest_debug_proxy.py",
    "pytest_proxy.py",
    "pyproject.toml",
    "ruff.toml",
    ".coveragerc",
])

py_library(
    name = "python",
    srcs = glob(["*.py"]),
    pip_deps = [
        "ipdb",
        "pytest",
        "pytest-mock",
    ],
)

py_library(
    name = "pdb_deps",
    pip_deps = [
        "ipdb",
        "pudb",
    ],
)

sh_binary(
    name = "ruff",
    srcs = ["@ruff-linux//:ruff-x86_64-unknown-linux-gnu/ruff"],
    data = [":ruff.toml"],
    tags = ["linter"],
)

make_interactive_tools(
    deps = [":python"] + all_pip_build,
)
