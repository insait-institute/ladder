load("@aspect_bazel_lib//lib:copy_to_directory.bzl", "copy_to_directory")
load("@pip-inference-robots-robot-arena//:requirements.bzl", all_pip_inference_robot_arena = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "py_binary", "py_library", "uv_export")
load("//build_support/python:py_tests.bzl", "make_py_checks", "make_py_test")

package(default_visibility = ["//visibility:public"])

uv_export(
    name = "pip_compile_inference_robot_arena",
    pyproject_toml = "//ladder/components/inference/robots/robot_arena:pyproject.toml",
    requirements_txt = "//ladder/components/inference/robots/robot_arena:requirements.txt.lock",
    tags = ["build"],
    uv_lock = "//ladder/components/inference/robots/robot_arena:uv.lock",
)

make_interactive_tools(
    deps = [":robot_arena"] + all_pip_inference_robot_arena,
)

py_library(
    name = "robot_arena",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    data = [
        "pyproject.toml",
        "requirements.txt.lock",
        ":configs",
        ":robot_arena_data",
    ],
    pip_deps = [
        "genesis-world",
        "numpy",
        "opencv-python-headless",
        "scipy",
        "torch",
    ],
    deps = [
        "//ladder/components/inference/api/core",
        "//ladder/components/inference/api/server",
        "//ladder/core/compute:seeder",
        "//ladder/core/databib",
    ],
)

make_py_checks(
    name = "robot_arena_tests",
    test_srcs = glob(
        ["test/*.py"],
        exclude = [
            "test/test_robot_arena_server.py",
            "test/test_robot_arena_setup.py",
        ],
    ),
    deps = [":robot_arena"],
)

# Configs for RobotArena environments
filegroup(
    name = "configs",
    srcs = glob(["configs/**/*.yaml"]),
)

# RobotArena pre-generated scene data and robot assets
filegroup(
    name = "robot_arena_data",
    srcs = [":robot_arena_data_downloaded"],
)

# Download and extract RobotArena data (scenes, assets, backgrounds, robot URDFs)
copy_to_directory(
    name = "robot_arena_data_downloaded",
    srcs = ["@robot_arena_archive//:robot_arena_data"],
    out = "robot_arena_data",
    include_external_repositories = ["*robot_arena_archive"],
)

py_binary(
    name = "robot_arena_server",
    srcs = ["robot_arena_server.py"],
    deps = [":robot_arena"],
)

make_py_test(
    name = "test_robot_arena_setup",
    srcs = ["test/test_robot_arena_setup.py"],
    deps = [":robot_arena"],
)

# Renders each composition task's reset frame and writes the decomposition-benchmark records
# (instruction + the suite's own subtask list) beside them. GPU node only.
py_binary(
    name = "export_scenes",
    srcs = ["composition/export_scenes.py"],
    data = ["composition/decomp_cache.json"],
    main = "composition/export_scenes.py",
    deps = [":robot_arena"],
)

make_py_test(
    name = "test_robot_arena_server",
    timeout = "long",
    srcs = ["test/test_robot_arena_server.py"],
    tags = ["needs-gpu"],
    deps = [
        ":robot_arena",
        "//ladder/components/inference/scripts:example_client_py",
        "//ladder/core/compute:checksum",
    ],
)
