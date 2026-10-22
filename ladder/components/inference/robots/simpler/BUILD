load("@aspect_bazel_lib//lib:copy_to_directory.bzl", "copy_to_directory")
load("@pip-inference-robots-simpler//:requirements.bzl", all_pip_inference_simpler = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_binary", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks", "make_py_test")

package(default_visibility = ["//visibility:public"])

pip_compile(
    name = "pip_compile_inference_simpler",
    requirements_in = "//ladder/components/inference/robots/simpler:pyproject.toml",
    requirements_txt = "//ladder/components/inference/robots/simpler:requirements.txt.lock",
    tags = ["build"],
)

make_interactive_tools(
    deps = [":simpler"] + all_pip_inference_simpler,
)

py_library(
    name = "simpler",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "**/test/**",
            "generate_perturbations.py",
        ],  # generate_perturbations.py is not part of the release
    ),
    data = [
        # The tracked plan cache. At eval time the Decomposer reads it from the source tree (it
        # resolves a repo-relative path against REPO_ROOT); listing it here also puts it in the
        # runfiles, so tests can check the cache covers the registry.
        "composition/decomp_cache.json",
        "pyproject.toml",
        "requirements.txt.lock",
        ":configs",
        ":maniskill_assets",
        ":sapien_physx",
    ],
    pip_deps = [
        "gymnasium",
        "mani-skill",
        "numpy",
        "opencv-python-headless",
        "sapien",
        "scipy",
        "torch",
        "transforms3d",
        "trimesh",
    ],
    deps = [
        "//ladder/components/inference/api/core",
        "//ladder/components/inference/api/server",
        "//ladder/core/databib",
    ],
)

make_py_checks(
    name = "simpler_tests",
    test_srcs = glob(
        ["test/*.py"],
        # test_simpler_server needs a GPU and has its own make_py_test target below.
        exclude = ["test/test_simpler_server.py"],
    ),
    # The composition spec tests build every layout of every spec: ~60 s, on the edge of "short".
    test_timeout = "moderate",
    deps = [":simpler"],
)

# Configs for simpler environments
filegroup(
    name = "configs",
    srcs = glob(["configs/**/*.yaml"]),
)

# Simulator assets for ManiSkill3 (bridge digital twins + widowx robot), bundled into the runfiles
# so env_builder can point MS_ASSET_DIR at them and no runtime download is needed.
filegroup(
    name = "maniskill_assets",
    srcs = [":maniskill_assets_downloaded"],
)

# Download and extract the ManiSkill3 assets (the data/ tree the env reads via MS_ASSET_DIR).
copy_to_directory(
    name = "maniskill_assets_downloaded",
    srcs = [
        "@maniskill_widowx_archive//:maniskill3_files",
        "@simpler_maniskill3_assets_archive//:maniskill3_files",
    ],
    out = "maniskill3",
    include_external_repositories = [
        "*maniskill_widowx_archive",
        "*simpler_maniskill3_assets_archive",
    ],
)

# The vendored SAPIEN PhysX GPU lib, laid out under sapien_physx/ in the runfiles so env_builder
# can seed ~/.sapien/physx/<version> from it (no runtime GitHub download; see env_builder.py).
copy_to_directory(
    name = "sapien_physx",
    srcs = ["@sapien_physx_precompiled//:physx_so"],
    out = "sapien_physx",
    include_external_repositories = ["*sapien_physx_precompiled"],
)

# Renders each composition scene's reset frame and writes the decomposition-benchmark records
# (instruction + the suite's own subtask list) beside them. GPU node only. The sibling exporters on
# the RobotArena and RoboLab suites emit the same shape into their own directory, so the benchmark
# is the concatenation of the three scenes.json files -- there is no cross-suite builder, because
# each suite needs its own simulator.
py_binary(
    name = "export_scenes",
    srcs = ["composition/export_scenes.py"],
    main = "composition/export_scenes.py",
    deps = [":simpler"],
)

py_binary(
    name = "simpler_server",
    srcs = ["simpler_server.py"],
    deps = [":simpler"],
)

# Prints "<env-v0>|<max_episode_steps>" per composition task, so a run can enumerate the
# registry from the registry itself (see README.md).
py_binary(
    name = "composition_manifest",
    srcs = ["composition/envs.py"],
    main = "composition/envs.py",
    deps = [":simpler"],
)

make_py_test(
    name = "test_simpler_server",
    srcs = ["test/test_simpler_server.py"],
    tags = ["needs-gpu"],
    deps = [
        ":simpler",
        "//ladder/components/inference/scripts:example_client_py",
        "//ladder/core/compute:checksum",
        "//ladder/core/compute:seeder",
    ],
)
