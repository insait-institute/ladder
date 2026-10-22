load("@aspect_bazel_lib//lib:copy_to_directory.bzl", "copy_to_directory")
load("@pip-inference-robots-robolab//:requirements.bzl", all_pip_inference_robots_robolab = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "py_binary", "py_library", "uv_export")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

# The tracked decomposer plan cache, beside the task package
# exactly as on the SimplerEnv and RobotArena composition benchmarks.
filegroup(
    name = "composition_data",
    srcs = glob(["composition/*.json"]),
)

# py3.12 (isaacsim 6 cp312 wheels) is selected automatically: this package's pip env
# (inference-robots-robolab) declares python_version = "3.12", so pip_defs transitions every
# py_binary/py_test here to py3.12 (`bazel run`/`bazel test` need no --python_version flag) and
# constrains the py_libraries with target_compatible_with, so no per-target select is needed.

# Stage the @robolab_assets_archive asset tree into this package as assets/ (next to the entry
# points in runfiles); they set ROBOLAB_ASSET_DIR here since the code-only @robolab_repo ships none.
copy_to_directory(
    name = "robolab_assets_downloaded",
    srcs = ["@robolab_assets_archive//:assets_files"],
    out = "assets",
    include_external_repositories = ["*robolab_assets_archive"],
    replace_prefixes = {"assets/": ""},
    # manual for the same reason as the eval binaries below, which are the only consumers: without
    # it `//...` expands these two directly and downloads the ~1 GB archive on every cold CI run.
    tags = ["manual"],
)

filegroup(
    name = "robolab_assets",
    srcs = [":robolab_assets_downloaded"],
    tags = ["manual"],
)

uv_export(
    name = "pip_compile_inference_robots_robolab",
    common_args = [
        # --fork-strategy is important, otherwise 2 versions of numpy in the output file
        "--fork-strategy",
        "fewest",
    ],
    py3_runtime = "@python_3_12//:py3_runtime",  # env is ==3.12.* (isaacsim wheels)
    pyproject_toml = "//ladder/components/inference/robots/robolab:pyproject.toml",
    requirements_txt = "//ladder/components/inference/robots/robolab:requirements.txt.lock",
    tags = ["build"],
    uv_lock = "//ladder/components/inference/robots/robolab:uv.lock",
)

# The package's own sources, with no third-party environment attached. Split out so a consumer that
# only reads first-party python -- the ruff lint and the Isaac-free spec tests -- does not pull the
# isaacsim wheels into its runfiles: one file in the extscache is named in non-ASCII characters
# (omni.usd's test_unicode_reference_<...>.usda), and the CI runner's non-UTF-8 filesystem locale
# cannot open it, which fails the runfiles tree before any test runs.
py_library(
    name = "robolab_srcs",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    data = [
        "pyproject.toml",
        "requirements.txt.lock",
        # The tracked plan cache. At eval time the Decomposer reads it from the source tree (it
        # resolves a repo-relative path against REPO_ROOT); listing it here also puts it in the
        # runfiles, so tests can check the cache covers the registry.
        ":composition_data",
    ],
    deps = ["//ladder/components/inference/api/core"],
)

py_library(
    name = "robolab",
    # NOTE: isaacsim lists some of its core packages as extra dependencies, which don't get
    # included by default; all_requirements makes sure every pip package is a dependency.
    deps = [
        ":robolab_srcs",
        "@openpi_client_repo//:openpi_client",
        "@robolab_repo//:robolab_src",
    ] + all_pip_inference_robots_robolab,
)

# Renders one reset frame per composition scene plus the scene records the decomposition benchmark
# is scored against. Manual: it boots Isaac and needs the ~1 GB asset tree.
py_binary(
    name = "export_scenes",
    srcs = ["composition/export_scenes.py"],
    data = [
        ":composition_data",  # the plan cache the scene records take their ground truth from
        ":robolab_assets",
    ],
    main = "composition/export_scenes.py",
    tags = ["manual"],  # same rationale as env_server
    deps = [":robolab"],
)

# Post-hoc integrity gate for a finished eval run: flags episodes whose scene stopped being
# rendered. env_server refuses to start an episode on a full card, so this is the backstop for
# anything that reaches the same end state by another route. Manual: it shells out to ffmpeg over
# a run's recorded videos.
py_binary(
    name = "check_scene_rendered",
    srcs = ["composition/check_scene_rendered.py"],
    main = "composition/check_scene_rendered.py",
    tags = ["manual"],
    deps = [":robolab"],
)

# Serves RoboLab envs over the RobotServer ZMQ protocol to the standard run_model +
# SimplerEpisodeRunner stack, which is how scripts/vlam_robolab.sh evaluates.
py_binary(
    name = "env_server",
    srcs = ["env_server.py"],
    data = [":robolab_assets"],
    main = "env_server.py",
    # manual: keep `//...` wildcards (CI's `bazel test //...`) from materializing the multi-GB
    # isaacsim env + asset tree; the launcher builds these by explicit label.
    tags = ["manual"],
    deps = [
        ":robolab",
        "//ladder/components/inference/api/core",
        "//ladder/core/common:platform",
    ],
)

# Maps an eval run to RoboLab's episode_results.jsonl (dashboard compatibility).
py_binary(
    name = "export_robolab_results",
    srcs = ["export_robolab_results.py"],
    main = "export_robolab_results.py",
    pip_deps = ["pyyaml"],
)

# Lint only. :robolab_srcs and not :robolab so ruff reads the package sources without
# materialising the isaacsim wheel set.
make_py_checks(
    name = "robolab_tests",
    test_srcs = glob(["test/*.py"]),
    deps = [":robolab_srcs"],
)

# venv picks up py3.12 from this env; ipython3/jupyter are manual because they carry the whole
# isaacsim dependency set.
make_interactive_tools(
    tags = ["manual"],
    deps = [":robolab"],
)
