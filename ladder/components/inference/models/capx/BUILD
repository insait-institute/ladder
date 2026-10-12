load("@pip-inference-models-capx//:requirements.bzl", all_pip_inference_models_capx = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_binary", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

PIP_DEPS = [
    "numpy",
    "pillow",
    "scipy",
    "torch",
    "torchvision",
    "timm",
    "einops",
    "ftfy",
    "regex",
    "huggingface-hub",
    "psutil",
    "iopath",
    "pycocotools",
    "opencv-python-headless",
]

pip_compile(
    name = "pip_compile_inference_models_capx",
    requirements_in = "//ladder/components/inference/models/capx:pyproject.toml",
    requirements_txt = "//ladder/components/inference/models/capx:requirements.txt.lock",
    tags = ["build"],
)

make_interactive_tools(
    deps = [":capx"] + all_pip_inference_models_capx,
)

py_library(
    name = "capx",
    srcs = glob(
        ["**/*.py"],
        exclude = ["**/test/**"],
    ),
    pip_deps = PIP_DEPS,
    deps = [
        "//ladder/components/inference/api/client",
        "//ladder/components/inference/api/core",
        "//ladder/core/common/logger",
        "//ladder/core/databib",
        "@sam3_repo//:sam3_src",
    ],
)

py_binary(
    name = "run_model",
    srcs = ["//ladder/components/inference/api/client:run_model.py"],
    # A capx config imports the benchmark's shared vlams chain by relative path at load time, so
    # both trees have to be in the runfiles.
    data = glob(["configs/**/*.yaml"]) + ["//ladder/components/inference/models/vlams/configs"],
    main = "//ladder/components/inference/api/client:run_model.py",
    pip_deps = PIP_DEPS + ["ipdb"],
    deps = [":capx"],
)

# This package had no checks at all, so a name used but not imported reached the branch. Ruff's
# pyflakes rules catch that; the package ships no unit tests, hence the empty test glob.
make_py_checks(
    name = "capx_tests",
    test_srcs = glob(
        ["**/test/*.py"],
        allow_empty = True,
    ),
    deps = [":capx"],
)
