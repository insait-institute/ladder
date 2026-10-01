load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_binary", "py_library")
load("//build_support/python:py_tests.bzl", "make_py_checks")

package(default_visibility = ["//visibility:public"])

PIP_DEPS = [
    "absl-py",
    "augmax",
    "beartype",
    "chex",
    "einops",
    "etils",
    "filelock",
    "flax",
    "fsspec",
    "gcsfs",
    "jax",
    "jaxtyping",
    "ml-collections",
    "ml-dtypes",
    "msgpack",
    "numpy",
    "numpydantic",
    "optax",
    "orbax-checkpoint",
    "pillow",
    "pydantic",
    # Not a test dependency: openpi's models_pytorch/gemma_pytorch.py has a stray top-level
    # `import pytest`, and that module is on the import path.
    "pytest",
    "pyyaml",
    "rich",
    "safetensors",
    "sentencepiece",
    "torch",
    "tqdm",
    "tqdm-loggable",
    "transformers",
    "typing-extensions",
    "tyro",
]

pip_compile(
    name = "pip_compile_inference_models_external_openpi",
    py3_runtime = "@python_3_12//:py3_runtime",
    requirements_in = "//ladder/components/inference/models/external/openpi:pyproject.toml",
    requirements_txt = "//ladder/components/inference/models/external/openpi:requirements.txt.lock",
    tags = ["build"],  # Tag for the test target
)

make_interactive_tools(
    deps = [":openpi"],
)

py_library(
    name = "openpi",
    srcs = ["policy.py"],
    pip_deps = PIP_DEPS,
    deps = [
        "//ladder/components/inference/api/core",
        "//ladder/core/common/logger",
        "//ladder/core/databib",
        "@openpi_client_repo//:openpi_src",
    ],
)

# jax looks for its CUDA libraries beside the plugin wheel, where a venv of one repository per
# wheel does not put them, so the venv's own directories go on the loader path.
_VENV_SITE_PACKAGES = "{}/_run_model.venv/lib/python3.12/site-packages".format(package_name())

_CUDA_LIB_DIRS = [
    "{}/nvidia/{}/lib".format(_VENV_SITE_PACKAGES, lib)
    for lib in [
        "cublas",
        "cuda_cupti",
        "cuda_runtime",
        "cudnn",
        "cufft",
        "cusolver",
        "cusparse",
        "nccl",
        "nvjitlink",
    ]
]

py_binary(
    name = "run_model",
    srcs = ["//ladder/components/inference/api/client:run_model.py"],
    # The robolab arms import the benchmark's shared config chain by relative path at load time,
    # so those yamls have to be in the runfiles too.
    data = glob(["configs/**/*.yaml"]) + ["//ladder/components/inference/models/vlams/configs"],
    env = {
        "LD_LIBRARY_PATH": ":".join(_CUDA_LIB_DIRS),
        # One model per lane and a simulator on the same card, so jax must not take a fixed share
        # of it up front. Growing on demand needs no per-card number, which a fraction does.
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    },
    main = "//ladder/components/inference/api/client:run_model.py",
    pip_deps = PIP_DEPS,
    deps = [
        ":openpi",
        "//ladder/components/inference/api/client",
    ],
)

make_py_checks(
    name = "openpi_tests",
    deps = [":openpi"],
)
