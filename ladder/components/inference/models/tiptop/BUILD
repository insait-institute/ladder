load("@aspect_bazel_lib//lib:copy_to_directory.bzl", "copy_to_directory")
load("@pip-inference-models-tiptop//:requirements.bzl", all_pip_inference_models_tiptop = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

PIP_DEPS = [
    "einops",
    "filelock",
    "hydra-core",
    "iopath",
    "jaxtyping",
    "msgpack-numpy",
    "numpy",
    "omegaconf",
    "open3d",
    "openai",
    "opencv-python-headless",
    "pillow",
    "pyyaml",
    "pyzmq",
    "requests",
    "rerun-sdk",
    "roma",
    "scikit-learn",
    "scipy",
    "setuptools",
    "torch",
    "torchvision",
    "trimesh",
    "tqdm",
    "warp-lang",
    "yourdfpy",
    "ninja",
    "nvidia-cuda-nvcc-cu12",
]

pip_compile(
    name = "pip_compile_inference_models_tiptop",
    py3_runtime = "@python_3_12//:py3_runtime",
    requirements_in = "//ladder/components/inference/models/tiptop:pyproject.toml",
    requirements_txt = "//ladder/components/inference/models/tiptop:requirements.txt.lock",
    tags = ["build"],
)

make_interactive_tools(
    deps = [":tiptop"] + all_pip_inference_models_tiptop,
)

py_library(
    name = "tiptop",
    srcs = glob(["*.py"]),
    data = [
        ":weights",
        "@m2t2_src//:m2t2_data",
        "@sam2_src//:sam2_configs",
    ],
    pip_deps = PIP_DEPS,
    deps = [
        "//ladder/components/inference/api/client",
        "@curobo_src//:curobo",
        "@cutamp_src//:cutamp",
        "@m2t2_src//:m2t2",
        "@sam2_src//:sam2",
    ],
)

py_binary(
    name = "run_model",
    srcs = ["//ladder/components/inference/api/client:run_model.py"],
    data = glob(["configs/**/*.yaml"]) + [
        ":cuda_home",
        "//ladder/components/inference/models/vlams/configs",
    ],
    main = "//ladder/components/inference/api/client:run_model.py",
    pip_deps = PIP_DEPS + ["ipdb"],
    deps = [":tiptop"],
)

copy_to_directory(
    name = "cuda_home",
    srcs = ["@cuda_home_archive//:files"],
    out = "cuda_home",
    include_external_repositories = ["*cuda_home_archive"],
    replace_prefixes = {"cuda_home/": ""},
    tags = ["manual"],
)

copy_to_directory(
    name = "weights",
    srcs = [
        "@m2t2_weights//file",
        "@sam2_hiera_large_weights//file",
    ],
    out = "weights",
    include_external_repositories = [
        "*m2t2_weights",
        "*sam2_hiera_large_weights",
    ],
    replace_prefixes = {"file/": ""},
    tags = ["manual"],
)
