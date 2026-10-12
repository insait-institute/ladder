load("@pip-inference-models-vlams-v0//:requirements.bzl", all_pip_inference_models_vlams_v0 = "all_requirements")
load("//build_support/python:interactive_tools.bzl", "make_interactive_tools")
load("//build_support/python:pip_defs.bzl", "pip_compile", "py_binary")

package(default_visibility = ["//visibility:public"])

PIP_DEPS = [
    "backports.strenum",
    "numpy",
    "pandas",
    "transformers",
    "torch",
    "torchvision",
    "flash_attn",
    "xformers",
    "timm",
    "roma",
    "sentencepiece",
]

make_interactive_tools(
    deps = ["//ladder/components/inference/models/vlams:vlams_adaptive"] + all_pip_inference_models_vlams_v0,
)

pip_compile(
    name = "pip_compile_inference_models_vlams_envs_v0",
    requirements_in = "//ladder/components/inference/models/vlams/envs/v0:pyproject.toml",
    requirements_txt = "//ladder/components/inference/models/vlams/envs/v0:requirements.txt.lock",
    tags = ["build"],  # Tag for the test target
)

py_binary(
    name = "run_model",
    srcs = ["//ladder/components/inference/api/client:run_model.py"],
    data = [
        "//ladder/components/inference/models/plugin:configs",
        "//ladder/components/inference/models/vlams/configs",
    ],
    main = "//ladder/components/inference/api/client:run_model.py",
    pip_deps = PIP_DEPS + ["ipdb"],
    deps = ["//ladder/components/inference/models/vlams:vlams_adaptive"],
)

py_binary(
    name = "batch_inference_server",
    srcs = ["//ladder/components/inference/api/client:batch_inference/batch_policy_server.py"],
    main = "//ladder/components/inference/api/client:batch_inference/batch_policy_server.py",
    pip_deps = PIP_DEPS,
    deps = ["//ladder/components/inference/models/vlams:vlams_adaptive"],
)

