load("//build_support/python:pip_defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

# The plugin configs, one per suite. A plugged-in orchestrator runs inside its low-level policy's own
# entry point (models/vlams, models/external/openpi), so those carry them as data too.
filegroup(
    name = "configs",
    srcs = glob(["configs/**/*.yaml"]),
)

# The entry point of a plugged-in policy server or agent (api/client/plugin): the client library and
# nothing else, so it builds in seconds. scripts/eval.sh picks the right entry point.
py_binary(
    name = "run_model",
    srcs = ["//ladder/components/inference/api/client:run_model.py"],
    data = [":configs", "//ladder/components/inference/models/vlams/configs"],
    main = "//ladder/components/inference/api/client:run_model.py",
    # the client library's own environment (pip-inference.MODULE.bazel maps packages to envs by path)
    pip_env = "inference-client",
    pip_deps = ["numpy", "openai", "pillow"],
    deps = ["//ladder/components/inference/api/client"],
)
