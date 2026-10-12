load("@rules_uv//uv:venv.bzl", "create_venv")
load("//build_support/python:pip_defs.bzl", "determine_env", "env_python_version", "py_binary")

def make_interactive_tools(
    deps = [],
    requirements_txt = "requirements.txt.lock",
    visibility = None,
    destination_folder = None,
    tags = [],
):
    """
    Creates venv, ipython3, and jupyter targets for a pip environment.

    Args:
        deps: List of dependencies (including pip dependencies) for the interactive tools.
        requirements_txt: Label to the requirements.txt.lock file. Defaults to "requirements.txt.lock".
        visibility: Visibility of the targets.
        destination_folder: Where to create the venv. Defaults to "<package>/venv".
        tags: Tags for the ipython3/jupyter binaries. Pass ["manual"] when the environment's
            dependencies are large enough that `//...` should not build them.
    """

    if visibility == None:
        visibility = ["//visibility:public"]

    # `create_venv` otherwise builds the venv with the default interpreter, which cannot install a
    # lock resolved for a different one.
    python_version = env_python_version(determine_env(native.package_name(), None))
    py3_runtime = None
    if python_version:
        py3_runtime = "@python_{}//:py3_runtime".format(python_version.replace(".", "_"))

    create_venv(
        name = "venv",
        requirements_txt = requirements_txt,
        destination_folder = destination_folder or native.package_name() + "/venv",
        uv_args = ["--no-deps"],
        py3_runtime = py3_runtime,
    )

    py_binary(
        name = "ipython3",
        srcs = ["//tools:ipython3.py"],
        deps = deps,
        tags = tags,
        visibility = visibility,
    )

    native.alias(
        name = "ipython",
        actual = "ipython3",
        # An untagged alias would drag ipython3 back into `//...` regardless of its own tags.
        tags = tags,
    )

