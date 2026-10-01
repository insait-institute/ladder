# The repo-wide interpreter, declared by the `python.toolchain(is_default = True, python_version = ...)`
# call in MODULE.bazel. An env naming this version is not pinned: it needs neither a constraint nor a
# transition, and pinning it would fork the configuration of every target in the repo for nothing.
_DEFAULT_PYTHON_VERSION = "3.10"

def _write_dict(destination, data, var_name):
    destination.append('{var_name} = {{'.format(var_name=var_name))
    for key, value in data.items():
        value_type = type(value)
        if value_type == "list":
            value = '[{}]'.format(", ".join(['"{}"'.format(v) for v in value]))
        elif value_type == "string":
            value = '"{}"'.format(value)
        else:
            fail("Unsupported value type '{}' for key '{}' of value '{}' in variable '{}'".format(value_type, key, value, var_name))
        destination.append(
            '"{key}": {value},'.format(key=key, value=value)
        )
    destination.append('}')
    destination.append('')
    return destination

def _pip_env_impl(ctx):
    envs = {}

    # 1. Collect all environments and their configs
    for mod in ctx.modules:
        for tag in mod.tags.env:
            if tag.name in envs:
                fail("Environment '{}' defined multiple times".format(tag.name))

            envs[tag.name] = {
                "repo": tag.repo or ("pip-" + tag.name),
                "deps": tag.deps,
                "paths": tag.paths,
                "pyproject": tag.pyproject,
                "python_version": tag.python_version,
            }

    env_names = sorted(envs.keys())

    # 2. Build ENV_DEPS - the dependency chain for each environment
    ENV_DEPS = {}  # Map of env name to all envs it depends on (including itself)

    for name in env_names:
        info = envs[name]

        # Determine the dependency chain via BFS
        env_deps = [name]
        queue = []
        queue.extend(info["deps"])
        visited = {name: True}

        for _ in range(len(envs)):  # Unfortunately starlark doesn't have a while loop
            if len(queue) == 0:
                break
            env_name = queue.pop(0)

            # Make sure a valid env name
            if env_name not in envs:
                fail("Environment '{}' depends on unknown environment '{}'".format(name, env_name))

            # If env is already visited, we move on
            if env_name in visited:
                continue
            visited[env_name] = True

            # Add to dependency list
            env_deps.append(env_name)

            # Add the env direct dependencies to the queue
            for dep in envs[env_name]["deps"]:
                if dep not in visited:
                    queue.append(dep)

        ENV_DEPS[name] = env_deps

    # 3. Build ENV_DERIVATIVES - the derivatives of each environment
    ENV_DERIVATIVES = {}  # Map of env name to all envs that depend on it (including itself)

    for name in env_names:
        ENV_DERIVATIVES[name] = [name]
        for other_name in env_names:
            if (other_name != name) and (name in ENV_DEPS[other_name]):
                ENV_DERIVATIVES[name].append(other_name)


    # 4. Build ENV_CONFIGS - contains the pip macro and label for each environment
    ENV_CONFIGS = {}  # Map of env name to its pip macro and config label
    for name in env_names:
        safe_name = name.replace("-", "_")
        ENV_CONFIGS[name] = {
            "macro": "pip_{}".format(safe_name),
            "label": "//build_support/python:env_{}".format(safe_name),
        }

    # 5. Build ENV_PATHS - contains the paths for each environment
    ENV_PATHS = {}  # Map of path to an env name
    for name, info in envs.items():
        for path in info["paths"]:
            # Make sure no other env has defined this path already
            if path in ENV_PATHS:
                fail(
                    "Path '{}' is defined in multiple environments: '{}' and '{}'".format(
                        path, name, ENV_PATHS[path]
                    )
                )
            ENV_PATHS[path] = name

    # Sort the paths from longest to shortest to ensure the most specific match is found first in pip_deps
    sorted_keys = sorted(ENV_PATHS.keys(), key=len, reverse=True)
    ENV_PATHS = {key: ENV_PATHS[key] for key in sorted_keys}

    # 5b. Build ENV_PYTHON_VERSION - only the envs pinned to an interpreter other than the repo
    # default. Deciding that here means membership alone answers "is this env pinned", so pip_defs
    # needs no notion of what the default version is.
    ENV_PYTHON_VERSION = {
        name: envs[name]["python_version"]
        for name in env_names
        if envs[name]["python_version"] and envs[name]["python_version"] != _DEFAULT_PYTHON_VERSION
    }

    # 6. Build PYPROJECT_DEPS - the dependency chain for each pyprojec.toml
    PYPROJECT_DEPS = {}
    for name in env_names:
        pyproject_toml = envs[name]["pyproject"]
        pyproject_toml_deps = []
        for dep_name in ENV_DEPS[name][1:]:  # Skip the env itself (i.e. the pyproject.toml itself)
            pyproject_toml_deps.append(envs[dep_name]["pyproject"])
        PYPROJECT_DEPS[pyproject_toml] = pyproject_toml_deps

    # 7. Build ENV_COMPILE_ORDER - sorted env_name to env_package mapping for pip_compile_all
    ENV_COMPILE_ORDER = {}

    # Sort the environments by their number of deps starting from the shortest dep
    pending_envs = {
        env_name: list(ENV_DEPS[env_name][1:])
        for env_name in sorted(dict(ENV_DEPS), key=lambda x: len(ENV_DEPS[x]))
    }

    for _ in range(len(env_names)):
        ready_env = None
        for env_name, env_deps in pending_envs.items():
            has_pending_dep = False
            for dep_name in env_deps:
                if dep_name in pending_envs:
                    has_pending_dep = True
                    break

            if not has_pending_dep:
                ready_env = env_name
                break

        if ready_env == None:
            fail("Cyclic dependency in pip environments: {}".format(pending_envs.items()))

        ENV_COMPILE_ORDER[ready_env] = envs[ready_env]["pyproject"].replace(":pyproject.toml", "")
        pending_envs.pop(ready_env)

    # 8. Generate the contents for defs.bzl as a list of strings
    file_contents = [
        "# Autogenerated file - do not edit",
        "",
    ]

    # 8.1 Write load statements for each environment's pip requirements
    for env_name in env_names:
        info = envs[env_name]
        safe_name = env_name.replace("-", "_")
        file_contents.append(
            'load("@{repo}//:requirements.bzl", pip_{safe_name}="requirement")'.format(
                repo=info["repo"], safe_name=safe_name
            )
        )

    # 8.2 Write ENV_DEPS
    file_contents = _write_dict(file_contents, ENV_DEPS, "ENV_DEPS")

    # 8.3 Write ENV_DERIVATIVES
    file_contents = _write_dict(file_contents, ENV_DERIVATIVES, "ENV_DERIVATIVES")

    # 8.4 Write ENV_CONFIGS
    file_contents.append('ENV_CONFIGS = {')
    for env_name, env_info in ENV_CONFIGS.items():
        value = '{{"macro": {}, "label": "{}"}}'.format(env_info["macro"], env_info["label"])
        file_contents.append(
            '"{key}": {value},'.format(key=env_name, value=value)
        )
    file_contents.append('}')
    file_contents.append('')

    # 8.5 Write ENV_PATHS
    file_contents = _write_dict(file_contents, ENV_PATHS, "ENV_PATHS")

    # 8.5b Write ENV_PYTHON_VERSION
    file_contents = _write_dict(file_contents, ENV_PYTHON_VERSION, "ENV_PYTHON_VERSION")

    # 8.6 Write PYPROJECT_DEPS
    file_contents = _write_dict(file_contents, PYPROJECT_DEPS, "PYPROJECT_DEPS")

    # 8.7 Write ENV_COMPILE_ORDER
    file_contents = _write_dict(file_contents, ENV_COMPILE_ORDER, "ENV_COMPILE_ORDER")

    # 8.8 Write ALL_ENVS
    file_contents.append('ALL_ENVS = [')
    for env_name in env_names:
        file_contents.append('"{}",'.format(env_name))
    file_contents.append(']')
    file_contents.append('')

    # Write content to repo
    pip_env_config_repo(
        name = "pip_env_config",
        content = "\n".join(file_contents),
    )


pip_env = module_extension(
    implementation = _pip_env_impl,
    tag_classes = {
        "env": tag_class(attrs = {
            "name": attr.string(mandatory = True),
            "repo": attr.string(),
            "deps": attr.string_list(doc = "List of environments this environment depends on"),
            "paths": attr.string_list(doc = "List of package path prefixes that map to this environment"),
            "pyproject": attr.string(mandatory = True, doc = "Label of the env pyproject.toml file"),
            "python_version": attr.string(doc = "Interpreter this env runs on (e.g. \"3.12\"), when it is not the repo default. Must match this env's pip.parse python_version. pip_defs then transitions the env's py_binary/py_test targets onto it and constrains its py_libraries, so no BUILD file needs a py-version select and no command line needs a --python_version flag."),
        }),
    },
)


def _pip_env_config_repo_impl(ctx):
    ctx.file("BUILD", 'exports_files(["defs.bzl"])')
    ctx.file("defs.bzl", ctx.attr.content)


pip_env_config_repo = repository_rule(
    implementation = _pip_env_config_repo_impl,
    attrs = {
        "content": attr.string(),
    },
)
