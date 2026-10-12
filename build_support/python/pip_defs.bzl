load("@rules_python//python:defs.bzl", _py_binary = "py_binary", _py_test = "py_test", _py_library = "py_library")
load("@rules_uv//uv:pip.bzl", _pip_compile = "pip_compile")
load("@rules_uv//uv:uv_export.bzl", _uv_export = "uv_export")
load("@pip_env_config//:defs.bzl", "ENV_CONFIGS", "ENV_DERIVATIVES", "ENV_PATHS", "PYPROJECT_DEPS", "ENV_COMPILE_ORDER", "ENV_PYTHON_VERSION")

# -----------------------------------------------------------------------------
# Dynamic Dependency Resolution Logic
# -----------------------------------------------------------------------------


def determine_env(pkg, override_env):
    """
    Determines the pip environment to use based on the package path or user override.
    """
    if override_env:
        return override_env

    # Check for specific path prefixes defined in MODULE files
    for pkg_prefix, env_name in ENV_PATHS.items():
        if pkg.startswith(pkg_prefix):
            return env_name

    # Default to build if no other match
    return "build"

def env_python_version(env):
    """
    The interpreter this env is pinned to, or None when it runs on the repo default.

    Only envs that declared a non-default `python_version` are in ENV_PYTHON_VERSION, so
    membership is the whole predicate. py_binary/py_test pass the result straight to
    rules_python's `python_version` attr, which reconfigures the target and its entire dep subtree
    via the same rule transition that already carries `pip_env` -- so `bazel run`/`bazel test`
    pick the right interpreter with no --python_version on the command line.
    """
    return ENV_PYTHON_VERSION.get(env)

def _incompatible_at_default_pip_env():
    """
    Marks a target unbuildable while `pip_env` still has its default value.

    Bazel skips a target whose `target_compatible_with` is unsatisfied -- silently under a wildcard
    like `//...`, with an error if it is named explicitly -- and, crucially, does not analyse its
    dependencies. That last part is the point here.

    A py_library cannot transition itself, so it is analysed in whichever configuration reaches it,
    and for a library in a pinned env three of those occur and must behave differently:

    * its own env's py_binary/py_test, which moved `pip_env` and the interpreter together ->
      compatible, and the env's cp3XX-only pip hub resolves;
    * `pip_env=build`, the flag default, which is what `//...` sees when it names this library as a
      top-level target in its own right -> incompatible, so bazel skips it rather than trying (and
      failing) to resolve a pip hub that has no wheels for the default interpreter;
    * any other env -> left compatible on purpose. Analysis proceeds and the pip hub reports "No
      matching wheel for current configuration's Python version", which is what we want: depending
      on pinned-env code from another env cannot work, and marking it incompatible here would
      silently drop the offending target from the build rather than report it.
    """
    return select({
        ENV_CONFIGS["build"]["label"]: ["@platforms//:incompatible"],
        "//conditions:default": [],
    })

def make_pip_deps(names, pip_env = None, terminal = False):
    """
    Selects the correct pip dependencies based on the configuration setting.
    Dynamically checks which environments contain the requested packages.

    Args:
        names: A list of pip package names (e.g., ["numpy", "pytest"]).
        pip_env: Optional. Explicitly set the pip environment (e.g. "vlams", "core").
        terminal: When True, resolve dependencies directly for the detected environment
            without generating select() branches. Used for endpoint py_binary and py_test targets

    Returns:
        A select() statement resolving pip dependencies for valid derived environments,
        or a direct list of labels for the detected environment when terminal=True.
    """
    pkg = native.package_name()
    default_env_name = determine_env(pkg, pip_env)

    if default_env_name not in ENV_CONFIGS:
         fail("Unknown environment '{}' for package '{}'.".format(default_env_name, pkg))

    if terminal:
        default_config = ENV_CONFIGS[default_env_name]
        return [default_config["macro"](name) for name in names]

    select_dict = {}

    # Only generate select branches for environments where this code is valid
    # (i.e., the environment itself and any environment that depends on it).
    # NOTE: It is required that all pip packages from the default env are also present in the derived envs
    for env_name in ENV_DERIVATIVES[default_env_name]:
        config = ENV_CONFIGS[env_name]
        select_dict[config["label"]] = [config["macro"](name) for name in names]

    # Ensure the default condition handles the base environment
    if "//conditions:default" not in select_dict:
        default_config = ENV_CONFIGS[default_env_name]
        select_dict["//conditions:default"] = [default_config["macro"](name) for name in names]

    return select(select_dict)

# -----------------------------------------------------------------------------
# Wrapper macros around rules_python
# -----------------------------------------------------------------------------

def py_library(name, pip_env = None, deps = [], pip_deps = [], terminal = False, **kwargs):
    """
    Wrapper for py_library for consistency and easy future extension.
    """
    deps = deps + make_pip_deps(pip_deps, pip_env = pip_env, terminal = terminal)

    env = determine_env(native.package_name(), pip_env)

    # A pinned env's pip hub has no wheels for the default interpreter, and a library cannot
    # transition itself onto the right one, so keep it out of the default configuration entirely.
    if env_python_version(env):
        kwargs["target_compatible_with"] = _incompatible_at_default_pip_env()

    _py_library(name = name, deps = deps, **kwargs)


def py_binary(name, pip_env = None, deps = [], pip_deps = [], terminal = True, python_version = None, **kwargs):
    """
    Wrapper for py_binary that automatically selects the pip environment to run via config_settings.
    """
    pkg = native.package_name()
    env = determine_env(pkg, pip_env)

    deps = deps + make_pip_deps(pip_deps, pip_env = pip_env, terminal = terminal)

    # https://rules-python.readthedocs.io/en/latest/api/rules_python/python/private/py_binary_rule.html#py_binary.config_settings
    # Inject config_settings into rules_python.py_binary to allow for transitioning the
    # pip dependencies of the underlying py_libraries and determine the target run environment
    config_settings = kwargs.pop("config_settings", {})
    config_settings["//build_support/python:pip_env"] = env

    # Add debugger packages to every py_binary to allow for seamless debugging
    if type(deps) == list and "//build_support/python:pdb_deps" not in deps:
        deps = deps + ["//build_support/python:pdb_deps"]
    else:
        deps = deps + ["//build_support/python:pdb_deps"]

    if "main" not in kwargs:
        kwargs["main"] = name + ".py"

    _py_binary(
        name = name,
        config_settings = config_settings,
        deps = deps,
        python_version = python_version or env_python_version(env),
        **kwargs
    )


def py_test(name, pip_env = None, deps = [], pip_deps = [], terminal = True, python_version = None, **kwargs):
    """
    Wrapper for py_test that automatically selects the pip environment to run via config_settings.
    """
    pkg = native.package_name()
    env = determine_env(pkg, pip_env)

    deps = deps + make_pip_deps(pip_deps, pip_env = pip_env, terminal = terminal)

    # https://rules-python.readthedocs.io/en/latest/api/rules_python/python/private/py_test_rule.html#py_test.config_settings
    # Inject config_settings into rules_python.py_binary to allow for transitioning the
    # pip dependencies of the underlying py_libraries and determine the target run environment
    config_settings = kwargs.pop("config_settings", {})
    config_settings["//build_support/python:pip_env"] = env

    if "main" not in kwargs:
        kwargs["main"] = name + ".py"

    _py_test(
        name = name,
        config_settings = config_settings,
        deps = deps,
        python_version = python_version or env_python_version(env),
        **kwargs
    )

# -----------------------------------------------------------------------------
# pip_compile and uv_export wrappers
# -----------------------------------------------------------------------------

def _compute_no_emit_packages(pyproject_toml):
    pyproject_toml_deps = PYPROJECT_DEPS[pyproject_toml]

    extra_args = []

    for pyrpoject_dep in pyproject_toml_deps:
        extra_args.extend([
            "--no-emit-package",
            pyrpoject_dep.replace("//", '').replace(":pyproject.toml", "").replace("/", "-").replace("_", "-")
            # _pyproject_toml_path_to_pyproject_package_name(pyrpoject_dep),
        ])
    return extra_args


def pip_compile(name, requirements_in, data = [], extra_args = [], test_timeout = "long", **kwargs):
    pyproject_toml = requirements_in

    # Add all pyproject.toml dependencies to data
    data = data + PYPROJECT_DEPS[requirements_in]

    # Add --no-emit-package for all pyproject.toml dependencies
    extra_args = extra_args + _compute_no_emit_packages(requirements_in)

    # Remove duplicates
    data = list(set(data))

    _pip_compile(
        name = name,
        requirements_in = requirements_in,
        data = data,
        extra_args = extra_args,
        timeout = test_timeout,
        **kwargs
    )

def uv_export(name, pyproject_toml, data = [], extra_args = [], test_timeout = "long", **kwargs):
    # Add all pyproject.toml dependencies to data
    data = data + PYPROJECT_DEPS[pyproject_toml]

    # Add --no-emit-package for all pyproject.toml dependencies
    extra_args = extra_args + _compute_no_emit_packages(pyproject_toml)

    # Remove duplicates
    data = list(set(data))

    _uv_export(
        name = name,
        pyproject_toml = pyproject_toml,
        data = data,
        export_args = extra_args,
        timeout = test_timeout,
        **kwargs
    )

# -----------------------------------------------------------------------------
# pip_compile_all macro
# -----------------------------------------------------------------------------

def pip_compile_all(name, visibility = None, tags = []):
    # NOTE: We invoke bazel within bazel; the outer `bazel run` releases the workspace lock
    # before running this script, so the nested query/build below run fine. The per-env
    # compilations are then executed directly from their runfiles (not via `bazel run`) so
    # they can run in parallel without contending on that single workspace lock.
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'BAZEL="${BAZEL:-bazel}"',
        'if [ -z "${BUILD_WORKSPACE_DIRECTORY:-}" ]; then',
        '  printf "\\n\\033[31mError: BUILD_WORKSPACE_DIRECTORY is not set.\\033[0m\\n"',
        "  exit 1",
        "fi",
        'cd "$BUILD_WORKSPACE_DIRECTORY"',
        "",
        "# Packages in ENV_COMPILE_ORDER: no-dependency envs first, then their dependents.",
        "PACKAGES=(",
    ]

    for pkg in ENV_COMPILE_ORDER.values():
        script_lines.append('  "{}"'.format(pkg))

    script_lines.extend([
        ")",
        "",
        "# Discover the compile target of every package in a single query, scoped to the pip",
        "# package subtrees.",
        "SCOPE=",
        'for PKG in "${PACKAGES[@]}"; do',
        '  if [ -n "$SCOPE" ]; then SCOPE="$SCOPE + ${PKG}/..."; else SCOPE="${PKG}/..."; fi',
        "done",
        "declare -A PKG2TARGET",
        "while IFS= read -r LABEL; do",
        '  [ -z "$LABEL" ] && continue',
        '  PKG2TARGET["${LABEL%:*}"]="$LABEL"',
        'done < <("$BAZEL" query "kind(\\"pip_compile rule\\", $SCOPE) union kind(\\"uv_export rule\\", $SCOPE)" --output=label)',
        "",
        "# Resolve targets in compile order, failing early on a package without a compile rule.",
        "TARGETS=()",
        'for PKG in "${PACKAGES[@]}"; do',
        '  TARGET="${PKG2TARGET[$PKG]:-}"',
        '  if [ -z "$TARGET" ]; then',
        '    printf "\\n\\033[31mError: no pip compile rule found in package \'%s\'.\\033[0m\\n" "$PKG"',
        "    exit 1",
        "  fi",
        '  TARGETS+=("$TARGET")',
        "done",
        "",
        "# Build every runner once (parallel analysis/build inside one bazel invocation).",
        'printf "\\n\\033[35mBuilding %d pip-compile runners ...\\033[0m\\n" "${#TARGETS[@]}"',
        '"$BAZEL" build "${TARGETS[@]}"',
        "",
        'BAZEL_BIN="$("$BAZEL" info bazel-bin)"',
        'LOGDIR="$(mktemp -d)"',
        "trap 'rm -rf \"$LOGDIR\"' EXIT",
        "",
        "# Launch all compilations in parallel. Each runner is executed directly from its",
        "# runfiles tree (cwd = runfiles/_main so its short_path inputs resolve, and",
        "# BUILD_WORKSPACE_DIRECTORY set so the lock is written into the source tree).",
        "PIDS=()",
        'for i in "${!TARGETS[@]}"; do',
        '  TARGET="${TARGETS[$i]}"',
        '  PKG_PATH="${TARGET#//}"; PKG_PATH="${PKG_PATH%:*}"',
        '  EXE="$BAZEL_BIN/$PKG_PATH/${TARGET##*:}"',
        '  RF="$EXE.runfiles"',
        '  printf "\\033[35mCompiling %s ...\\033[0m\\n" "$TARGET" >&2',
        '  ( cd "$RF/_main" && BUILD_WORKSPACE_DIRECTORY="$BUILD_WORKSPACE_DIRECTORY" RUNFILES_DIR="$RF" exec "$EXE" ) \\',
        '    >"$LOGDIR/$i.log" 2>&1 &',
        "  PIDS[$i]=$!",
        "done",
        "",
        "# Wait in ENV_COMPILE_ORDER. The first non-zero wait is the order-earliest failure",
        "# (every earlier env has already been confirmed to succeed): show only that failure,",
        "# kill the still-running jobs, and stop without waiting for the rest.",
        'for i in "${!PIDS[@]}"; do',
        '  if wait "${PIDS[$i]}"; then',
        '    printf "\\033[32mDone %s\\033[0m\\n" "${TARGETS[$i]}" >&2',
        "  else",
        '    printf "\\n\\033[31mError: \'%s\' failed:\\033[0m\\n\\n" "${TARGETS[$i]}" >&2',
        '    cat "$LOGDIR/$i.log" >&2',
        '    for PID in "${PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done',
        "    wait 2>/dev/null || true",
        "    exit 1",
        "  fi",
        "done",
    ])

    script_content = "\n".join(script_lines) + "\n"
    # Escape dollars because genrule cmd interprets '$' placeholders.
    script_content = script_content.replace("$", "$$")

    script_target = name + "_script"
    native.genrule(
        name = script_target,
        outs = [name + "_script.sh"],
        cmd = "cat > $@ <<'EOF'\n{}\nEOF\nchmod +x $@".format(script_content),
        visibility = ["//visibility:private"],
        tags = tags,
    )

    native.sh_binary(
        name = name,
        srcs = [":" + script_target],
        visibility = visibility,
        tags = tags,
    )
