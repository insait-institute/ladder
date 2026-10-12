"""
Helper functions for defining flake8, pylint, mypy and python tests
"""

load("//build_support/python:pip_defs.bzl", "determine_env", "env_python_version", "make_pip_deps", "py_binary", "py_library", "py_test")


def mypy(
    name,
    deps,
    exclude = [],
    size = "small",
    timeout = "moderate",
    tags = []
):
    proxy_script = "//build_support/python:mypy_proxy.py"
    ini = "//build_support/python:mypy.ini"

    run_dir = native.package_name()

    if run_dir == "":
        run_dir = "."

    run_dir = run_dir.replace("/", ".")

    py_test(
        name = name,
        srcs = [proxy_script],
        main = proxy_script,
        data = [ini],
        deps = deps,
        pip_deps = [
            "mypy",
            "types-decorator",
            "types-frozendict",
            "types-PyYAML",
            "types-python-dateutil",
        ],
        size = size,
        timeout = timeout,
        tags = tags + ["lint", "mypy"],
        args = [
            "--config $(location {})".format(ini),
        ] + ["--exclude {}".format(e) for e in exclude] +
        [
            "-p",
            run_dir,
        ],
    )

def _first_party_srcs_impl(ctx):
    # rules_python exposes a py_library's transitive python sources as the `compilation_outputs`
    # output group, but that group also carries every wheel's site-packages/**/*.py. External repos
    # materialize *beside* _main in the runfiles tree, and ruff runs `check <package>` with _main as
    # its working directory, so it can never read them -- they are ~100k dead symlinks (hundreds of
    # MB on disk, since every symlink target is long enough to cost a full filesystem block) per lint
    # target. Keep only the first-party sources, which is exactly the set ruff looks at.
    # `lib` carries the transition below, and bazel hands a transitioned label over as a
    # single-element list.
    lib = ctx.attr.lib[0]
    return DefaultInfo(files = depset([
        f
        for f in lib[OutputGroupInfo].compilation_outputs.to_list()
        if not f.short_path.startswith("../")
    ]))

_PIP_ENV_FLAG = "//build_support/python:pip_env"
_PYTHON_VERSION_FLAG = "@rules_python//python/config_settings:python_version"

def _lib_env_transition_impl(settings, attr):
    # A library in a pip environment on a non-default python version is target_compatible_with that
    # environment (see pip_defs), so reading it from the default configuration would make this rule
    # -- and the lint test that consumes it -- incompatible, and bazel would skip the lint silently.
    # Move both settings the way that env's py_binary/py_test would, so the library is compatible and
    # its pip hub resolves while we collect its sources. Default-python envs are left untouched, so
    # their lint targets keep exactly the configuration they had.
    if not attr.python_version:
        return settings
    return {
        _PIP_ENV_FLAG: attr.pip_env,
        _PYTHON_VERSION_FLAG: attr.python_version,
    }

_lib_env_transition = transition(
    implementation = _lib_env_transition_impl,
    inputs = [_PIP_ENV_FLAG, _PYTHON_VERSION_FLAG],
    outputs = [_PIP_ENV_FLAG, _PYTHON_VERSION_FLAG],
)

first_party_srcs = rule(
    implementation = _first_party_srcs_impl,
    attrs = {
        "lib": attr.label(mandatory = True, cfg = _lib_env_transition),
        "pip_env": attr.string(),
        "python_version": attr.string(),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def ruff(
    name,
    lib = None,
    srcs = [],
    size = "small",
    tags = [],
    timeout = "short",
    exclude = None,
    target = None,
):
    ruff_toml = "//build_support/python:ruff.toml"

    if target == None:
        run_dir = native.package_name()
        if run_dir == "":
            run_dir = "."
    else:
        run_dir = target

    args = [
        "check",
        "--config=$(location {})".format(ruff_toml),
    ]

    # Overwrite the target python version if the environment is not the default one
    env = determine_env(native.package_name(), None)
    python_version = env_python_version(env)
    if python_version:
        args.append("--target-version=py" + python_version.replace(".", ""))

    args.append(run_dir)

    if exclude:
        args = ["--extend-exclude=" + ",".join(exclude)] + args

    data = [ruff_toml] + srcs

    # Ruff is a static linter, so it needs the sources and nothing else. Depending on `lib` directly would
    # drag the whole python environment into the runfiles tree for no benefit, so take its first-party source
    # files instead. Note this makes non-trivial difference - ~100k+ symlinks decreased to ~100
    if lib:
        srcs_name = "_{}_srcs".format(name)
        first_party_srcs(
            name = srcs_name,
            lib = lib,
            pip_env = env,
            python_version = python_version,
            testonly = True,
        )
        data = data + [":" + srcs_name]

    native.sh_test(
        name = name,
        srcs = ["@ruff-linux//:ruff-x86_64-unknown-linux-gnu/ruff"],
        timeout = timeout,
        data = data,
        size = size,
        tags = tags + ["lint", "ruff"],
        args = args,
    )


def make_py_test(
    name,
    deps,
    pip_deps = [],
    args = [],
    env = {},
    srcs = [],
    data = [],
    tags = [],
    size = "medium",
    timeout = "short",
    with_coverage = False,
    min_coverage_pct = 75,
    flaky = False,
    local = False,
    allow_internet = False,
    pip_env = None,
    **test_kwargs,
):
    """
    Extended helper pytest rule that also defines a debug rule for ipdb and optionally checks for coverage
    Timeout values: https://bazel.build/reference/test-encyclopedia#role-test-runner
    """

    proxy_script = "//build_support/python:pytest_proxy.py"

    test_args = [
        native.package_name(),
        "--allow-unix-socket",  # pytest-socket: allow unix sockets (eg. for testcontainers)
        "--capture=no",  # Don't duplicate logs
        "--show-capture=stderr",  # Print logs BEFORE error. Captures and prints only stderr
        "--durations=5",  # show 5 slowest setup/test durations
        "--durations-min=1.0",  # minimum seconds for slowest durations
        "--no-header",  # skip printing version information on every test
        "--tb=long",  # exhaustive, informative traceback formatting
        "-p no:cacheprovider",  # disable caching
        "-rfE",  # show summary of failed and errored tests at the end
        "-v",  # increase verbosity for diffs
        "-W ignore::DeprecationWarning",  # disable deprecation warnings for clean logs
        "--ignore-glob=*.venv",  # ignore venv directories created by rules_python
    ]
    if not allow_internet:
        test_args += [
            "--allow-hosts=127.0.0.1,172.17.0.1",  # pytest-socket: allow localhost, docker bridge network gateway
            "--disable-socket",  # pytest-socket: no internet access by default during tests
        ]


    pytest_deps = make_pip_deps([
        "pytest",
        "pytest-mock",
        "pytest-socket",
        "pytest-timeout",
        "debugpy",
    ], pip_env = pip_env, terminal = True)
    pytest_deps += make_pip_deps(pip_deps, pip_env = pip_env, terminal = True)
    pytest_tags = ["test", "pytest"] + tags

    cov_args = []
    if with_coverage:
        pytest_tags.append("coverage")
        pytest_deps += make_pip_deps(["pytest-cov"], pip_env = pip_env, terminal = True)
        cov_args += [
            "--cov={}".format(native.package_name().replace("/", ".")),
            "--cov-context=test",
            "--cov-fail-under={}".format(str(min_coverage_pct)),
            "--cov-report=html:/tmp/",
            "--cov-report=term-missing",
            "--no-cov-on-fail",
            # Ignore runfiles directories from coverage
            "--cov-config=$(location //build_support/python:.coveragerc)",
        ]

    py_test(
        name = name,
        srcs = [proxy_script] + srcs,
        main = proxy_script,
        env = env,
        data = ["//:pyproject.toml"] + (["//build_support/python:.coveragerc"] if with_coverage else []) + data,
        deps = pytest_deps + deps,
        size = size,
        timeout = timeout,
        tags = pytest_tags,
        args = test_args + cov_args + args,
        flaky = flaky,
        local = local,
        pip_env = pip_env,
        **test_kwargs,
    )

    # automatically add debug rule for every py_test
    py_test_debug(
        name = name + "_debug",
        deps = pytest_deps + deps,
        srcs = srcs,
        data = data,
        env = env,
        tags = pytest_tags,
        test_args = test_args + args,
        pip_env = pip_env,
    )

def py_test_debug(
    name,
    deps,
    env = {},
    srcs = [],
    data = [],
    tags = [],
    test_args = [],
    pip_env = None,
):
    """Create a test rule that supports ipdb debugging"""

    proxy_script = "//build_support/python:pytest_proxy.py"
    debug_proxy_script = "//build_support/python:pytest_debug_proxy.py"

    py_binary(
        name = name,
        srcs = [proxy_script, debug_proxy_script] + srcs,
        main = proxy_script,
        data = data,
        env = env,
        testonly = True,
        deps = deps,
        tags = ["debug"] + tags,
        args = test_args,
        pip_env = pip_env,
    )

def make_py_checks(
    name,
    deps,
    pip_deps = [],
    srcs = [],
    data = [],
    test_srcs = [],
    # Pytest options
    test_env = {},
    test_flaky = False,
    test_size = "medium",
    test_tags = [],
    test_timeout = "short",
    min_coverage_pct = 75,
    with_coverage = False,
    # Pylint options
    ruff_test = True,
    ruff_test_size = "small",
    ruff_test_timeout = "short",
    exclude_from_ruff_test = [],
    # Mypy options
    mypy_test = False,
    mypy_test_size = "small",
    mypy_test_timeout = "moderate",
    exclude_from_mypy_test = [],
    pylint_test = False,
    flake8_test = False,
):
    """Create separate bazel rules for flake8, pylint, mypy and python tests"""

    py_tests_tags = ["py_tests"]

    # Determine environment
    pkg = native.package_name()
    env = determine_env(pkg, None)

    lib_name = "_{}_py_tests_lib".format(name)

    # Create the base library for tests
    # We use the correct environment here so that deps resolve correctly for linting and testing
    py_library(
        name = lib_name,
        # Add test_srcs and filter overlaps with srcs
        srcs = srcs + [src for src in test_srcs if ruff_test and src not in srcs],
        tags = py_tests_tags,
        deps = deps,
        testonly = True,
        pip_env = env,
        pip_deps = pip_deps,
        terminal = True,
    )

    all_deps = deps + [":{}".format(lib_name)]

    tests = []

    # Ruff test rule
    if ruff_test:
        ruff_test_name = "{}_ruff".format(name)

        # The library already depends on `deps`, so its transitive sources are everything ruff lints.
        ruff(
            name = ruff_test_name,
            lib = ":{}".format(lib_name),
            tags = py_tests_tags,
            size = ruff_test_size,
            timeout = ruff_test_timeout,
            exclude = exclude_from_ruff_test,
        )
        tests = tests + [ruff_test_name]

    # Mypy test rule
    if mypy_test:
        mypy_test_name = "{}_mypy".format(name)
        mypy(
            name = mypy_test_name,
            exclude = exclude_from_mypy_test + test_srcs,
            tags = py_tests_tags,
            deps = all_deps,
            size = mypy_test_size,
            timeout = mypy_test_timeout,
        )
        tests = tests + [mypy_test_name]

    # Python tests rule
    if len(test_srcs) > 0:
        py_test_name = "{}_py_test".format(name)
        make_py_test(
            name = py_test_name,
            srcs = test_srcs,
            data = data,
            tags = py_tests_tags + test_tags,
            env = test_env,
            size = test_size,
            timeout = test_timeout,
            with_coverage = with_coverage,
            min_coverage_pct = min_coverage_pct,
            deps = all_deps,
            pip_deps = pip_deps,
            flaky = test_flaky,
            pip_env = env,
        )
        tests = tests + [py_test_name]

    native.test_suite(
        name = name,
        tests = tests,
    )
