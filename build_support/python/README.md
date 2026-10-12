# ruff

Docs:
- Homepage: https://github.com/astral-sh/ruff
- List of all rules: https://docs.astral.sh/ruff/rules
- List of config options: https://docs.astral.sh/ruff/settings/
- Pylint rules implemented: https://github.com/astral-sh/ruff/issues/970
- VSCode extension: https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff

Running manually
```bash
bazel run //build_support/python:ruff -- check --help
```

Inserting ignore for particular error code:
```bash
bazel run //build_support/python:ruff -- check --add-noqa --select RULE_CODE,RULE_CODE FILES
# For example
bazel run //build_support/python:ruff -- check --add-noqa --select N801 $PWD/ladder/components/inference
```


Running manually on the entire repo:
```bash
bazel run //build_support/python:ruff -- check --config build_support/python/ruff.toml $PWD/ladder/
```

# pylint / flake8
Disabled in favor of ruff, which should provide better coverage and faster execution

- If VSCode `flake8` extension is installed, you need to switch to the pre-release version to disable
it for the workspace (as the `enable` option is not available in the release version)

# basedpyright

Docs:
- Command line: https://docs.basedpyright.com/latest/configuration/command-line/
- Configuration: https://docs.basedpyright.com/latest/configuration/config-files/
- Sample config: https://docs.basedpyright.com/latest/configuration/config-files/#sample-config-file
- Stubs: https://docs.basedpyright.com/latest/usage/type-stubs/
