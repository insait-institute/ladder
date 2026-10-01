import os
import re
import sys
import tempfile
from typing import Tuple

from pytest import console_main


def get_test_dirs() -> Tuple[str, str]:
    """
    Bazel sets the following environment variables with locations
    where tests can store temporary data and undeclared outputs
    """
    tmp_dir = os.environ.get('TEST_TMPDIR', tempfile.mkdtemp())
    undeclared_outputs_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp())

    return (tmp_dir, undeclared_outputs_dir)


def patch_argv(undeclared_outputs_dir: str) -> None:
    for idx, arg in enumerate(sys.argv):
        if '--cov-report=html' in arg:
            coverage_path = os.path.join(undeclared_outputs_dir, 'coverage-html-report')
            sys.argv[idx] = f'--cov-report=html:{coverage_path}'
            continue


def patch_environment(tmp_dir: str, undeclared_outputs_dir: str) -> None:
    # COVERAGE_FILE is read by `coverage` to figure out where to store output.
    os.environ["COVERAGE_FILE"] = os.path.join(undeclared_outputs_dir, '.coverage')

    # TORCH_HOME is read by `pytorch` to figure out where to store the temporary data.
    os.environ["TORCH_HOME"] = os.path.join(tmp_dir, 'torch-home')


if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])

    test_tmp_dir, test_undeclared_outputs_dir = get_test_dirs()

    patch_argv(test_undeclared_outputs_dir)
    patch_environment(test_tmp_dir, test_undeclared_outputs_dir)

    exit_code = console_main()

    sys.exit(exit_code)
