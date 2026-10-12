import sys

from ipdb import slaunch_ipdb_on_exception
from pytest import main

from .pytest_proxy import get_test_dirs, patch_argv, patch_environment

if __name__ == '__main__':
    test_tmp_dir, test_undeclared_outputs_dir = get_test_dirs()
    patch_argv(test_undeclared_outputs_dir)
    patch_environment(test_tmp_dir, test_undeclared_outputs_dir)

    with slaunch_ipdb_on_exception():
        sys.exit(main())
