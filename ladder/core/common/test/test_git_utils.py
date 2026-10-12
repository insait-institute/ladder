import os
import unittest
import unittest.mock as mock
from typing import Callable

import git

from ladder.core.common.git_utils import get_git_branch, get_git_diff, get_git_hash
from ladder.core.common.path_utils import create_tempdir


class GitUtilTest(unittest.TestCase):
    def setUp(self):
        self.output_path = create_tempdir(self)

    def _assert_get_git_file(
        self,
        git_filepath: str,
        file_content: str,
        getter: Callable[[], str],
    ):
        """
        1. Change directory to a temporary location
        2. Create a 'ladder/core/common/docker/{git_filename}' file with `file_content`
        3. Assert that `getter()` returns `file_content`
        The reason -> don't overwrite any actual files on disk and isolate the changes in a temp dir
        """
        filepath = os.path.join(self.output_path, git_filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(file_content)

        prev_cwd = os.getcwd()
        try:
            os.chdir(self.output_path)
            self.assertEqual(getter(), file_content)
        finally:
            os.chdir(prev_cwd)

    @mock.patch('ladder.core.common.git_utils.is_running_in_docker', lambda: True)
    def test_get_git_hash_in_docker(self):
        self._assert_get_git_file(
            git_filepath='ladder/core/common/docker/git.hash',
            file_content='7f38147f39f97f6f2cc4b5b05ae83ec91ce4f401',
            getter=get_git_hash,
        )

    @mock.patch('ladder.core.common.git_utils.is_running_in_docker', lambda: True)
    def test_get_git_diff_in_docker(self):
        self._assert_get_git_file(
            git_filepath='ladder/core/common/docker/git.diff',
            file_content='diff --git a/a.py b/a.py\n+print("hello")',
            getter=get_git_diff,
        )

    @mock.patch('ladder.core.common.git_utils.is_running_in_docker', lambda: True)
    def test_get_git_branch_in_docker(self):
        self._assert_get_git_file(
            git_filepath='ladder/core/common/docker/git.branch',
            file_content='main',
            getter=get_git_branch,
        )

    @mock.patch('ladder.core.common.git_utils.is_running_in_docker', lambda: False)
    def test_get_git_hash_in_build_info_staged(self):
        self._assert_get_git_file(
            git_filepath='ladder/core/common/build_info/git.hash',
            file_content='7f38147f39f97f6f2cc4b5b05ae83ec91ce4f401',
            getter=get_git_hash,
        )

    @mock.patch('ladder.core.common.git_utils.is_running_in_docker', lambda: False)
    def test_get_git_diff_in_build_info_staged(self):
        self._assert_get_git_file(
            git_filepath='ladder/core/common/build_info/git.diff',
            file_content='diff --git a/a.py b/a.py\n+print("hello")',
            getter=get_git_diff,
        )

    @mock.patch('ladder.core.common.git_utils.is_running_in_docker', lambda: False)
    def test_get_git_branch_in_build_info_staged(self):
        self._assert_get_git_file(
            git_filepath='ladder/core/common/build_info/git.branch',
            file_content='main',
            getter=get_git_branch,
        )

    @mock.patch('ladder.core.common.git_utils.is_running_in_docker', lambda: False)
    def test_get_git_hash_not_running_in_docker(self):
        self.assertRaises(git.exc.InvalidGitRepositoryError, get_git_hash)


if __name__ == "__main__":
    unittest.main()
