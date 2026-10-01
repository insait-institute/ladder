import os

import git

from ladder.core.common.logger import no_stdout_within_functions
from ladder.core.common.platform import REPO_ROOT, is_running_in_docker, is_test_env


def _get_git_file_if_in_docker(git_filename: str) -> str:
    # git meta files will be generated in this directory when docker image is created
    filename = os.path.join('ladder/core/common/docker/', git_filename)

    with open(filename, 'r') as f:
        return f.read().rstrip('\n')


def _get_git_file_if_staged(git_filename: str) -> str:
    # git meta files will be generated in this directory when ssubmit.sh is used
    filename = os.path.join('ladder/core/common/build_info/', git_filename)

    with open(filename, 'r') as f:
        return f.read().rstrip('\n')


def get_git_diff() -> str:
    """Returns git diff relative to current commit"""
    if is_running_in_docker():
        return _get_git_file_if_in_docker('git.diff')

    try:
        return _get_git_file_if_staged('git.diff')
    except FileNotFoundError:
        pass

    try:
        with no_stdout_within_functions():
            repo = git.Repo()
            t = repo.head.commit.tree
            return repo.git.diff(t, color='always') + '\n'
    except git.exc.InvalidGitRepositoryError:
        raise git.exc.InvalidGitRepositoryError(
            "This needs to be run from the root of the repository as the working directory. "
            "If this function is being run in a test, remember to mock it."
        )


def get_git_hash() -> str:
    """Returns the git commit hash"""
    if is_running_in_docker():
        return _get_git_file_if_in_docker('git.hash')

    try:
        return _get_git_file_if_staged('git.hash')
    except FileNotFoundError:
        pass

    try:
        with no_stdout_within_functions():
            repo = git.Repo()
            return repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        raise git.exc.InvalidGitRepositoryError(
            "This needs to be run from the root of the repository as the working directory. "
            "If this function is being run in a test, remember to mock it."
        )


def get_git_branch() -> str:
    """Returns the git branch"""
    if is_running_in_docker():
        return _get_git_file_if_in_docker('git.branch')

    try:
        return _get_git_file_if_staged('git.branch')
    except FileNotFoundError:
        pass

    try:
        with no_stdout_within_functions():
            repo = git.Repo()
            return repo.active_branch.name
    except git.exc.InvalidGitRepositoryError:
        raise git.exc.InvalidGitRepositoryError(
            "This needs to be run from the root of the repository as the working directory. "
            "If this function is being run in a test, remember to mock it."
        )
    except TypeError:
        # When the HEAD is detached, accessing repo.active_branch raises TypeError
        return f"(HEAD detached at {get_git_hash()})"


def is_root_git_repo(path: str) -> bool:
    try:
        _ = git.Repo(path).git_dir  # noqa: F841
        return True
    except git.exc.InvalidGitRepositoryError:
        return os.path.samefile(path, REPO_ROOT)


def ensure_in_git_repo():
    """
    Change the current working directory to the root of the git repository.
    Otherwise raises `git.exc.InvalidGitRepositoryError`.
    If running in test env, nothing changes
    """
    if is_test_env():
        return

    if not is_root_git_repo(os.getcwd()):
        os.chdir(REPO_ROOT)

    cwd_path = os.getcwd()
    if not is_root_git_repo(cwd_path):
        raise git.exc.InvalidGitRepositoryError(
            f"Current working directory {cwd_path} not a git repo. "
            f"Please set REPO_ROOT environment variable to navigate to the git repo root."
        )
