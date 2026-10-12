import errno
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Generator, Optional, Union


def update_symlink(source_path: Union[str, Path], destination_path: Union[str, Path], cleanup: bool = False):
    """
    Overwrites a possibly existing symlink. If `cleanup` is True, the old symlink target
    is deleted. Assumes relative instead of absolute paths, so that symlinks are still
    valid if copied or mounted on another machine.
    source_path: path of the actual source file
    destination_path: path of the symlink
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    assert source_path.exists(), f"{source_path} doesn't exist"

    # Delete old destination
    if cleanup and destination_path.is_symlink():
        previous_source_path = destination_path.resolve()
        delete_path(previous_source_path)

    # Update the symlink
    common_prefix = os.path.commonpath([source_path, destination_path])
    source_relative_path = source_path.relative_to(common_prefix)

    if destination_path.is_symlink():
        destination_path.unlink()

    destination_path.symlink_to(source_relative_path)


def delete_path(path: Path | str):
    """Removes file, symlink or directory"""
    path = Path(path)
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def get_filepaths(directory: str, relative: bool = False) -> Generator[str, None, None]:
    """Generates the filenames in a directory by walking the tree."""
    for root, _, files in os.walk(directory):
        for filename in files:
            absolute_path = os.path.join(root, filename)
            if relative:
                yield os.path.relpath(absolute_path, directory)
            else:
                yield absolute_path


def get_immediate_subdirectories(directory: str) -> Generator[str, None, None]:
    """Returns the list of directories directly under the given directory"""
    return (name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)))


def maybe_make_dir(directory: str) -> None:
    """makes directory (recursively) if it doesn't already exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_sandbox_safe_tmpdir(prefix: str = '', suffix: str = '') -> str:
    return tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=os.environ.get('TEST_TMPDIR', None))


def create_sandbox_safe_tmpdir_with_cleanup(
    prefix: str = '', suffix: str = ''
) -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix=prefix, suffix=suffix, dir=os.environ.get('TEST_TMPDIR', None))


def create_tempdir(test: Optional[unittest.TestCase] = None, prefix_path: Optional[str] = None) -> str:
    if 'TEST_TMPDIR' in os.environ:
        # in the case of testing, we do not want to write data outside of the sandbox, so let's patch the path to write in the right place
        prefix_path = os.path.join(os.environ.get('TEST_TMPDIR', ''), prefix_path or '')

    if prefix_path is not None:
        os.makedirs(prefix_path, exist_ok=True)

    path = tempfile.mkdtemp(dir=prefix_path)

    if test is not None:
        assert isinstance(test, unittest.TestCase), type(test)
        test.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))

    return path


def hardlink_or_copy_file(source_path: str | Path, output_path: Path) -> None:
    """Hardlink a file, falling back to a copy when it would cross a filesystem boundary.

    Comparing ``st_dev`` is not reliable: filesystems such as CephFS, btrfs subvolumes or
    bind mounts can share a device id yet still reject hardlinks across subtrees with EXDEV.
    Attempt the hardlink and only copy when the kernel reports it is not possible.
    """
    source_path = Path(source_path)

    try:
        os.link(source_path, output_path)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source_path, output_path)
