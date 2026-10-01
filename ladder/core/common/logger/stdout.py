import contextlib
import os
import sys


class DummyFile:
    def write(self, x):
        pass


@contextlib.contextmanager
def no_stdout_within_functions():
    """
    Silences std output (but not error) within this context manager,
    even when calling external functions and processes
    """
    # open fds
    null_fds = os.open(os.devnull, os.O_RDWR)
    # save the current file descriptors to a tuple
    save = os.dup(1)
    # put /dev/null fds on 1 (stout) but not 2 (stderr)
    os.dup2(null_fds, 1)

    # *** run the function ***
    yield

    # restore file descriptors so I can print the results
    os.dup2(save, 1)
    # close the temporary fds
    os.close(null_fds)


@contextlib.contextmanager
def no_stdout():
    """silences output within this context manager"""
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
