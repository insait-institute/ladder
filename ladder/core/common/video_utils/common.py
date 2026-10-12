import contextlib
import multiprocessing
import os
import threading
from typing import Iterator, Optional, Set

import tqdm

# Bookkeeping for the lines occupied by concurrent tqdm progress bars (see tqdm_position). We track
# the line slots that are currently in use so a new bar can claim the lowest free one and release it
# when it finishes. Multiprocessing behaviour:
# - spawn: each process re-imports this module, so the state starts empty and independent per process.
# - fork: the child inherits a copy of the parent's state. os.register_at_fork clears it so the child
#   starts fresh instead of sharing (and racing on) slots with the parent.
_TQDM_POSITION_LOCK = threading.Lock()
_ACTIVE_TQDM_SLOTS: Set[int] = set()


def _reset_tqdm_state() -> None:
    global _TQDM_POSITION_LOCK  # noqa: PLW0603
    global _ACTIVE_TQDM_SLOTS  # noqa: PLW0603

    _TQDM_POSITION_LOCK = threading.Lock()
    _ACTIVE_TQDM_SLOTS = set()


os.register_at_fork(after_in_child=_reset_tqdm_state)

# Each process writes its bars into its own band of the terminal so that bars from different
# processes sharing one terminal do not collide. Within a band, bars are packed contiguously from
# the top and freed as they close.
_PROCESS_TQDM_POSITION_STRIDE = 256


def _parse_trailing_integer(value: str) -> Optional[int]:
    trailing_digits_reversed = []
    for character in reversed(value):
        if not character.isdigit():
            break
        trailing_digits_reversed.append(character)

    if len(trailing_digits_reversed) == 0:
        return None

    return int("".join(reversed(trailing_digits_reversed)))


def _resolve_tqdm_process_base() -> int:
    """First line of the band reserved for the current process."""
    process_name = multiprocessing.current_process().name
    process_index = _parse_trailing_integer(process_name)
    if process_index is None:
        return 0

    return max(process_index - 1, 0) * _PROCESS_TQDM_POSITION_STRIDE


def _acquire_tqdm_slot() -> int:
    """Reserve the lowest free line within this process's band and return its absolute position."""
    with _TQDM_POSITION_LOCK:
        slot = 0
        while slot in _ACTIVE_TQDM_SLOTS:
            slot += 1
        _ACTIVE_TQDM_SLOTS.add(slot)
    return _resolve_tqdm_process_base() + slot


def _release_tqdm_slot(position: int) -> None:
    with _TQDM_POSITION_LOCK:
        _ACTIVE_TQDM_SLOTS.discard(position - _resolve_tqdm_process_base())


@contextlib.contextmanager
def tqdm_position() -> Iterator[int]:
    """
    Reserve a distinct terminal line for a tqdm progress bar for the duration of the ``with`` block.

    Simultaneously active bars are packed contiguously from the top of the process's band, so each
    gets its own line and none overwrites another while it runs. The slot is released on exit and
    reused by a later bar. Most callers want :func:`tqdm_progress_bar`, which layers the correct
    ``leave`` / completion-line handling on top of this.
    """
    position = _acquire_tqdm_slot()
    try:
        yield position
    finally:
        _release_tqdm_slot(position)


@contextlib.contextmanager
def tqdm_progress_bar(**tqdm_kwargs) -> Iterator[tqdm.tqdm]:
    """
    A tqdm progress bar that keeps finished bars on screen without corrupting concurrent ones.

    While it runs, the bar occupies its own line (see :func:`tqdm_position`) and uses ``leave=False``
    so that on close it clears that line rather than leaving an orphaned partial render behind - the
    failure mode of ``leave=True`` at a fixed position, where a finishing bar redraws at position 0
    and never erases its last in-place render. Once the work completes, the bar's final line is
    re-emitted via ``tqdm.write`` (serialised through tqdm's lock), so completed bars remain on
    screen, later bars open on new lines below them, and no blank lines appear in between. If the
    ``with`` body raises, the bar is closed and no completion line is written.

    Pass the same keyword arguments you would pass to ``tqdm.tqdm`` (``desc``, ``total``, ``unit``,
    ...); ``position``, ``leave`` and ``dynamic_ncols`` are managed here.
    """
    with tqdm_position() as position:
        progress_bar = tqdm.tqdm(position=position, leave=False, dynamic_ncols=True, **tqdm_kwargs)
        try:
            yield progress_bar
            final_line = str(progress_bar)
        finally:
            progress_bar.close()
        tqdm.tqdm.write(final_line, file=progress_bar.fp)
