import functools
import logging
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional

import numpy as np

from ladder.core.common.logger import LadderLogger, get_logger

from .perf_counters import PerformanceCounter

LOGGER = get_logger(__name__)

_HOOKS: dict = {}

EVENT_TYPE_PATT = re.compile('^[a-z_0-9]+(?:\\.[a-z_0-9]+)*$')


class TimedScope:
    """
    A context manager to time blocks of code. Logs the successful/erroneous completion
    to a provided logger and has hooks to inform other code about the timing.

    Args:
        name: The name of the action being profiled. Must use `snake_casing`.
        logger: Reference to the  logger instance from the calling module.
        log_level: Log level to log the timing with.
        log_exceptions: Whether to log the stacktrace and exception one occurs within the scoped block.
        perf_counter: Optional reference to a PerformanceCounter to send timings to.
        metric_prefix: Prefix to add to the name when reporting to Datadog.

    Example:
    >>> with TimedScope('fetch_hf_content', logger):
    >>>     conn = httplib.HTTPConnection('huggingface.co')
    >>>     conn.request('GET', '/')
    """

    # Joins nested scope names into the chain that identifies a scope. Deliberately not '.', which
    # `EVENT_TYPE_PATT` allows inside a single scope name - a separator a name can contain leaves the
    # chain ambiguous about where one scope ends and the next begins
    SCOPE_SEPARATOR = '/'

    # Names of the scopes currently entered on this thread, outermost first. A scope's `full_name` is
    # the path through this stack, which gives every scope a unique stable identity
    active_scope_names: List[str] = []

    def __init__(
        self,
        name: str,
        logger: Optional[LadderLogger] = None,
        log_level: int = logging.INFO,
        log_exceptions: bool = True,
        perf_counter: Optional[PerformanceCounter] = None,
        **kwargs,
    ):
        if not valid_name(name):
            raise ValueError(f"TimedScope: 'name' must match {EVENT_TYPE_PATT}, but given: '{name}'")

        self.name = name
        # Resolved at construction, not on enter: a scope is built before the `with` body runs, so the
        # stack already holds its parents but not itself. Nested scopes built together (e.g. a
        # CPU/GPU pair) therefore agree on the same parent path.
        self.full_name = TimedScope.nested_name(name)
        self._logger: LadderLogger = logger if logger else LOGGER
        self._log_level = log_level
        self._log_exceptions = log_exceptions
        self._perf_counter = perf_counter
        self._info = kwargs
        self._start_secs = 0.0
        self.elapsed_secs = 0.0
        self._cancelled = False

    @staticmethod
    def nested_name(name: str) -> str:
        """
        `SCOPE_SEPARATOR`-separated chain of the active scope names plus the given `name`.
        Callers that need a scope's identity *before* constructing it - to look up or create its
        performance counter, say - must derive it this way so both agree on the same key.
        """
        return TimedScope.nest(*TimedScope.active_scope_names, name)

    @staticmethod
    def nest(*names: str) -> str:
        """
        The nested name a scope called `names[-1]` gets when entered inside `names[:-1]`. For code that
        has to name a scope it is not currently inside - declaring a counter up front, say.
        """
        return TimedScope.SCOPE_SEPARATOR.join(names)

    @staticmethod
    def clear_active_scopes() -> None:
        """
        Forget the scopes currently entered, for a process that inherited someone else's.

        A fork copies the parent's stack, but the child never runs the bodies those scopes wrap and so
        never pops them. Left in place they prefix every name the child records with whatever the
        parent happened to be doing at fork time.
        """
        TimedScope.active_scope_names = []

    @property
    def elapsed_ms(self) -> float:
        return round(self.elapsed_secs, 4) * 1000

    def add_info(self, **info) -> None:
        self._info = {
            **self._info,
            **info,
        }

    def cancel(self) -> None:
        if self._cancelled:
            raise RuntimeError(f"Tried to cancel TimedScope {self.name} that was already cancelled.")

        self._cancelled = True
        self._logger.log(self._log_level, self.name, status='cancel', **self._info)

    def _maybe_log_time(self):
        if self._perf_counter is not None:
            self._perf_counter.add_time_ms(self.elapsed_ms)

    def __enter__(self):
        self._start_secs = time.perf_counter()
        self._logger.log(self._log_level, self.name, status='start', **self._info)
        TimedScope.active_scope_names.append(self.name)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        TimedScope.active_scope_names.pop()
        if self._cancelled:
            return
        self.elapsed_secs = time.perf_counter() - self._start_secs
        duration_secs = round(self.elapsed_secs, 4)
        duration_ms = duration_secs * 1000

        self._maybe_log_time()

        hook = _HOOKS.get(self.name, None)
        if hook is not None:
            hook(self._info, duration_ms, exc_value)

        if exc_type and self._logger:
            # Do not log normal control flow exceptions like StopIteration.
            if self._log_exceptions and not isinstance(exc_value, (StopIteration, GeneratorExit)):
                self._logger.exception(
                    self.name, exc_value, status='error', duration_ms=duration_ms, **self._info
                )
        else:
            self._logger.log(
                self._log_level, self.name, status='success', duration_ms=duration_ms, **self._info
            )


DebugTimedScope = functools.partial(TimedScope, log_level=logging.DEBUG)

# The stack a child inherits is never true of the child, whoever forked it and whatever it goes on to
# do, so clear it at the fork rather than at each worker entry point
os.register_at_fork(after_in_child=TimedScope.clear_active_scopes)


class TimedScopeIterator:
    """Time how long it takes to retrieve each element from the iterator"""

    def __init__(self, iterator: Iterator, timed_scope_factory: Callable):
        self.iterator = iterator
        self.timed_scope_factory = timed_scope_factory

    def __iter__(self):
        return self

    def __next__(self):
        with self.timed_scope_factory():
            return next(self.iterator)


def timed_scope(logger: Optional[LadderLogger] = None) -> Callable:
    """
    Function wrapper that runs a TimedScope around a function and prints its execution duration
    every time the function is called if `logger` is provided.
    """

    def timed_scope_decorator(func):
        @functools.wraps(func)
        def timed_scope_inner(*args, **kwargs):
            with TimedScope(func.__name__, logger):
                return func(*args, **kwargs)

        return timed_scope_inner

    return timed_scope_decorator


@dataclass
class Stats:
    times_ms: List[float] = field(default_factory=list)
    error_count: int = 0

    def __str__(self):
        count = len(self.times_ms)
        return "count = {}, errors = {}, p50_ms = {}, p99_ms = {}, worst_ms = {}".format(
            count,
            self.error_count,
            np.percentile(self.times_ms, 50) if count > 0 else 0,
            np.percentile(self.times_ms, 99) if count > 0 else 0,
            np.max(self.times_ms) if count > 0 else 0,
        )


@contextmanager
def track_timed_scope_stats(name: str, group_by_keys: Iterable[str] = ()):
    groups: defaultdict = defaultdict(Stats)

    def hook(info, time_ms, error):
        key = "::".join(str(info[k]) for k in group_by_keys)
        group = groups[key]
        group.times_ms.append(time_ms)
        if error:
            group.error_count += 1

    _HOOKS[name] = hook
    yield groups
    del _HOOKS[name]


def register_hook(name: str, hook: Callable):
    """
    Register a callback function that will be called whenever the scope
    with matching name exits. The hook function should match the signature:

    def hook(info: dict, duration_ms: float, exc_val: Optional[BaseException])
    """
    _HOOKS[name] = hook


@functools.lru_cache(1024)
def valid_name(name: str) -> bool:
    return bool(EVENT_TYPE_PATT.fullmatch(name))
