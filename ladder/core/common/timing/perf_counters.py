import ctypes
import os
import queue
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass
from multiprocessing import Array, Value
from typing import Callable, Dict, List, MutableSequence, Optional, Sequence, Tuple, cast

import numpy as np

from ladder.core.common.logger import LadderLogger, get_logger

LOGGER = get_logger(__name__)
USE_PERF_COUNTERS = os.environ.get('USE_PERF_COUNTERS', '') == 'true'


class PerformanceCounter(ABC):
    @dataclass
    class Stats:
        count: float
        stddev_count_per_process: float
        num_procs: float
        total_ms: float
        mean_ms: float
        median_ms: float
        percentile_95_ms: float
        # percentile95th excess over mean_ms as % of mean_ms:
        percentile95th_over_mean: Optional[int] = None

    EMPTY_STATS = Stats(*((0,) * 8))
    REGISTERED_COUNTERS: Dict[str, 'PerformanceCounter'] = OrderedDict()

    def __init__(self, name: str, logger: Optional[LadderLogger] = None):
        self.name = name
        self._logger = logger

    @abstractmethod
    def add_time_ms(self, time_ms: float) -> None:
        pass

    @abstractmethod
    def add_get_time_ms(self, get_time_ms: Callable[[], float]) -> None:
        pass

    @abstractmethod
    def get_stats_and_reset(self) -> 'PerformanceCounter.Stats':
        pass

    def print_stats_and_reset(self) -> None:
        stats = self.get_stats_and_reset()
        if stats.count > 0 and self._logger:
            self._logger.info(self.name + '_perf_stats', **asdict(stats))

    @classmethod
    def create(cls, name: str, logger: LadderLogger, always_enable=False, **kwargs) -> 'PerformanceCounter':
        if always_enable or USE_PERF_COUNTERS:
            counter = MultiprocessingPerformanceCounter(name, logger, **kwargs)
            cls.REGISTERED_COUNTERS[name] = counter
            return counter
        return DummyPerformanceCounter(name, logger)

    @classmethod
    def _build_stats(
        cls, counts_per_process: List[int], num_procs: int, times_ms: List[float]
    ) -> "PerformanceCounter.Stats":
        if len(times_ms) == 0:
            return cls.EMPTY_STATS

        mean_ms = float(round(np.mean(times_ms), 2))
        percentile_95_ms = float(round(np.percentile(times_ms, 95), 2))
        percentile95th_over_mean: int = (
            int((percentile_95_ms - mean_ms) / mean_ms * 100) if mean_ms > 0 else 0
        )
        return cls.Stats(
            count=len(times_ms),
            stddev_count_per_process=float(round(np.std(counts_per_process), 1)),
            num_procs=num_procs,
            total_ms=float(round(np.sum(times_ms))),
            mean_ms=mean_ms,
            median_ms=float(round(np.median(times_ms), 2)),
            percentile_95_ms=percentile_95_ms,
            percentile95th_over_mean=percentile95th_over_mean,
        )


class DummyPerformanceCounter(PerformanceCounter):
    def __init__(self, name: str, logger: LadderLogger, **_):
        super(DummyPerformanceCounter, self).__init__(name, logger)

    def add_time_ms(self, time_ms: float) -> None:
        pass

    def add_get_time_ms(self, get_time_ms: Callable[[], float]) -> None:
        pass

    def get_stats_and_reset(self) -> PerformanceCounter.Stats:
        return PerformanceCounter.EMPTY_STATS


class SingleProcessPerformanceCounter(PerformanceCounter):
    # TODO: delete once we know how to insert queues into MultiprocessingPerformanceCounter

    def __init__(self, name: str, logger: LadderLogger, max_length: int = 8192):
        super(SingleProcessPerformanceCounter, self).__init__(name, logger)

        self._buffer: np.ndarray = np.empty(max_length)

        self._count: int = 0
        self._next_index: int = 0
        self._max_length: int = max_length

        self._queue: queue.Queue = queue.Queue()

    def add_time_ms(self, time_ms: float) -> None:
        self._buffer[self._next_index] = time_ms
        self._next_index = (self._next_index + 1) % self._max_length
        self._count = min(self._count + 1, self._max_length)

    def add_get_time_ms(self, get_time_ms: Callable[[], float]) -> None:
        self._queue.put(get_time_ms)

    def get_stats_and_reset(self) -> PerformanceCounter.Stats:
        self._empty_queue()
        stats = self._build_stats(
            counts_per_process=[self._count], num_procs=1, times_ms=self._buffer[: self._count]  # type: ignore[arg-type]
        )

        self._count = self._next_index = 0
        return stats

    def _empty_queue(self) -> None:
        while not self._queue.empty():
            get_time_ms = self._queue.get()
            self.add_time_ms(time_ms=get_time_ms())


class MultiprocessingPerformanceCounter(PerformanceCounter):
    """
    A multiprocessing-compatible performance counter, using shared memory to collect
    stats which can be printed by the main process coordinating a worker pool.

    WHY `max_procs` EXISTS: it is the size of a fixed pool, not a safety limit. Children read and write
    these buffers because they *inherit* them across the fork, and a child cannot create memory the
    parent is able to read - so every buffer has to be allocated up front, which means the number of
    writing processes has to be known up front too. That is also why there is no "unlimited" option:
    supporting an unbounded number of writers would mean a different transport (a queue the children
    push to, or on-demand shared_memory segments registered somewhere).

    Size it as `1 + <number of child processes that will write>`; threads inside a process share its
    slot and do not need one each. A slot costs `max_length` floats (~32 KB by default).
    """

    def __init__(
        self, name: str, logger: Optional[LadderLogger] = None, max_length: int = 8192, max_procs: int = 4
    ):
        super(MultiprocessingPerformanceCounter, self).__init__(name, logger)

        # Allocate all shared buffers in advance (prior to forking), so the main process has access to them.
        # Individual buffers are allocated to worker processes on first use.
        self._shared_buffers: Sequence[MutableSequence[float]] = [
            cast(MutableSequence[float], Array(ctypes.c_float, max_length, lock=False))
            for _ in range(max_procs)
        ]
        self._shared_count_and_indexes: Sequence[MutableSequence[int]] = [
            cast(MutableSequence[int], Array(ctypes.c_uint, 2, lock=False)) for _ in range(max_procs)
        ]
        self._shared_proc_count = Value(ctypes.c_uint)
        self._max_procs = max_procs

        self._times_ms: Optional[MutableSequence[float]] = None
        self._count_and_index: Optional[MutableSequence[int]] = None
        self._pid: Optional[int] = None
        self._max_length = max_length

    def add_time_ms(self, time_ms: float) -> None:
        if self._times_ms is None or self._pid != os.getpid():
            self._times_ms, self._count_and_index = self._allocate_shared_buffer_to_process()
            self._pid = os.getpid()

        assert self._times_ms is not None
        self._times_ms[self._get_next_index()] = time_ms
        self._set_count(min(self._get_count() + 1, len(self._times_ms)))
        self._set_next_index((self._get_next_index() + 1) % len(self._times_ms))

    def add_get_time_ms(self, get_time_ms: Callable[[], float]) -> None:
        self.add_time_ms(time_ms=get_time_ms())  # TODO: refactor

    def get_stats_and_reset(self) -> PerformanceCounter.Stats:
        with self._shared_proc_count.get_lock():
            num_procs = self._shared_proc_count.value  # type: ignore[attr-defined]

            counts_per_process = [cast(int, self._shared_count_and_indexes[i][0]) for i in range(num_procs)]
            times_ms: List[float] = sum(
                [
                    cast(List[float], self._shared_buffers[i][0 : counts_per_process[i]])
                    for i in range(num_procs)
                ],
                [],
            )

            stats = self._build_stats(counts_per_process, num_procs, times_ms)

            for i in range(num_procs):
                self._shared_count_and_indexes[i][1] = 0
                self._shared_count_and_indexes[i][0] = 0

            return stats

    def _allocate_shared_buffer_to_process(self) -> Tuple[MutableSequence[float], MutableSequence[int]]:
        with self._shared_proc_count.get_lock():
            if self._shared_proc_count.value >= self._max_procs:  # type: ignore[attr-defined]
                # Everything this process records from here on is written to a throwaway buffer and
                # silently dropped, so signal a warning - the timings just go missing
                LOGGER.warn(
                    'max_procs_exceeded',
                    counter_name=self.name,
                    max_procs=self._max_procs,
                    pid=os.getpid(),
                )
                return [0], [0, 0]

            buffer_index = self._shared_proc_count.value  # type: ignore[attr-defined]
            self._shared_proc_count.value += 1  # type: ignore[attr-defined]

            LOGGER.debug(
                'allocate_buffer',
                counter_name=self.name,
                pid=os.getpid(),
                buffer_index=buffer_index,
            )

            return (
                self._shared_buffers[buffer_index],
                self._shared_count_and_indexes[buffer_index],
            )

    def _get_count(self) -> int:
        assert self._count_and_index is not None
        return self._count_and_index[0]

    def _set_count(self, v: int) -> None:
        assert self._count_and_index is not None
        self._count_and_index[0] = v

    def _get_next_index(self) -> int:
        assert self._count_and_index is not None
        return self._count_and_index[1]

    def _set_next_index(self, v: int) -> None:
        assert self._count_and_index is not None
        self._count_and_index[1] = v
