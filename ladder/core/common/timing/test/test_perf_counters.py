import unittest
from multiprocessing import Pool

from ladder.core.common.logger import get_logger
from ladder.core.common.timing.perf_counters import MultiprocessingPerformanceCounter as PerformanceCounter

LOGGER = get_logger(__name__)

global_counter = None


def global_counter_add(i):
    global_counter.add_time_ms(i + 1)


class PerfCountersTest(unittest.TestCase):
    def test_perf_counter(self):
        counter = PerformanceCounter('test', LOGGER)

        for i in range(100):
            counter.add_time_ms(i + 1)

        stats = counter.get_stats_and_reset()

        self.assertEqual(stats.count, 100)
        self.assertEqual(stats.mean_ms, 50.5)
        self.assertAlmostEqual(stats.percentile_95_ms, 95, places=0)

        counter.add_time_ms(30)
        stats = counter.get_stats_and_reset()

        self.assertEqual(stats.count, 1)
        self.assertEqual(stats.mean_ms, 30)
        self.assertAlmostEqual(stats.percentile_95_ms, 30, places=0)

    def test_empty_perf_counter_does_not_throw(self):
        counter = PerformanceCounter('test_empty', LOGGER)
        counter.print_stats_and_reset()

    def test_perf_counter_with_multiprocessing(self):
        global global_counter  # noqa: PLW0603
        counter = global_counter = PerformanceCounter('test_mp', LOGGER)

        pool = Pool(processes=4)
        pool.map(global_counter_add, range(100))

        stats = counter.get_stats_and_reset()

        self.assertEqual(stats.count, 100)
        self.assertEqual(stats.mean_ms, 50.5)
        self.assertAlmostEqual(stats.percentile_95_ms, 95, places=0)

        pool.map(global_counter_add, [29])
        stats = counter.get_stats_and_reset()

        self.assertEqual(stats.count, 1)
        self.assertEqual(stats.mean_ms, 30)
        self.assertAlmostEqual(stats.percentile_95_ms, 30, places=0)


if __name__ == '__main__':
    unittest.main()
