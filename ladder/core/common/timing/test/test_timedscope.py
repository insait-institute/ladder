import logging
import unittest
from unittest import mock
from unittest.mock import ANY, call

from ladder.core.common.timing.timedscope import TimedScope, TimedScopeIterator, register_hook


class TestTimedScope(unittest.TestCase):
    def setUp(self):
        pass

    def test_timed_scope_calls_registered_hook(self):
        hook = mock.Mock()
        register_hook("metric_name", hook)
        scope = TimedScope(
            name="metric_name",
            logger=mock.Mock(),
            log_level=logging.INFO,
            log_exceptions=True,
            perf_counter=None,
        )
        with scope:
            pass

        info = {}
        exc_val = None
        hook.assert_called_once_with(info, ANY, exc_val)

    def test_timed_scope_iterator_calls_registered_hook(self):
        hook = mock.Mock()
        register_hook("metric_name", hook)

        input_values = [1, 2, 3, 4]
        iterator = TimedScopeIterator(
            iter(input_values),
            lambda: TimedScope(
                name="metric_name",
                logger=mock.Mock(),
                log_level=logging.INFO,
                log_exceptions=True,
                perf_counter=None,
            ),
        )

        results = list()
        for n in iterator:
            results.append(n)

        hook.assert_has_calls(([call({}, ANY, None)] * (len(input_values))) + [call({}, ANY, ANY)])
        self.assertEqual(
            input_values,
            results,
            "Wrapped iterator returned %s which is different to input %s" % (results, input_values),
        )
