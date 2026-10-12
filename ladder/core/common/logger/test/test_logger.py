# fmt: off
import logging
import unittest
from unittest.mock import MagicMock

from ladder.core.common.logger.logger import LadderLogger


class TestLadderLogger(unittest.TestCase):
    def setUp(self):
        self.py_logger = MagicMock(logging.Logger)
        self.logger = LadderLogger(self.py_logger)

    def test_info(self):
        for i in range(2):
            self.logger.info('info', i=i)

        self.assertEqual(self.py_logger.log.call_count, 2)

    def test_once(self):
        for i in range(2):
            self.logger.info_once('info', i=i)

        self.assertEqual(self.py_logger.log.call_count, 1)

    def test_once_different_call_site(self):
        self.logger.info_once('info', i=1)
        self.logger.info_once('info', i=2)

        self.assertEqual(self.py_logger.log.call_count, 2)

    def test_once_same_call_site(self):
        def log(i):
            self.logger.info_once('info', i=i)

        log(1)
        log(2)

        self.assertEqual(self.py_logger.log.call_count, 1)

    def test_once_same_line(self):
        for i in range(2):
            self.logger.info_once('info', i=i)

        self.assertEqual(self.py_logger.log.call_count, 1)


if __name__ == '__main__':
    unittest.main()
