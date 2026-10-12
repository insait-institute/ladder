import unittest

import numpy as np

from ladder.core.common.np_utils import np_nearest_neighbour_indices, np_ranges, np_unique


class TestNpUtils(unittest.TestCase):
    def test_np_unique(self):
        data = np.array([1, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4])
        expected_indices_of_first_occurence = np.array([0, 3, 5, 9])
        expected_inverse_indices = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
        expected_counts = np.array([3, 2, 4, 2])
        expected_unique_ids = np.array([1, 2, 0, 4])

        unique_ids, indices_of_first_occurence, inverse_indices, counts = np_unique(data)

        np.testing.assert_array_equal(unique_ids, expected_unique_ids)
        np.testing.assert_array_equal(indices_of_first_occurence, expected_indices_of_first_occurence)
        np.testing.assert_array_equal(inverse_indices, expected_inverse_indices)
        np.testing.assert_array_equal(counts, expected_counts)

    def test_np_nearest_neighbour_indices(self):
        with self.subTest('outside_left'):
            values = np.array([1, 2, 6, 10])
            queries = np.array([0, 3, 13])
            expected_indices = np.array([0, 1, 3])
            np.testing.assert_array_equal(
                np_nearest_neighbour_indices(values, queries, side='left'), expected_indices
            )

        with self.subTest('outside_right'):
            values = np.array([1, 2, 6, 10])
            queries = np.array([0, 3, 13])
            expected_indices = np.array([0, 1, 3])

            np.testing.assert_array_equal(
                np_nearest_neighbour_indices(values, queries, side='right'), expected_indices
            )

        with self.subTest('left_side'):
            values = np.array([1, 2, 6, 10])
            queries = np.array([4, 8])
            expected_indices = np.array([1, 2])
            np.testing.assert_array_equal(
                np_nearest_neighbour_indices(values, queries, side='left'), expected_indices
            )

        with self.subTest('right_side'):
            values = np.array([1, 2, 6, 10])
            queries = np.array([4, 8])
            expected_indices = np.array([2, 3])
            np.testing.assert_array_equal(
                np_nearest_neighbour_indices(values, queries, side='right'), expected_indices
            )

    def test_np_ranges(self):
        lows = np.array([1, 8, 20])
        highs = np.array([4, 14, 22])
        expected_values = np.array([1, 2, 3, 8, 9, 10, 11, 12, 13, 20, 21])
        np.testing.assert_array_equal(np_ranges(lows, highs), expected_values)

        with self.subTest('assert_raises'):
            with self.assertRaises(AssertionError):
                lows = np.array([1, 8, 20])
                highs = np.array([4, 14, 20])
                np_ranges(lows, highs)
