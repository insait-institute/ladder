import unittest

import numpy as np
import torch

from ladder.core.compute.tensor_utils import one_hot, verify_torch_tensor


class TestVerifyTorchTensor(unittest.TestCase):
    def test_matching_shape_and_dtype_pass(self):
        tensor = torch.zeros((2, 3), dtype=torch.int64)
        verify_torch_tensor(tensor, shape=[2, 3], dtype=torch.int64)
        verify_torch_tensor(tensor, shape=[-1, 3], dtype=[torch.int32, torch.int64])

    def test_wrong_ndim_or_shape_raise(self):
        tensor = torch.zeros((2, 3), dtype=torch.int64)
        with self.assertRaises(ValueError):
            verify_torch_tensor(tensor, shape=[-1], dtype=torch.int64)
        with self.assertRaises(ValueError):
            verify_torch_tensor(tensor, shape=[2, 4], dtype=torch.int64)

    def test_wrong_dtype_or_type_raise(self):
        tensor = torch.zeros((2, 3), dtype=torch.float32)
        with self.assertRaises(TypeError):
            verify_torch_tensor(tensor, shape=[2, 3], dtype=[torch.int32, torch.int64])
        with self.assertRaises(TypeError):
            verify_torch_tensor(np.zeros((2, 3)), shape=[2, 3], dtype=torch.float32)


class TestOneHot(unittest.TestCase):
    def test_basic_case(self):
        indices_tensor = torch.tensor(
            [
                [0],
                [1],
                [2],
                [0],
                [1],
                [2],
            ]
        )
        depth = 3

        expected_one_hot = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_array_equal(one_hot(indices_tensor, depth=depth), expected_one_hot)

    def test_custom_dim(self):
        indices_tensor = torch.tensor(
            [
                [[0, 1]],
                [[2, 0]],
                [[1, 2]],
            ]
        )
        depth = 3
        dim = 1

        expected_one_hot = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            ]
        )

        np.testing.assert_array_equal(one_hot(indices_tensor, depth=depth, dim=dim), expected_one_hot)

    def test_dim_out_of_bounds(self):
        indices_tensor = torch.tensor(
            [
                [0],
                [1],
                [2],
                [0],
                [1],
                [2],
            ]
        )

        depth = 3
        dim = 2

        with self.assertRaises(ValueError):
            one_hot(indices_tensor, depth=depth, dim=dim)

    def test_index_out_of_bounds(self):
        indices_tensor = torch.tensor(
            [
                [0],
                [1],
                [2],
                [0],
                [1],
                [2],
            ]
        )

        depth = 2

        with self.assertRaises(ValueError):
            one_hot(indices_tensor, depth=depth)

    def test_non_singleton_dim(self):
        indices_tensor = torch.tensor(
            [
                [[0, 1]],
                [[2, 0]],
                [[1, 2]],
            ]
        )
        depth = 3

        with self.assertRaises(ValueError):
            one_hot(indices_tensor, depth=depth)


if __name__ == '__main__':
    unittest.main()
