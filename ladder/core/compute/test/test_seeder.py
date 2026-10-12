import unittest

import torch

from ladder.core.compute.seeder import SEEDER


class TestSeeder(unittest.TestCase):
    def tearDown(self):
        SEEDER.reset()

    def test_produce_same_random_numbers(self):
        SEEDER.seed_everything(42)
        prng = SEEDER.new_prng_np()
        first_rnd = prng.uniform(size=5)
        prng = SEEDER.new_prng_np()
        second_rnd = prng.uniform(size=5)
        generator = SEEDER.new_prng_torch()
        third_rnd = torch.empty(5).uniform_(generator=generator)

        SEEDER.reset()
        SEEDER.seed_everything(42)
        prng = SEEDER.new_prng_np()
        after_reset_first_rnd = prng.uniform(size=5)
        prng = SEEDER.new_prng_np()
        after_reset_second_rnd = prng.uniform(size=5)
        generator = SEEDER.new_prng_torch()
        after_reset_third_rnd = torch.empty(5).uniform_(generator=generator)

        self.assertListEqual(first_rnd.tolist(), after_reset_first_rnd.tolist())
        self.assertListEqual(second_rnd.tolist(), after_reset_second_rnd.tolist())
        self.assertTrue(any(first_rnd != second_rnd))
        self.assertListEqual(third_rnd.tolist(), after_reset_third_rnd.tolist())

    def test_seed_everything_is_consistent(self):
        SEEDER.seed_everything(42)
        torch_prng = SEEDER.new_prng_torch()
        np_prng = SEEDER.new_prng_np()
        initial_seed_torch_1 = torch_prng.initial_seed()
        initial_seed_np_1 = np_prng.get_state()[1][0]

        SEEDER.reset()

        torch_prng = SEEDER.new_prng_torch()
        np_prng = SEEDER.new_prng_np()
        SEEDER.seed_everything(42)
        initial_seed_torch_2 = torch_prng.initial_seed()
        initial_seed_np_2 = np_prng.get_state()[1][0]

        self.assertEqual(initial_seed_torch_1, initial_seed_torch_2)
        self.assertEqual(initial_seed_np_1, initial_seed_np_2)
        self.assertNotEqual(initial_seed_np_1, initial_seed_torch_1)

    def test_new_prng_torch_device(self):
        SEEDER.seed_everything(42)
        default_rnd = torch.rand(5, generator=SEEDER.new_prng_torch())
        next_seed = SEEDER.new_prng_np().get_state()[1][0]

        with self.subTest("default device is cpu"):
            SEEDER.reset()
            SEEDER.seed_everything(42)
            prng = SEEDER.new_prng_torch("cpu")
            self.assertEqual(prng.device, torch.device("cpu"))
            self.assertListEqual(torch.rand(5, generator=prng).tolist(), default_rnd.tolist())

        with self.subTest("seeds of later prngs do not depend on the device"):
            SEEDER.reset()
            SEEDER.seed_everything(42)
            SEEDER.new_prng_torch(torch.device("cpu"))
            self.assertEqual(SEEDER.new_prng_np().get_state()[1][0], next_seed)
