import unittest

import torch

from ladder.core.compute.seeder import SEEDER


class TestSeederGPU(unittest.TestCase):
    def tearDown(self):
        SEEDER.reset()

    def test_cuda_generator_lives_on_cuda(self):
        prng = SEEDER.new_prng_torch('cuda')
        self.assertEqual(prng.device.type, 'cuda')
        self.assertEqual(torch.randperm(8, generator=prng, device='cuda').device.type, 'cuda')

    def test_seed_everything_reseeds_cuda_generators(self):
        prng = SEEDER.new_prng_torch('cuda')
        SEEDER.seed_everything(42)
        first_seed = prng.initial_seed()
        first_perm = torch.randperm(64, generator=prng, device='cuda').tolist()

        SEEDER.seed_everything(42)
        self.assertEqual(prng.initial_seed(), first_seed)
        self.assertListEqual(torch.randperm(64, generator=prng, device='cuda').tolist(), first_perm)

    def test_cuda_and_cpu_generators_get_the_same_seed_sequence(self):
        SEEDER.seed_everything(42)
        cuda_seed = SEEDER.new_prng_torch('cuda').initial_seed()

        SEEDER.reset()
        SEEDER.seed_everything(42)
        cpu_seed = SEEDER.new_prng_torch('cpu').initial_seed()

        self.assertEqual(cuda_seed, cpu_seed)
