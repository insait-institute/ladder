from __future__ import annotations

import hashlib
import os
import random
import struct
from typing import Dict, List, Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None

# When WORLD_MANAGER is in scope, import it for automatic rank handling. If not in scope, we
# are most likely running a single process and WORLD_MANAGER is not needed.
try:
    from ladder.train.core.common.world_manager import WORLD_MANAGER
except ImportError:
    WORLD_MANAGER = None


class _SeedManager:
    """
    Singleton class that takes care of seeding

    - seed_everything() seeds torch, numpy, random and all prngs created via new_prng_np() / new_prng_torch()
    - Use new_prng_np() / new_prng_torch() instead of global calls to libraries.
        Allows for clear separation of random number generation of one object from another
        and easier diff tracking when a new random generation call is introduced
    - new_prng_torch(device) creates the generator on the given device. The seed of every prng depends only
        on the order of new_prng_*() calls, never on the requested device, so moving a generator to another
        device leaves the seeds of every other prng unchanged
    - Any random numbers generated before seed_everything() will be non-deterministic

    NOTE: Not suitable for distributed training. Use ladder/train/core/common/seeder.py instead.
    """

    def __init__(self):
        self._seeded = False
        self._seed_generator = np.random.RandomState()
        self._generated_prngs: Dict[np.random.RandomState | torch.Generator, int] = {}

    @property
    def seeded(self) -> bool:
        return self._seeded

    def seed_everything(self, seed: int):
        if seed < 0:
            raise ValueError("Invalid seed.")

        if WORLD_MANAGER is not None:
            # Make sure we seed different ranks with different seeds
            seed = seed + WORLD_MANAGER.rank % np.iinfo(np.int32).max

        # Seed the local seeder
        self._seed_generator.seed(seed)

        self._seeded = True

        # Sample separate seeds from seeder to avoid correlation among modules
        # Note that the function generating the new seeds is still a deterministic function of seed
        if torch is not None:
            torch_seed = self._make_int32_seed()
            torch.manual_seed(torch_seed)
            # NOTE: Generate a CUDA seed no matter if CUDA is available or not. Otherwise, randomness
            # reproducibility depends on hardware and leads to inconsistencies, especially with tests.
            torch_cuda_seed = self._make_int32_seed()
            if self._torch_cuda_available:
                torch.cuda.manual_seed(torch_cuda_seed)
        np.random.seed(self._make_int32_seed())  # noqa: NPY002
        random.seed(self._make_int32_seed())

        # Seed all PRNGS
        for prng in self._generated_prngs:
            if isinstance(prng, np.random.RandomState):
                prng.seed(self._make_int32_seed())
            elif torch is not None and isinstance(prng, torch.Generator):
                prng.manual_seed(self._make_int32_seed())

    def new_prng_np(self) -> np.random.RandomState:
        """
        Create a new instance of a pseudo-random number generator (PRNG). If `self._seeded==True` (seed has been
        set globally), then the PRNG is also seeded deterministically. Otherwise, it is seeded with a random seed.
        If different runs of a program change the order of calls to this subroutine, then results will be different.
        """

        # Create the new PRNG and add it to the list of generated PRNGs
        prng = np.random.RandomState()
        self._generated_prngs[prng] = len(self._generated_prngs)

        # Seed the new PRNG, no matter if self._seeded == True.
        # We still get randomness if self._seeded == True and deterministic seed if self._seeded == True
        seed = self._make_int32_seed()
        prng.seed(seed)

        return prng

    def new_prng_torch(self, device: Union[str, torch.device] = 'cpu') -> torch.Generator:
        """
        Create a new instance of a pseudo-random number generator (PRNG). If `self._seeded==True` (seed has been
        set globally), then the PRNG is also seeded deterministically. Otherwise, it is seeded with a random seed.
        If different runs of a program change the order of calls to this subroutine, then results will be different.

        Args:
            device: Device the generator lives on. Tensors drawn with it must be on the same device. The seed is
                drawn from the seeder in call order regardless of the device, so requesting a different device
                does not change the seed of this or any later prng. `seed_everything()` reseeds generators on
                any device through `manual_seed()`
        """
        if torch is None:
            raise RuntimeError("Torch is not available.")

        # Create the new PRNG and add it to the list of generated PRNGs
        prng = torch.Generator(device=device)
        self._generated_prngs[prng] = len(self._generated_prngs)

        # Seed the new PRNG, no matter if self._seeded == True.
        # We still get randomness if self._seeded == True and deterministic seed if self._seeded == True
        seed = self._make_int32_seed()
        prng.manual_seed(seed)

        return prng

    def reset(self):
        """Restores seeder to pristine state."""
        self._seeded = False
        self._seed_generator = np.random.RandomState()
        self._generated_prngs = {}

    def _make_int32_seed(self) -> int:
        return self._hash_seed(self._make_seed(max_bytes=4))

    def _make_seed(self, seed=None, max_bytes=4) -> int:
        if seed is not None:
            seed = seed % 2 ** (8 * max_bytes)
        else:
            if self._seeded:
                seed = self._seed_generator.randint(0, 2 ** (8 * max_bytes))
            else:
                seed = self._bigint_from_bytes(os.urandom(max_bytes))
        return seed

    def _hash_seed(self, seed=None, max_bytes=4) -> int:
        """
        Different modules are likely to request an autogenerated seed. There is literature
        indicating that generating seeds in a linear fashion will cause correlated outputs
        generated by those seeds. To avoid this, we need to generate the seeds in a non-linear
        fashion. For more details check:

        http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
        http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
        http://dl.acm.org/citation.cfm?id=1276928

        Thus, we hash the seeds, which should get rid of simple correlations.

        Args:
            seed: int
            max_bytes: Maximum number of bytes to use in the hashed seed.
        Returns:
            int - the hashed seed
        """
        hash_code = hashlib.sha512(str(seed).encode('utf8')).digest()
        return self._bigint_from_bytes(hash_code[:max_bytes])

    def _bigint_from_bytes(self, data: bytes) -> int:
        sizeof_int = 4
        padding = sizeof_int - len(data) % sizeof_int
        data += b'\0' * padding
        int_count = int(len(data) / sizeof_int)
        unpacked = struct.unpack("{}I".format(int_count), data)
        accum = 0
        for i, val in enumerate(unpacked):
            accum += 2 ** (sizeof_int * 8 * i) * val
        return accum

    def _int32_list_from_bigint(self, bigint: int) -> List[int]:
        """Convert int of any size into a list of int32 numbers"""
        # Special case 0
        if bigint < 0:
            raise ValueError(f'Seed must be non-negative, not {bigint}')
        if bigint == 0:
            return [0]

        ints = []
        while bigint > 0:
            bigint, mod = divmod(bigint, 2**32)
            ints.append(mod)
        return ints

    def get_state(self) -> dict:
        state = {
            'np_rng_state': np.random.get_state(),  # noqa: NPY002
            'random_rng_state': random.getstate(),
        }
        if torch is not None:
            state['torch_rng_cpu_state'] = torch.get_rng_state()
            # NOTE: Explicitly not using get_rng_state_all. We care only about this rank
            if self._torch_cuda_available:
                state['torch_rng_cuda_state'] = torch.cuda.get_rng_state()
        return state

    def set_state(self, state: dict):
        if torch is not None:
            torch.set_rng_state(state['torch_rng_cpu_state'])
            if 'torch_rng_cuda_state' in state and self._torch_cuda_available:
                torch.cuda.set_rng_state(state['torch_rng_cuda_state'])
        else:
            assert 'torch_rng_cpu_state' not in state
            assert 'torch_rng_cuda_state' not in state
        np.random.set_state(state['np_rng_state'])  # noqa: NPY002
        random.setstate(state['random_rng_state'])

    def get_prngs_order(self) -> Dict[np.random.RandomState | torch.Generator, int]:
        """Used for debugging and figuring out if the order in which PRNGs have been created is different"""
        return {**self._generated_prngs}

    def get_prng_index(self, prng: np.random.RandomState | torch.Generator) -> int:
        """Used for debugging and figuring out the index of a generated PRNG in the sequence of all PRNGs"""
        return self._generated_prngs.get(prng, -1)

    @property
    def _torch_cuda_available(self) -> bool:
        """
        torch.cuda.is_available() can be misleading on if no GPUs have been allocated to the job. Instead
        use torch.cuda.is_initialized(). This relies on the main process having initialized CUDA already.
        """
        return torch is not None and torch.cuda.is_initialized()


SEEDER = _SeedManager()
