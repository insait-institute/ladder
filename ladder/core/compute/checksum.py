from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import PIL.Image

try:
    import torch
except ImportError:
    torch = None

import xxhash

from ladder.core.databib.dataclasses import DataclassInterface


def calculate_model_state_dict_sum(state_dict: Dict[str, torch.Tensor], hashed: bool = True) -> int | float:
    if torch is None:
        raise RuntimeError("torch is not available")
    if hashed:
        output = sum(_tensor_hash(param) for param in state_dict.values())
    else:
        output = sum(_tensor_checksum(param) for param in state_dict.values())
    return output


def calculate_model_parameters_sum(model: torch.nn.Module, hashed: bool = True) -> int | float:
    if torch is None:
        raise RuntimeError("torch is not available")
    if hashed:
        output = sum(_tensor_hash(param) for param in model.parameters())
    else:
        output = sum(_tensor_checksum(param) for param in model.parameters())
    return output


def calculate_attributes_checksums(data: DataclassInterface, hashed: bool = True) -> Dict[str, int | float]:
    return {key: calculate_data_checksum(value, hashed=hashed) for key, value in data.items(recursive=True)}


def calculate_data_checksum(
    data: DataclassInterface | torch.Tensor | np.ndarray | Tuple | list | dict | None,
    hashed: bool = True,
) -> int | float:
    if data is None:
        return 0

    if hashed:
        _tensor_sum = _tensor_hash
        _ndarray_sum = _ndarray_hash
    else:
        _tensor_sum = _tensor_checksum
        _ndarray_sum = _ndarray_checksum

    if torch is not None and isinstance(data, torch.Tensor):
        output = _tensor_sum(data)

    elif isinstance(data, np.ndarray):
        output = _ndarray_sum(data)

    elif isinstance(data, PIL.Image.Image):
        output = _ndarray_sum(np.asarray(data))

    elif isinstance(data, DataclassInterface):
        output = sum(calculate_data_checksum(value, hashed=hashed) for _, value in data.items(recursive=True))

    elif isinstance(data, dict):
        output = sum(calculate_data_checksum(value, hashed=hashed) for value in data.values())

    elif isinstance(data, (tuple, list)):
        output = sum(calculate_data_checksum(item, hashed=hashed) for item in data)

    elif isinstance(data, (int, float, str, bool, np.generic)):
        if hashed:
            output = _str_hash(str(data))
        else:
            if isinstance(data, (str)):
                output = _ndarray_checksum(np.frombuffer(data.encode(), dtype=np.uint8))
            else:
                output = abs(float(data))

    else:
        raise TypeError(f'Unsupported data type for checksum calculation: {type(data)}')

    return output


def _str_hash(data: str) -> int:
    return xxhash.xxh64(str(data).encode()).intdigest()


def _ndarray_hash(
    data: np.ndarray,
    dtype: torch.dtype | np.number | None = None,
    shape: torch.Size | Tuple[int, ...] | None = None,
    device: torch.device | str | None = None,
) -> int:
    dtype = dtype or data.dtype
    shape = shape or data.shape
    device = device or 'cpu'

    h64 = xxhash.xxh64()
    h64.update(np.ascontiguousarray(data).tobytes())
    h64.update(str(shape).encode())
    h64.update(str(dtype).encode())
    output = h64.intdigest()

    return output


def _tensor_hash(tensor: torch.Tensor) -> int:
    if torch is None:
        raise RuntimeError("torch is not available")
    dtype = tensor.dtype
    if dtype.is_complex:
        raise NotImplementedError("Checksum for complex tensors is not implemented.")
    # Cast float16, bfloat16 and any float8 to float32 since numpy doesn't support them
    if dtype.is_floating_point and dtype.itemsize <= 2:
        dtype = torch.float32
    return _ndarray_hash(
        tensor.detach().to(dtype=dtype, device='cpu').numpy(),
        dtype=tensor.dtype,
        shape=tensor.shape,
        device=tensor.device,
    )


def _ndarray_checksum(data: np.ndarray) -> float:
    if np.issubdtype(data.dtype, np.number):
        value = np.abs(data).sum().item()
    else:
        value = np.frombuffer(data.astype('S').tobytes(), dtype=np.uint8).sum().item()
    return float(value)


def _tensor_checksum(tensor: torch.Tensor, chunk_size: int | None = 2**18) -> float:
    """
    This is very tricky to implement efficiently and correctly! Note the caveats:

    1. `torch.sum()` is not guaranteed to be deterministic. The reason is that for large tensors,
    parallel reductions (sum) are not guaranteed to be associative, and PyTorch may choose different
    reduction trees depending on backend, thread scheduling, contiguity, device, or even memory alignment.
    This causes non-deterministic summation order, which in turn causes different floating-point rounding
    or overflow behavior, even when the input tensor is identical. Unfortunately, additional configurations
    such as `torch.set_num_threads(1)` and `torch.use_deterministic_algorithms(True)` still do not guarantee
    absolutely deterministic summation order

    2. We can always implement this with a for loop, but this can be extremely slow for large tensors or
    for computing the checksum of entire models
    """
    if torch is None:
        raise RuntimeError("torch is not available")

    assert isinstance(tensor, torch.Tensor), type(tensor)

    if tensor.dtype.is_complex:
        raise NotImplementedError("Checksum for complex tensors is not implemented.")

    if tensor.dtype == torch.bool:
        # torch.Tensor.abs is not implemented for bool
        value = tensor.detach().sum().cpu().item()
        return float(value)

    # Determine the dtype for summation to avoid overflow and improve accuracy
    if tensor.dtype.is_floating_point:
        dtype = torch.float64  # This is very important
    else:
        dtype = None  # Let torch.sum handle upcasting to avoid overflow

    tensor = tensor.detach().abs()

    # Compute the sum in fixed-size chunks to improve reproducibility
    if chunk_size is not None:
        tensor = tensor.flatten().contiguous()

        chunk_sums = torch.stack([chunk.sum(dtype=dtype) for chunk in tensor.split(chunk_size)])
        value = chunk_sums.sum(dtype=dtype).cpu().item()

    # Compute the sum by summing over each dimension iteratively
    else:
        value = tensor
        for dim in range(tensor.ndim):
            value = value.sum(dim=dim, dtype=dtype)
        value = value.cpu().item()

    return float(value)
