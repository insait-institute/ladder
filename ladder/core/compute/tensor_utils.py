import functools
from typing import Any, Callable, Sequence, Union

import numpy as np
import torch


def verify_torch_tensor(
    tensor: torch.Tensor, shape: Sequence[int], dtype: Union[torch.dtype, Sequence[torch.dtype]]
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected instance of torch.Tensor, but got instance of {type(tensor)}")

    if tensor.ndim != len(shape):
        raise ValueError(f"Expected ndim={len(shape)}, but got {tensor.ndim}")

    if any(
        dim_size != edim_size
        for dim_size, edim_size in zip(tensor.shape, shape, strict=True)
        if edim_size != -1
    ):
        raise ValueError(f"Expected shape {list(shape)}, but got {list(tensor.shape)}")

    if isinstance(dtype, torch.dtype):
        dtype = [dtype]

    if tensor.dtype not in dtype:
        raise TypeError(f"Expected dtype out of {list(dtype)}, but got dtype {tensor.dtype}")


def one_hot(indices: torch.Tensor, depth: int, dim: int = -1) -> torch.Tensor:
    if not -indices.ndim <= dim < indices.ndim:
        raise ValueError(f"one_hot expected {-indices.ndim} <= dim < {indices.ndim}, got dim = {dim}")
    if indices.shape[dim] != 1:
        raise ValueError(
            f"one_hot expected indices.shape[dim] = 1, got indices.shape = {indices.shape}, "
            f"indices.shape[{dim}] = {indices.shape[dim]}"
        )
    if indices.min() < 0 or indices.max() >= depth:
        raise ValueError(
            f"one_hot expected 0 <= indices < depth = {depth}, "
            f"got min(indices) = {indices.min()}, max(indices) = {indices.max()}"
        )

    one_hot_shape = list(indices.shape)
    one_hot_shape[dim] = depth

    # Scatter expects int64
    indices = indices.to(torch.int64)

    return torch.zeros(one_hot_shape, device=indices.device).scatter_(dim=dim, index=indices, value=1)


@torch.jit.script
def shape_as_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)


def expand_dims(tensor: torch.Tensor, ndim: int, order: Sequence[int]) -> torch.Tensor:
    """
    Expand the dimensions of `tensor` to `ndim` such that all new dimensions have size of 1
    Args:
        tensor: torch.Tensor of any shape
        ndim: Number of output dimensions. Must be >= `tensor.ndim`
        order: Sequence of size `tensor.ndim + 1`. Contains only values of 1 and a single value of -1,
            indicating where the new `ndim - tensor.ndim` dimensions will be inserted
    Returns:
        torch.Tensor with dimensions `ndim`, a view of `tensor`

    Ex:
        expand_dims(torch.ones([2, 3, 4]), ndim=5, order=[1, -1, 1, 1]).shape -> [2, 1, 1, 3, 4]
        expand_dims(torch.ones([2, 3, 4]), ndim=5, order=[-1, 1, 1, 1]).shape -> [1, 1, 2, 3, 4]
        expand_dims(torch.ones([2, 3, 4]), ndim=5, order=[1, 1, 1, -1]).shape -> [2, 3, 4, 1, 1]
    """
    assert tensor.ndim <= ndim, f"{tensor.ndim} > {ndim}; shape={tensor.shape}"
    assert len(order) == tensor.ndim + 1, f"{len(order)} != {tensor.ndim + 1}; shape={tensor.shape}"

    order = list(order)

    assert order.count(-1) == 1, "Order must have exactly one value of -1"
    assert order.count(1) == len(order) - 1, "Order must have exactly len(order) - 1 values of 1"

    if tensor.ndim == ndim:
        return tensor

    insert_index = order.index(-1)
    view = list(tensor.shape[:insert_index]) + [1] * (ndim - tensor.ndim) + list(tensor.shape[insert_index:])

    tensor = tensor.view(view)
    return tensor


def numpy_compatible(function: Callable[..., Any]) -> Callable[..., Any]:
    """
    Let a function implemented in torch be called with numpy arrays.

    When any argument is an np.ndarray, every np.ndarray argument reaches `function` as a torch.Tensor,
    and every torch.Tensor in the result is handed back to the caller as an np.ndarray. A call without
    a single np.ndarray argument reaches `function` untouched and its result is returned as-is.

    Both conversions share memory, so the arrays going in and the arrays coming out are views onto the
    buffers the wrapped function operates on, and no data is ever copied. Sharing only works for arrays
    torch can address directly, so an array with negative strides (such as `array[::-1]`) raises a
    ValueError instead of being silently copied.

    Arrays are converted only at the top level of the arguments and of the result, where the result may
    also be a tuple, a list or a dict of tensors. Arrays nested any deeper are left alone.

    NOTE: numpy -> torch conversion is safe, but the reverse is not. Do NOT implement the reverse as it
    breaks torch functionality - silent CPU-GPU transfer or differentiability and grad computation.
    """

    @functools.wraps(function)
    def numpy_compatible_function(*args: Any, **kwargs: Any) -> Any:
        if not any(isinstance(value, np.ndarray) for value in (*args, *kwargs.values())):
            return function(*args, **kwargs)

        result = function(
            *(_maybe_np_to_torch(value) for value in args),
            **{name: _maybe_np_to_torch(value) for name, value in kwargs.items()},
        )

        return _maybe_torch_to_np(result)

    return numpy_compatible_function


def _maybe_np_to_torch(value: Any) -> Any:
    return torch.from_numpy(value) if isinstance(value, np.ndarray) else value


def _maybe_torch_to_np(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, tuple):
        return tuple(_maybe_torch_to_np(item) for item in value)
    if isinstance(value, list):
        return [_maybe_torch_to_np(item) for item in value]
    if isinstance(value, dict):
        return {key: _maybe_torch_to_np(item) for key, item in value.items()}
    return value
