from enum import Enum
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
import torch

from ladder.core.databib.dataclasses import Dataclass, DataclassDict, DataclassList


class PadSide(Enum):
    """Which side of each sequence to pad when batching to a common length."""

    LEFT = "left"
    RIGHT = "right"


TensorT = Union[
    Dataclass,
    torch.Tensor,
    np.ndarray,
    Dict[str, torch.Tensor],
    Dict[str, np.ndarray],
    Dict[str, Dataclass],
    List[torch.Tensor],
    List[np.ndarray],
    List[Dataclass],
    None,
]


def safe_collate(discard_fraction=0.0, dim=0):
    def inner_function(batch, share_memory=True):
        """Ensures we only include data we were able to load"""
        requested_batch_size = len(batch)
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) <= discard_fraction * requested_batch_size:
            return None
        return collate_elements(batch, share_memory, dim=dim)

    return inner_function


def apply_on_dataclasses(
    batch: Sequence[TensorT],
    apply_on_batch: Callable[[str, Sequence[torch.Tensor | np.ndarray]], torch.Tensor | np.ndarray],
    skip_keys: Sequence[str] = (),
    checks_enabled: bool = True,
    key_path: str = "",
) -> TensorT:
    """
    Applies the function `apply_on_batch` recursively on the sequence of elements in `batch`.

    Args:
        batch: Sequence of elements to apply the function on
        apply_on_batch: Function to apply on the leaf elements
        checks_enabled: If True, makes sure that all elements in `batch` are of the same type and size
        key_path: Path to the current element in the batch. Internal usage only.
        skip_keys: Paths of keys to skip. If a key starts with any value in `skip_keys`, `apply_on_batch`
            will not be called on it and `batch` will be returned unchanged.

    Returns:
        Instance of type `batch[0]`

    Example:
        output = apply_on_dataclasses(
            batch = [
                {"a": torch.tensor([1, 3]), "b": torch.tensor([2, 4])},
                {"a": torch.tensor([5, 7]), "b": torch.tensor([6, 8])},
            ],
            apply_on_batch = lambda key_path, data: torch.stack(data, dim=0),
        )
        # output: {"a": tensor([[1, 3], [5, 7]]), "b": tensor([[2, 4], [6, 8]])}
    """

    # Return batch as is if we want to skip this key. NOTE: We apply before running checks
    # and before returning None because we want to bypass those checks and logic
    for skip_key in skip_keys:
        if key_path.startswith(skip_key):
            return apply_on_batch(key_path, batch)

    if all(element is None for element in batch):
        return None

    # Make sure all elements are of the same type
    if checks_enabled and len(set({type(element) for element in batch})) != 1:
        raise ValueError(
            f"Expected all elements in batch to be of the same type, but got "
            f"types {[type(element) for element in batch]} for {key_path.strip('.')} and elements\n{batch}"
        )

    ElementType = type(batch[0])

    if issubclass(ElementType, (torch.Tensor, np.ndarray)):
        try:
            return apply_on_batch(key_path, batch)  # type: ignore[arg-type]
        except (RuntimeError, ValueError) as exc:
            raise type(exc)(
                f"{exc}\nKey: {key_path}, shapes: {[element.shape for element in batch]}"  # type: ignore
            ) from exc

    key_path = f"{key_path}." if key_path != "" else ""

    if issubclass(ElementType, Dataclass):
        batched_values = {
            key: apply_on_dataclasses(
                [getattr(element, key) for element in batch],  # type: ignore[arg-type]
                apply_on_batch,
                skip_keys=skip_keys,
                checks_enabled=checks_enabled,
                key_path=f"{key_path}{key}",
            )
            for key in ElementType.keys()  # type: ignore[union-attr,call-arg]
        }
        return ElementType(**batched_values)  # type: ignore[misc]

    if issubclass(ElementType, (DataclassDict, dict)):
        # Make sure all elements have the same keys
        if checks_enabled and len(set(tuple(element.keys()) for element in batch)) != 1:  # type: ignore[union-attr]
            raise ValueError(
                f"Expected all elements in batch to have the same keys, but got keys "
                f"{[element.keys() for element in batch]} for {key_path} and elements\n{batch}"  # type: ignore[union-attr]
            )
        return ElementType(
            **{  # type: ignore[arg-type]
                key: apply_on_dataclasses(
                    [element[key] for element in batch],  # type: ignore[index]
                    apply_on_batch,
                    skip_keys=skip_keys,
                    checks_enabled=checks_enabled,
                    key_path=f"{key_path}{key}",
                )
                for key in batch[0].keys()  # type: ignore[union-attr]
            }
        )

    if issubclass(ElementType, (DataclassList, list, tuple)):
        # Make sure all elements have the same size
        if checks_enabled and len(set(len(element) for element in batch)) != 1:  # type: ignore[arg-type]
            raise ValueError(
                f"Expected all elements in batch to be of the same size, but got sizes "
                f"{[len(element) for element in batch]} for {key_path} and elements\n{batch}"  # type: ignore[arg-type]
            )
        return ElementType(
            [
                apply_on_dataclasses(
                    [element[i] for element in batch],  # type: ignore[index]
                    apply_on_batch,
                    skip_keys=skip_keys,
                    checks_enabled=checks_enabled,
                    key_path=f"{key_path}{i}",
                )  # type: ignore
                for i in range(len(batch[0]))  # type: ignore[arg-type]
            ]
        )

    raise TypeError(
        f"Batch must contain items of None, torch.Tensor, np.ndarray, Dataclass, dict or list, "
        f"but found '{key_path.rstrip('.')}' of type {ElementType}"
    )


def collate_elements(
    batch: Sequence[TensorT],
    share_memory: bool = False,
    dim: int = 0,
    skip_keys: Sequence[str] = (),
) -> TensorT:
    """
    Packs batch into a single TensorT, such that torch.Tensor attributes across all elements in `batch`
    are packed together in a single torch.Tensor.

    Requirements:
    - Every element in `batch` is of the same type (if not, unknown exception is raised)
    - torch.Tensor attributes with identical keys are expected to have the same shape across the elements in batch
    - Any dict or list attributes must have the same size or keys

    Args:
    - share_memory: If True, put the resulting torch.Tensor objects in shared memory. Makes a difference only
        if using torch.multiprocessing.Queue to transfer data to another process. False by default, since
        unnecessary usage of shared memory can cause the process to crash
    - dim: Dimension in the tensors to be added and used for collation

    Returns the same type as the elements in `batch`
    """
    if len(batch) == 0:
        raise ValueError("Can't collate an empty sequence")

    return apply_on_dataclasses(
        batch,
        lambda key_path, data: collate_tensors(
            key_path=key_path, batch=data, share_memory=share_memory, dim=dim, skip_keys=skip_keys
        ),
        skip_keys=skip_keys,
    )


def collate_and_pad_elements(
    batch: Sequence[TensorT],
    pads: Dict[str, Dict[str, Any]],
    share_memory: bool = False,
    dim: int = 0,
    skip_keys: Sequence[str] = (),
) -> TensorT:
    """
    Same as collate_elements, but additionally supports padding variable-length sequences when collating
    """
    assert dim == 0, "Padding not yet supported for non-zero dimension"

    if len(batch) == 0:
        raise ValueError("Can't collate an empty sequence")

    return apply_on_dataclasses(
        batch,
        lambda key_path, data: collate_and_pad_tensors(
            key_path=key_path,
            batch=data,
            pads=pads,
            dim=dim,
            share_memory=share_memory,
            skip_keys=skip_keys,
        ),
        skip_keys=skip_keys,
    )


def collate_tensors(
    key_path: str,
    batch: Sequence[torch.Tensor | np.ndarray | Any],
    share_memory: bool = False,
    dim: int = 0,
    skip_keys: Sequence[str] = (),
) -> torch.Tensor | np.ndarray | List[Any]:
    # If skip keys, just return the batch. It's already collated because it's in a list
    for skip_key in skip_keys:
        if key_path.startswith(skip_key):
            return batch

    ElementType = type(batch[0])

    if issubclass(ElementType, torch.Tensor):
        expanded_tensors = [b.unsqueeze(dim) for b in batch]  # type: ignore[union-attr]
        if len(expanded_tensors) == 1:
            out = expanded_tensors[0]  # skip the cat for better performance when possible
        else:
            out = torch.cat(expanded_tensors, dim)
        if share_memory:
            # share_memory_() *should* move to shared memory storage in-place.
            # shared_memory makes a difference only with torch.multiprocessing.Queue
            out = out.share_memory_()
        return out

    if issubclass(ElementType, np.ndarray):
        expanded_arrays = [np.expand_dims(b, axis=dim) for b in batch]
        if len(expanded_arrays) == 1:
            out = expanded_arrays[0]  # skip the cat for better performance when possible
        else:
            out = np.concatenate(expanded_arrays, axis=dim)
        return out

    raise TypeError(f"Unsupported type {key_path}: {ElementType}")


def collate_and_pad_tensors(
    key_path: str,
    batch: Sequence[torch.Tensor | np.ndarray],
    pads: Dict[str, Dict[str, Any]],
    dim: int = 0,
    share_memory: bool = False,
    skip_keys: Sequence[str] = (),
) -> torch.Tensor | np.ndarray | List[Any]:
    # If skip keys, just return the batch. It's already collated because it's in a list
    for skip_key in skip_keys:
        if key_path.startswith(skip_key):
            return batch
    if key_path in pads:
        batch = pad_elements(batch, **pads[key_path])  # Pad the sequence
    return collate_tensors(key_path=key_path, batch=batch, dim=dim, share_memory=share_memory)


def concat_elements(
    batch: Sequence[TensorT],
    dim: int = 0,
    skip_keys: Sequence[str] = (),
) -> TensorT:
    """Concatenates all tensors along the selected dimension"""
    if len(batch) == 1:
        return batch[0]
    if len(batch) == 0:
        raise ValueError("Can't concatenate an empty sequence")

    return apply_on_dataclasses(
        batch,
        lambda key_path, data: concat_tensors(key_path, data, dim=dim, skip_keys=skip_keys),
        skip_keys=skip_keys,
    )


def concat_and_pad_elements(
    batch: Sequence[TensorT],
    pads: Dict[str, Dict[str, Any]],
    dim: int = 0,
    skip_keys: Sequence[str] = (),
) -> TensorT:
    """
    Same as concat_elements, but additionally supports padding variable-length sequences when collating
    """
    assert dim == 0, "Padding not yet supported for non-zero dimension"

    if len(batch) == 1:
        return batch[0]
    if len(batch) == 0:
        raise ValueError("Can't concatenate an empty sequence")

    return apply_on_dataclasses(  # type: ignore[assignment]
        batch,
        lambda key_path, data: concat_and_pad_tensors(
            key_path=key_path, batch=data, pads=pads, dim=dim, skip_keys=skip_keys
        ),
        skip_keys=skip_keys,
    )


def concat_tensors(
    key_path: str,
    batch: Sequence[torch.Tensor | np.ndarray | List[Any]],
    dim: int = 0,
    skip_keys: Sequence[str] = (),
) -> torch.Tensor | np.ndarray:
    if len(batch) == 0:
        raise ValueError("Cannot concatenate an empty sequence")
    if len(batch) == 1:
        return batch[0]

    # If skip keys, just return a flattened list of the batch. We flatten because that's equivalent to concat
    for skip_key in skip_keys:
        if key_path.startswith(skip_key):
            batch = [item for sublist in batch for item in sublist]
            return batch

    ElementType = type(batch[0])
    if issubclass(ElementType, torch.Tensor):
        return torch.cat(batch, dim)  # type: ignore[call-overload]
    if issubclass(ElementType, np.ndarray):
        return np.concatenate(batch, axis=dim)
    raise TypeError(f"Unsupported type: {ElementType}")


def concat_and_pad_tensors(
    key_path: str,
    batch: Sequence[torch.Tensor | np.ndarray | List[Any]],
    pads: Dict[str, Dict[str, Any]],
    dim: int = 0,
    skip_keys: Sequence[str] = (),
) -> torch.Tensor | np.ndarray | List[Any]:
    # If skip keys, just return a flattened list of the batch. We flatten because that's equivalent to concat
    for skip_key in skip_keys:
        if key_path.startswith(skip_key):
            batch = [item for sublist in batch for item in sublist]
            return batch

    if key_path in pads:
        batch = pad_elements(batch, **pads[key_path])  # Pad the sequence
    return concat_tensors(key_path=key_path, batch=batch, dim=dim)


def pad_elements(
    batch: list[torch.Tensor | np.ndarray],
    dim: int | Sequence[int],
    pad_value: Any,
    side: PadSide | str,
) -> list[torch.Tensor | np.ndarray]:
    """
    Pad `batch` such that tensor.shape[dim] matches across all elements.
    Args:
        batch: List of tensors with the same number of dimensions
        dim: Dimensions to pad (positive and negative values accepted)
        pad_value: Value to use for padding
        side: PadSide.RIGHT pads after the existing elements, PadSide.LEFT pads before them
            (left-padding is what autoregressive generation needs for a batch of prompts). The
            strings "right"/"left" are accepted and coerced.
    Returns:
        List[torch.Tensor] with the padded tensors
    """
    side = PadSide(side)
    if len(batch) <= 1:
        return batch

    dim = [dim] if isinstance(dim, int) else dim
    dim = np.array(dim)  # 1-D
    assert dim.ndim == 1, dim.shape
    ndims = batch[0].ndim

    # Make sure all dimensions are positive and sort such that we get them in order
    dim = np.sort((dim + ndims) % ndims)

    if np.any(np.diff(dim) > 1):
        # TODO: Amend the pad argument value to include 2 zeros per missing dimension
        raise NotImplementedError("Non-consecutive dimension padding not yet supported")

    shapes = np.stack([np.array(tensor.shape)[np.array(dim)] for tensor in batch])  # [len(batch), len(dim)]
    max_shape = np.max(shapes, axis=0)  # [len(dim)]
    zeros = np.zeros_like(max_shape)  # [len(dim)]

    if not (is_torch := isinstance(batch[0], torch.Tensor)):
        batch = [torch.from_numpy(tensor) for tensor in batch]

    batch = [
        torch.nn.functional.pad(
            tensor,
            # Per dim torch.pad wants (pad_left, pad_right); right-side padding is (0, amount),
            # left-side is (amount, 0). np.flip: `pad` starts from the last dimension backwards.
            # np.ravel + np.column_stack - interleave padding values and zeros.
            pad=tuple(
                np.flip(
                    np.ravel(
                        np.column_stack(
                            [max_shape - shape, zeros]
                            if side is PadSide.RIGHT
                            else [zeros, max_shape - shape]
                        )
                    )
                )
            ),
            value=pad_value,
        )
        for tensor, shape in zip(batch, shapes, strict=True)
    ]

    if not is_torch:
        batch = [tensor.numpy() for tensor in batch]

    return batch


def split_dataclass(input_data: TensorT, num_splits: int, dim: int = 0) -> Sequence[TensorT]:
    """
    Split all tensors along the selected dimension. Output tensors have the same number of dimensions
    NOTE: Works only when input_data contains only Tensors. Otherwise raises ValueError.
    The input must be a Dataclass that also supports tensor-aware slicing via `__getitem__`
    (TensorDataclass provides this in production).
    """
    assert isinstance(input_data, Dataclass)

    if num_splits == 1:
        return [input_data]

    # Find the size of the dimension
    dim_size = 0
    for _, value in input_data.items(recursive=True):
        if isinstance(value, torch.Tensor):
            dim_size = value.shape[dim]
            break

    # Dataclass is empty, no need to split
    if dim_size == 0:
        return [input_data] * num_splits

    if dim_size % num_splits == 0:
        section_size = dim_size // num_splits
        split_sections = torch.arange(0, dim_size + 1, section_size)
        # raise ValueError(f"Can't split dimension of size {dim_size} into {num_splits} equal chunks")
    else:
        section_size = dim_size // num_splits + 1
        # NOTE: The last index will be outside the dimension size, but torch handles this
        split_sections = torch.arange(0, dim_size + 1, section_size)

    section_size = dim_size // num_splits
    split_sections = torch.arange(0, dim_size + 1, section_size)

    splits = [
        input_data[tuple(slice(None) for _ in range(dim)) + (slice(low, high),)]
        for low, high in zip(split_sections[:-1], split_sections[1:], strict=True)
    ]

    return splits
