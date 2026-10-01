from typing import Sequence, Tuple, Type, Union

import numpy as np


def verify_np_array(
    array: np.ndarray, shape: Sequence[int], dtype: Union[Type[np.generic], Sequence[Type[np.generic]]]
):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected instance of np.ndarray, but got instance of {type(array)}")

    if array.ndim != len(shape):
        raise ValueError(f"Expected ndim={len(shape)}, but got {array.ndim}")

    if any(
        dim_size != edim_size
        for dim_size, edim_size in zip(array.shape, shape, strict=True)
        if edim_size != -1
    ):
        raise ValueError(f"Expected shape {shape}, but got {array.shape}")

    if not isinstance(dtype, Sequence):
        dtype = [dtype]

    if not any(_dtype_matches(array.dtype, dt) for dt in dtype):
        raise TypeError(f"Expected dtype out of {dtype}, but got dtype {array.dtype}")


def _dtype_matches(actual: np.dtype, expected: Type[np.generic]) -> bool:
    expected_dtype = np.dtype(expected)
    # If the expected dtype is flexible length, e.g. `np.str_` or `np.unicode_`, then only check the
    # kind of the actual dtype, not the itemsize. If the expected dtype is fixed length, e.g.
    # `np.dtype('U7')`, then check both the kind and the itemsize.
    if expected_dtype.kind in 'USV' and expected_dtype.itemsize == 0:
        return actual.kind == expected_dtype.kind
    return actual == expected_dtype


def np_unique(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unique elements in data and corresponding indices.

    np.unique returns the values in a sorted order, even if the source is not sorted. Thus, if you simply
    run np.unique on unsorted data, the indices you will get will be invalid.
    """

    # `data` might not be sorted. Thus we need to apply np.unique twice to get indices w.r.t. original array.
    # Example:
    #   data: [1, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4]
    #   indices: [5, 0, 3, 9]
    #   inverse: [1, 1, 1, 2, 2, 0, 0, 0, 0, 3, 3]
    #   indices_of_first_occurence: [0, 3, 5, 9]
    #   inverse_indices: [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]
    #   counts: [3, 2, 4, 2]
    #   unique_ids: [1, 2, 0, 4]

    _, indices, inverse = np.unique(data, return_index=True, return_inverse=True)

    # Re-adjust such that the order matches the order in the source
    _, indices_of_first_occurence, inverse_indices, counts = np.unique(
        indices[inverse], return_index=True, return_inverse=True, return_counts=True
    )

    unique_ids = data[indices_of_first_occurence]

    return unique_ids, indices_of_first_occurence, inverse_indices, counts


def np_nearest_neighbour_indices(values: np.ndarray, queries: np.ndarray, side: str = 'left') -> np.ndarray:
    """
    Find the indices of the nearest neighbours of `queries` inside `values`
    Args:
        values: 1D *sorted* array in ascending order
        queries: 1D array of queries for which to find nearest neighbour indices inside `values`
        side: 'left' or 'right'
            left: selects the left nearest neighbour if query is equidistant from its two neighbours
            right: selects the right nearest neighbour if query is equidistant from its two neighbours
    Returns:
        1D array of same size as queries containing the indices of the nearest neighbour in `values`
    """
    # Clip the insert indices to the range of the values array
    insert_indices = np.clip(np.searchsorted(values, queries), 0, len(values) - 1)

    if side == 'left':
        nearest_neighbour_indices = np.where(
            np.abs(values[insert_indices] - queries) < np.abs(values[insert_indices - 1] - queries),
            insert_indices,
            insert_indices - 1,
        )
    elif side == 'right':
        nearest_neighbour_indices = np.where(
            np.abs(values[insert_indices] - queries) <= np.abs(values[insert_indices - 1] - queries),
            insert_indices,
            insert_indices - 1,
        )
    else:
        raise ValueError(f"Invalid side: {side}")

    nearest_neighbour_indices = np.clip(nearest_neighbour_indices, 0, len(values) - 1)

    return nearest_neighbour_indices


def np_ranges(lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """
    Create a single array of ranges between lows and highs.
    Example:
        lows = [1, 4, 8]
        highs = [3, 5, 11]
        output -> [1, 2, 4, 8, 9, 10]
    Args:
        lows: Sorted 1D array of start values
        highs: Sorted 1D array of end values
    Returns:
        1D array of the ranges between lows and highs
    """
    assert lows.ndim == highs.ndim == 1, f"{lows.ndim}, {highs.ndim}"
    assert lows.shape == highs.shape, f"{lows.shape}, {highs.shape}"
    assert np.all(highs - lows > 0), f"{highs}, {lows}"

    if lows.size == 0:
        return np.array([], dtype=np.int64)

    # Calculate the length of each range
    lengths = highs - lows
    total_size = np.sum(lengths)

    # Create an array of repeated start values
    starts = np.repeat(lows, lengths)  # [total_size]

    # Create an array of repeated sizes of the previous range
    range_sizes = np.repeat(np.cumsum(np.concatenate([[0], lengths[:-1]])), lengths)

    # Create an array of offsets for each range
    offsets = np.arange(total_size) - range_sizes

    # Add the offsets to starts to get the output array
    output = starts + offsets

    return output
