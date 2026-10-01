import collections
from typing import Any, Sequence

import numpy as np
import PIL.Image
import torch


class HistoryBuffer:
    def __init__(
        self,
        past_observations_len: int,
        past_observations_stride: float | int | None,
        history_keys: Sequence[str] = (),
    ):
        """
        Args:
            past_observations_len: The number of past observations required
            past_observations_stride: The stride of past observations:
                - int - stride is defined in terms of indices
                - float - stride is defined in terms of seconds of timestamps
                - None - no stride
            history_key: Optional list of keys. If provided, only these keys from `obs` will be saved
                in history. Otherwise, the entire obs is saved in history
        """
        if isinstance(past_observations_stride, int):
            maxlen = (past_observations_len + 1) * past_observations_stride
        elif isinstance(past_observations_stride, float):
            raise NotImplementedError()
        else:
            raise TypeError(f"Unsupported type {type(self.past_observations_stride)}")

        self.maxlen = maxlen
        self.past_observations_len = past_observations_len
        self.past_observations_stride = past_observations_stride
        self.history_keys = history_keys

        self.buffer = collections.deque(maxlen=self.maxlen)
        self.timestamps_buffer = collections.deque(maxlen=self.maxlen)

    def append(self, obs: dict[str, Any], timestamp: float | None = None):
        # Filter by desired history_keys
        if len(self.history_keys) > 0:
            obs = {key: value for key, value in obs.items() if key in self.history_keys}

        self.buffer.append(obs)

        if timestamp is not None:
            self.timestamps_buffer.append(timestamp)
        elif timestamp is None and isinstance(self.past_observations_stride, float):
            raise ValueError(
                f"Timestamps can't be None when using past_observations_stride "
                f"{self.past_observations_stride} of type float"
            )

    def get_obs(self) -> dict[str, Any]:
        # If single observation, just take the latest
        if self.past_observations_len == 1:
            return collate_items([self.buffer[-1]])

        # Stride defined per number of observations
        if isinstance(self.past_observations_stride, int):
            query_indices = np.arange(self.past_observations_len) * self.past_observations_stride
            # Revert indices such they start from the end of the deque
            query_indices = len(self.buffer) - 1 - query_indices
            query_indices = query_indices[::-1]  # Order in increasing timestamp order

        elif isinstance(self.past_observations_stride, float):
            raise NotImplementedError()
            query_timestamps = (
                self.timestamps_buffer[-1]
                - np.arange(self.past_observations_len) * self.past_observations_stride
            )
            query_timestamps = query_timestamps[::-1]  # Order in increasing timestamp order

            indices = np.searchsorted(self.timestamps_buffer, query_timestamps, side='left')
            indices = np.clip(indices, 0, len(self.timestamps_buffer))
            # TODO: Find if the left or right index is closer (nearest neighbour search)

        # elif self.past_observations_stride is None:
        #     return self.buffer[-1]

        else:
            raise TypeError(f"Unsupported type {type(self.past_observations_stride)}")

        # Make sure query indices are in range. At the beginning of episodes, they might not be
        query_indices = np.clip(query_indices, 0, len(self.buffer) - 1)

        obs = [self.buffer[i] for i in query_indices]

        # Collate observations into lists, np.ndarrays or torch.Tensor
        obs = collate_items(obs)

        return obs

    def reset(self):
        self.buffer = collections.deque(maxlen=self.maxlen)
        self.timestamps_buffer = collections.deque(maxlen=self.maxlen)


def collate_items(batch: Sequence[Any], checks_enabled: bool = True, key_path: str = "") -> Any:
    if any(element is None for element in batch):
        return None

    # Make sure all elements are of the same type
    if checks_enabled and len(set({type(element) for element in batch})) != 1:
        raise ValueError(
            f"Expected all elements in batch to be of the same type, but got "
            f"types {[type(element) for element in batch]} for {key_path} and elements\n{batch}"
        )

    ElementType = type(batch[0])

    if issubclass(ElementType, (int, float, str)):
        return batch

    if issubclass(ElementType, PIL.Image.Image):
        batch = [np.asarray(item) for item in batch]

    ElementType = type(batch[0])

    if issubclass(ElementType, (torch.Tensor, np.ndarray)):
        try:
            if issubclass(ElementType, torch.Tensor):
                return torch.stack(batch)
            if issubclass(ElementType, np.ndarray):
                return np.stack(batch)
        except (RuntimeError, ValueError) as exc:
            raise type(exc)(
                f"{exc}\nKey: {key_path}, shapes: {[element.shape for element in batch]}"
            ) from exc

    key_path = f"{key_path}." if key_path != "" else ""

    # if issubclass(ElementType, Dataclass):
    #     batched_values = {
    #         key: apply_on_dataclasses(
    #             [getattr(element, key) for element in batch],
    #             apply_on_batch,
    #             checks_enabled=checks_enabled,
    #             key_path=f"{key_path}{key}",
    #         )
    #         for key in ElementType.keys()
    #     }
    #     return ElementType(**batched_values)

    if issubclass(ElementType, dict):
        # Make sure all elements have the same keys
        if checks_enabled and len(set(tuple(element.keys()) for element in batch)) != 1:
            raise ValueError(
                f"Expected all elements in batch to have the same keys, but got keys "
                f"{[element.keys() for element in batch]} for {key_path} and elements\n{batch}"
            )
        return {
            key: collate_items(
                [element[key] for element in batch],
                checks_enabled=checks_enabled,
                key_path=f"{key_path}{key}",
            )
            for key in batch[0].keys()
        }

    if issubclass(ElementType, (list, tuple)):
        # Make sure all elements have the same size
        if checks_enabled and len(set(len(element) for element in batch)) != 1:
            raise ValueError(
                f"Expected all elements in batch to be of the same size, but got sizes "
                f"{[len(element) for element in batch]} for {key_path} and elements\n{batch}"
            )
        return ElementType(
            [
                collate_items(
                    [element[i] for element in batch],
                    checks_enabled=checks_enabled,
                    key_path=f"{key_path}{i}",
                )
                for i in range(len(batch[0]))
            ]
        )

    raise TypeError(
        f"Batch must contain items of None, torch.Tensor, np.ndarray, Dataclass, dict or list, "
        f"but found {ElementType}"
    )
