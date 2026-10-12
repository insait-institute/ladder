from typing import Any, Dict, Iterator, Tuple


def zip_dataclass_items(*dataclasses, recursive: bool) -> Iterator[Tuple[str, Tuple[Any, ...]]]:
    if len(set(type(dataclass) for dataclass in dataclasses)) > 1:
        raise TypeError(
            f"Cannot zip dataclasses with non-matching types: {[type(dataclass) for dataclass in dataclasses]}"
        )

    zip_iterator = zip(*[dataclass.items(recursive=recursive) for dataclass in dataclasses], strict=True)
    tuples_of_items: Tuple[Tuple[str, Any], ...]
    for tuples_of_items in zip_iterator:
        items: Tuple[Any, ...] = sum(tuples_of_items, ())
        keys: Tuple[str, ...] = items[0::2]
        if len(set(keys)) > 1:
            raise ValueError(f"Dataclass keys do NOT match: {keys}")
        key: str = keys[0]
        values: Tuple[Any, ...] = items[1::2]
        yield key, values


def maybe_chained_keys_to_nested_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a dict with keys of the form "key1.key2.key3" to a nested dict"""
    unpacked_data: Dict[str, Any] = {}
    for key, value in data.items():
        if "." not in key:
            unpacked_data = merge_dicts_recursive(unpacked_data, {key: value})
        else:
            mainkey, subkey = key.split(".", maxsplit=1)
            nested_value = maybe_chained_keys_to_nested_dict({subkey: value})
            unpacked_data = merge_dicts_recursive(unpacked_data, {mainkey: nested_value})
    return unpacked_data


def merge_dicts_recursive(dict_1: Dict[str, Any], dict_2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges dict_1 with dict_2 recursively.
    Handles clashing keys:
        1. If both values are dicts, merges them recursively
        2. If any value is not a dict, raises ValueError
    """
    merged = dict(dict_1)

    for key, value in dict_2.items():
        if key in merged:
            if not type(merged[key]) is type(value) is dict:
                raise ValueError(f"Multiple values provided for key {key}: {merged[key]} and {value}")
            merged[key] = merge_dicts_recursive(merged[key], value)
        else:
            merged[key] = value

    return merged
