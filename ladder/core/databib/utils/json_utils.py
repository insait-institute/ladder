import dataclasses
import enum
import json
import math
import numbers
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Iterable, Union, get_args

import numpy as np

try:
    import torch
except (ImportError, OSError):
    # Don't explicitly depend on torch as it's a big one
    torch = None

from ladder.core.databib.dataclasses import Dataclass, DataclassDict, DataclassList
from ladder.core.databib.utils.date_utils import format_datetime, parse_datetime
from ladder.core.databib.utils.typing_utils import get_origin_type

JSON_SAFE_TYPES = (int, float, str)


def is_nan(obj):
    # 'nan' test
    # Not using math.isnan() since we will have to split the JSON_SAFE_TYPES
    # to (int, float) and (str). This is because nan is an instance of int/float
    # and not str. Using math.isnan() before safe_types results in failure for str instances
    # pylint: disable=R0124
    return obj != obj


def is_inf(obj):
    try:
        if math.isinf(obj):
            return True
    except TypeError:
        return False
    return False


def to_json(obj, safe_types=JSON_SAFE_TYPES) -> str:
    return json.dumps(to_json_serialisable(obj, safe_types=safe_types))


def to_json_serialisable(obj, safe_types=JSON_SAFE_TYPES):
    # pylint: disable=too-many-return-statements
    if obj is None:
        return obj

    # Dump numpy arrays and torch tensors
    if hasattr(obj, 'tolist'):
        return obj.tolist()

    if isinstance(obj, np.number):
        return obj.item()

    if is_nan(obj):
        return 'NaN'

    if is_inf(obj):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, safe_types):
        return obj

    if hasattr(obj, 'to_json'):
        return obj.to_json()

    if isinstance(obj, datetime):
        return format_datetime(obj)

    if isinstance(obj, (tuple, list)):
        return [to_json_serialisable(v, safe_types=safe_types) for v in obj]

    if isinstance(obj, enum.Enum):
        return obj.name

    if isinstance(obj, Dataclass):
        obj = obj.as_json()

    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)

    if isinstance(obj, Mapping):
        return {
            to_json_serialisable(k): to_json_serialisable(v, safe_types=safe_types) for k, v in obj.items()
        }

    return {
        to_json_serialisable(k): to_json_serialisable(getattr(obj, k), safe_types=safe_types)
        for k in dir(obj)
        if not k.startswith('_')
    }


def from_json(obj, cls: type, safe_types=JSON_SAFE_TYPES) -> str:
    return from_json_serialisable(json.loads(obj), cls, safe_types)


def from_json_serialisable(json_val: Any, cls: type, safe_types=JSON_SAFE_TYPES):
    # pylint: disable=too-many-return-statements
    if cls == type(None) and json_val is None:  # noqa
        return None

    origin_type = get_origin_type(cls)

    if origin_type == Union:
        # type is a union
        for union_cls in get_args(cls):
            try:
                return from_json_serialisable(json_val, union_cls, safe_types)
            except ValueError:
                # test the next union type
                pass

    if origin_type == dict:
        key_t, val_t, *_ = tuple(get_args(cls))
        return {
            from_json_serialisable(key, key_t, safe_types): from_json_serialisable(val, val_t, safe_types)
            for key, val in json_val.items()
        }

    if origin_type in (list, tuple, set):
        val_t, *_ = get_args(cls)
        vals = [from_json_serialisable(val, val_t, safe_types) for val in json_val]
        return origin_type(vals)

    if issubclass(cls, safe_types) and isinstance(json_val, safe_types):
        return cls(json_val)

    # Note: we support torch not being installed as we have compat issues in standard docker images
    if torch is not None and issubclass(cls, torch.Tensor) and isinstance(json_val, Iterable):
        return cls(json_val)

    if issubclass(cls, np.ndarray) and isinstance(json_val, Iterable):
        return np.array(json_val)

    if issubclass(cls, np.number) and isinstance(json_val, numbers.Number):
        return cls(json_val)

    if issubclass(cls, enum.Enum):
        return cls(json_val)

    if issubclass(cls, Dataclass) and isinstance(json_val, dict):
        d = {
            field.name: from_json_serialisable(json_val[field.name], field.type, safe_types)
            for field in cls.fields.values()  # noqa
            if field.name in json_val
        }
        return cls.make_empty().replace(**d)  # make_empty() to  support forward compatibility

    if issubclass(cls, DataclassList) and isinstance(json_val, Iterable):
        return DataclassList[cls.ValueT](  # type: ignore
            [from_json_serialisable(val, cls.ValueT, safe_types) for val in json_val]  # type: ignore
        )

    if issubclass(cls, DataclassDict) and isinstance(json_val, Mapping):
        return DataclassDict[cls.KeyT, cls.ValueT](  # type: ignore
            {
                from_json_serialisable(key, cls.KeyT, safe_types): (  # type: ignore
                    from_json_serialisable(val, cls.ValueT, safe_types)  # type: ignore
                )
                for key, val in json_val.items()
            }
        )

    if isinstance(json_val, str) and issubclass(cls, datetime):
        return parse_datetime(json_val)

    raise ValueError(f'Unable to deserialize value {json_val} to type {cls}')
