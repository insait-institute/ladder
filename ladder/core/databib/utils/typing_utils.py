import collections
import enum
import inspect
import types
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

ITERABLE_TYPES: Tuple[Type, ...] = (dict, list, set, tuple, Sequence)


def annotation_is_union(type_value: Type) -> bool:
    return getattr(type_value, '__origin__', None) is Union or type(type_value) is types.UnionType


def annotation_is_optional(type_value: Type) -> bool:
    if annotation_is_union(type_value):
        union_args = set(type_value.__args__)
        if len(union_args) == 2 and type(None) in union_args:
            return True
    return False


def get_maybe_optional_type(type_value: Type[Optional[Any]]) -> Type[Any]:
    if annotation_is_optional(type_value):
        type_args = get_annotation_type_args(type_value)
        if type_args[1] is type(None):
            return type_args[0]
        return type_args[1]
    return type_value


def annotation_is_dict_list_tuple(type_value: Type) -> bool:
    return get_origin_type(type_value) in (dict, tuple, list, Sequence)


def annotation_is_dict(type_value: Type) -> bool:
    return get_origin_type(type_value) is dict


def annotation_is_list(type_value: Type) -> bool:
    return get_origin_type(type_value) is list


def annotation_is_tuple(type_value: Type) -> bool:
    return get_origin_type(type_value) is tuple


def annotation_is_enum(type_value: Type) -> bool:
    return inspect.isclass(type_value) and issubclass(type_value, enum.Enum)


def get_origin_type(type_value: Type) -> Optional[Type]:
    return getattr(type_value, '__origin__', type_value)


def get_annotation_type_args(type_value: Type) -> Tuple[type, ...]:
    return type_value.__args__


def type_annotation_contains_type_var(type_value: Type) -> bool:
    if isinstance(type_value, TypeVar):
        return True
    type_args = getattr(type_value, '__args__', [])
    if type_args is None:
        return False
    return any(type_annotation_contains_type_var(type_arg) for type_arg in type_args)


def get_dict_annotation_type_args(type_value: Type) -> Tuple[type, ...]:
    type_args = get_annotation_type_args(type_value)
    if len(type_args) != 2:
        raise TypeError(
            f"Dict key and value type annotations required, but got {type_value}. "
            f"Scroll up the stack trace to find the offending class definition."
        )
    if type_args[0] is not str:
        raise TypeError(
            f"Only Dict with str key type annotations supported, but got {type_value}"
            f"Scroll up the stack trace to find the offending class definition."
        )
    if Ellipsis in type_args:
        raise TypeError(f"Ellipsis `...` type annotation not allowed in Dict, but got {type_value}")
    return type_args


def get_list_annotation_type_args(type_value: Type) -> Tuple[type, ...]:
    type_args = get_annotation_type_args(type_value)
    if len(type_args) != 1:
        raise TypeError(
            f"List items type annotations required, but got {type_value}. "
            f"Scroll up the stack trace to find the offending class definition."
        )
    if Ellipsis in type_args:
        raise TypeError(f"Ellipsis `...` type annotation not allowed in List, but got {type_value}")
    return type_args


def maybe_update_type_annotation(type_value: Any, update_map: Dict) -> Any:
    """
    Recursively traverse a type annotation and update any type in update_map with its corresponding value
    Supported 'recursive' types: Dict, List, Tuple, Optional, Union
    """
    if type_value in update_map:
        return update_map[type_value]
    if annotation_is_dict(type_value):
        type_args = get_dict_annotation_type_args(type_value)
        return Dict[str, maybe_update_type_annotation(type_args[1], update_map)]  # type: ignore[misc]
    if annotation_is_list(type_value):
        type_args = get_list_annotation_type_args(type_value)
        return List[maybe_update_type_annotation(type_args[0], update_map)]  # type: ignore[misc]
    if annotation_is_tuple(type_value):
        type_args = get_annotation_type_args(type_value)
        return Tuple[tuple(maybe_update_type_annotation(type_arg, update_map) for type_arg in type_args)]
    if annotation_is_union(type_value):  # Note that Union case also covers Optional case
        type_args = get_annotation_type_args(type_value)
        # Rebuild through `Union[...]` rather than `copy_with`, which does not exist on `int | str` unions
        return Union[tuple(maybe_update_type_annotation(type_arg, update_map) for type_arg in type_args)]
    return type_value


def value_type_is_annotation_type(  # pylint: disable=too-many-return-statements  # noqa: PLR0911
    value: Any,
    annotation_type: Type,
    check_function: Callable[[Any, Type], bool] = lambda value, annotation_type: type(value)
    is annotation_type,
) -> bool:
    """
    Check if value complies with annotation_type. Perform exact checks instead of isinstance checks,
    but this can be changed by passing check_function

    Supported typing module constructs: Any, Dict, List, Tuple, Optional, Union

    Args:
        value: The value to test
        annotation_type: The type provided in the annotation, e.g. Any, Dict[...], List[...], etc
        check_function: Callable that takes value_type and annotation_type and return True / False if the
            types match. Will be called only when annotation_type is an actual class. Will not be called
            when annotation_type is a container (e.g. Dict, List, Tuple, Optional, etc). Allows for
            overriding checks, e.g. using isinstance instead of exact match
    Returns:
        True if type(value) is the annotation type, False otherwise
    Raises:
        TypeError if forbidden annotations, e.g. dict, list, Ellipsis inside Dict or List
    """

    # Forbidden type annotations. Useful for demanding specificity. Forbid ambiguous annotations
    # such as `dict` or `list` and require specific type arguments. Note that patterns like
    # dict[str, int], Dict[str, int], list[int], List[int] are allowed and required
    forbidden_types: Tuple[Type, ...] = (dict, list)

    # Optional
    if annotation_is_optional(annotation_type) and value is None:
        return True

    annotation_type = get_maybe_optional_type(annotation_type)
    value_type = type(value)

    if annotation_type in forbidden_types:
        raise TypeError(f"Type annotation {annotation_type} not allowed. Please be more explicit")

    # Any
    if annotation_type is Any:
        return True

    # annotation_type is an actual type -> we can check for correctness
    if inspect.isclass(annotation_type):
        # Allow setting float fields with int - python handles this type conversion automatically
        if value_type is int and annotation_type is float:
            return True
        # Allow for np numeric types (e.g. np.float64, np.int64) to be seen as native python types
        if isinstance(value, np.number) and type(value.item()) is annotation_type:
            return True

        # return value_type is annotation_type
        return check_function(value, annotation_type)

    # Union (recursive)
    if annotation_is_union(annotation_type):
        union_args = get_annotation_type_args(annotation_type)
        return any(
            value_type_is_annotation_type(value, union_type_arg, check_function)
            for union_type_arg in union_args
        )

    # Dict (recursive)
    if annotation_is_dict(annotation_type):
        # Check for Mapping instead of dict. Many dict-like containers aren't actually instances of dict.
        # Ex: frozendict, DataclassDict
        if not isinstance(value, collections.abc.Mapping):
            return False

        type_args = get_dict_annotation_type_args(annotation_type)

        return all(
            value_type_is_annotation_type(k, type_args[0], check_function) for k in value.keys()
        ) and all(value_type_is_annotation_type(v, type_args[1], check_function) for v in value.values())

    # List (recursive)
    if annotation_is_list(annotation_type):
        # We check for Sequence instead of list. Many list-like containers aren't actually instances of list
        # Ex: tuple, DataclassList
        if not isinstance(value, collections.abc.Sequence):
            return False

        type_args = get_list_annotation_type_args(annotation_type)

        return all(value_type_is_annotation_type(v, type_args[0], check_function) for v in value)

    # Tuple (recursive)
    if annotation_is_tuple(annotation_type):
        if not isinstance(value, tuple):
            return False

        type_args = get_annotation_type_args(annotation_type)

        if Ellipsis in type_args:
            if len(type_args) != 2 or type_args[0] is Ellipsis:
                raise TypeError(
                    f"Tuple with Ellipsis `...` type annotation must follow "
                    f"the pattern Tuple[Type, ...], but got {annotation_type}"
                )
            return all(value_type_is_annotation_type(v, type_args[0], check_function) for v in value)

        if len(type_args) != len(value):
            return False

        return all(
            value_type_is_annotation_type(v, type_arg, check_function)
            for v, type_arg in zip(value, type_args, strict=True)
        )

    raise TypeError(f"Unsupported value {value} and annotation type {annotation_type}")


def maybe_autocast_native_types(value: Any, annotation_type: Type) -> Any:
    """
    Best-effort, lossless coercion of native Python values to match a simple annotation:
      - an int provided for a float field is cast to float
      - a float provided for an int field is cast to int only when it has no fractional part (e.g. 5.0)
      - a list provided for a Tuple field is cast to tuple (YAML / JSON have no tuple literal), recursively
        coercing each element to its corresponding tuple element type
    Anything else is returned unchanged so that a subsequent type check can still reject real mismatches
    (e.g. a non-integral float for an int field, or a wrong-length tuple). Note that bool is intentionally
    never cast: although it subclasses int, passing a bool to an int / float field is almost always a mistake.
    """
    # list -> Tuple. Only lists (ordered) are cast - casting an unordered set to a tuple makes no sense.
    if annotation_is_tuple(annotation_type) and isinstance(value, list):
        type_args = get_annotation_type_args(annotation_type)
        if Ellipsis in type_args:
            # Tuple[T, ...]: every element is coerced to T
            return tuple(maybe_autocast_native_types(item, type_args[0]) for item in value)
        if len(type_args) == len(value):
            return tuple(
                maybe_autocast_native_types(item, arg) for item, arg in zip(value, type_args, strict=True)
            )
        # Length mismatch: still cast to tuple so the type check reports the real (length) error
        return tuple(value)

    if type(value) is bool or not inspect.isclass(annotation_type):
        return value
    if annotation_type is float and type(value) is int:
        return float(value)
    if annotation_type is int and type(value) is float and value.is_integer():
        return int(value)
    return value


def cast_container_types(value: Any, annotation_type: Type) -> Any:
    """
    Maybe cast the entries inside `value` to annotation_type. Casting is performed out-of-place
    and only for containers of type Dict, List, Tuple. All other types remain unchanged.

    Supported typing annotation constructs: Any, Dict, List, Tuple, Optional, Union

    Args:
        value: The actual value
        annotation_type: The type provided in the annotation, e.g. Any, Dict[...], List[...], etc
    Returns:
        value or its casted equivalent
    Raises:
        TypeError if forbidden annotations, e.g. dict, list, Ellipsis inside Dict or List
    """

    # Forbidden type annotations. Useful for demanding specificity. Forbid ambibuous annotations
    # such as `dict` or `list` and require specific type arguments. Note that patterns like
    # dict[str, int], Dict[str, int], list[int], List[int] are allowed and required
    forbidden_types: Tuple[Type, ...] = (dict, list)

    # Optional
    if annotation_is_optional(annotation_type) and value is None:
        return value

    annotation_type = get_maybe_optional_type(annotation_type)

    if annotation_type in forbidden_types:
        raise TypeError(f"Type annotation {annotation_type} not allowed. Please be more explicit")

    # Any
    if annotation_type is Any:
        return value

    # annotation_type is an actual type -> we can check for correctness
    if inspect.isclass(annotation_type):
        from ladder.core.databib.dataclasses.dataclass_dict import DataclassDict
        from ladder.core.databib.dataclasses.dataclass_list import DataclassList

        # Cast value to the exact annotation_type so nested containers are wrapped recursively
        # NOTE: We check only for DataclassList/DataclassDict (and their Config variants). Checking
        # for DataclassInterface also caputes Config, where annotations might be AutoConfig, which
        # means value might already have the resolved type, but annotation_type is still AutoConfig
        if issubclass(annotation_type, (DataclassList, DataclassDict)) and (
            type(value) is not annotation_type
        ):
            return annotation_type(value)
        return value

    # Union (recursive)
    if annotation_is_union(annotation_type):
        return value
        # if any(
        #     value_type_is_annotation_type(value, union_type_arg, check_function)
        #     for union_type_arg in union_args
        # ):
        #     return value
        # raise TypeError(f"{value} of type {type(value)} not part of Union args: {annotation_type}")

    # Dict (recursive)
    if annotation_is_dict(annotation_type):
        # Check for Mapping instead of dict. Many dict-like containers aren't actually instances of dict.
        # Ex: frozendict, DataclassDict
        if not isinstance(value, collections.abc.Mapping):
            raise TypeError(f"Expected a Mapping, but got {value} of type {type(value)}")

        type_args = get_dict_annotation_type_args(annotation_type)

        return {
            # NOTE: Don't cast key types - they must be str every time
            k: cast_container_types(v, type_args[1])
            for k, v in value.items()
        }

    # List (recursive)
    if annotation_is_list(annotation_type):
        # We check for Sequence instead of list. Many list-like containers aren't actually instances of list
        # Ex: tuple, DataclassList
        if not isinstance(value, collections.abc.Sequence):
            raise TypeError(f"Expected a Sequence, but got {value} of type {type(value)}")

        type_args = get_list_annotation_type_args(annotation_type)

        return [cast_container_types(v, type_args[0]) for v in value]

    # Tuple (recursive)
    if annotation_is_tuple(annotation_type):
        if not isinstance(value, collections.abc.Sequence):
            raise TypeError(f"Expected a Sequence, but got {value} of type {type(value)}")

        type_args = get_annotation_type_args(annotation_type)

        if Ellipsis in type_args:
            if len(type_args) != 2 or type_args[0] is Ellipsis:
                raise TypeError(
                    f"Tuple with Ellipsis `...` type annotation must follow "
                    f"the pattern Tuple[Type, ...], but got {annotation_type}"
                )
            return tuple(cast_container_types(v, type_args[0]) for v in value)

        if len(type_args) != len(value):
            raise TypeError(
                f"Expected a Sequence of len {len(type_args)} for type {annotation_type}, "
                f"but got {value} of len {len(value)}"
            )

        return tuple(cast_container_types(v, type_arg) for v, type_arg in zip(value, type_args, strict=True))

    raise TypeError(f"Unsupported value {value} and annotation type {annotation_type}")


def substitute_annotation_container_types(
    annotation_type: Type, ListType: Type, DictType: Type  # noqa: N803
) -> Type:
    """
    Recursively rewrite a type annotation, replacing every List with `ListType` and every Dict with
    `DictType` (e.g. DataclassList / DataclassDict or ConfigList / ConfigDict). Tuple, Optional and
    Union are preserved with their members rewritten. Leaves (classes, Any, TypeVar) are unchanged.

    Supported typing constructs: Any, Dict, List, Tuple, Optional, Union
    """
    if annotation_is_dict(annotation_type):
        _, value_type = get_dict_annotation_type_args(annotation_type)
        return DictType[substitute_annotation_container_types(value_type, ListType, DictType)]
    if annotation_is_list(annotation_type):
        (value_type,) = get_list_annotation_type_args(annotation_type)
        return ListType[substitute_annotation_container_types(value_type, ListType, DictType)]
    if annotation_is_tuple(annotation_type) or annotation_is_union(annotation_type):
        args = get_annotation_type_args(annotation_type)
        new_args = tuple(
            arg if arg is Ellipsis else substitute_annotation_container_types(arg, ListType, DictType)
            for arg in args
        )
        # Preserve identity when nothing nested changed, so callers can cheaply detect a no-op via `is`.
        if all(new_arg is old_arg for new_arg, old_arg in zip(new_args, args, strict=True)):
            return annotation_type
        return annotation_type.copy_with(new_args)
    return annotation_type
