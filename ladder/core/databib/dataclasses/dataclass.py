import collections
import dataclasses
import enum
import inspect
from abc import ABCMeta
from typing import Any, Callable, ClassVar, Dict, List, Type, TypeVar, Union

import numpy as np
from frozendict import frozendict

try:
    import torch
except ImportError:
    # Don't explicitly depend on torch, it's optional
    torch = type('', (), {"Tensor": np.ndarray})()

from ladder.core.databib.dataclasses.dataclass_dict import DataclassDict
from ladder.core.databib.dataclasses.dataclass_interface import DataclassInterface
from ladder.core.databib.dataclasses.dataclass_list import DataclassList
from ladder.core.databib.dataclasses.dataclass_utils import maybe_chained_keys_to_nested_dict
from ladder.core.databib.utils.classproperty import cached_classproperty
from ladder.core.databib.utils.typing_utils import (
    ITERABLE_TYPES,
    annotation_is_dict,
    annotation_is_list,
    annotation_is_tuple,
    annotation_is_union,
    get_annotation_type_args,
    get_dict_annotation_type_args,
    get_list_annotation_type_args,
    get_maybe_optional_type,
    get_origin_type,
    value_type_is_annotation_type,
)

DataclassT = TypeVar("DataclassT", bound="Dataclass")


def dataclass(cls: Type[DataclassT], **kwargs) -> Type[DataclassT]:
    cls = dataclasses.dataclass(cls, frozen=True, repr=False, kw_only=True, **kwargs)  # type: ignore[call-overload]
    _prevent_forbidden_type_annotations(cls)
    return cls


def _prevent_forbidden_type_annotations(cls):
    """This cannot be added in a metaclass since __dataclass_fields__ does not exist when the metaclass is called"""
    for field in getattr(cls, "__dataclass_fields__", {}).values():
        field_type = get_maybe_optional_type(field.type)

        # Forbid Union or Tuple with any member of type
        # [DataclassInterface, Dict, List, Set, Tuple, TypeVar, Union, Optional]
        if annotation_is_union(field_type) or annotation_is_tuple(field_type):
            type_args = get_annotation_type_args(field_type)
            for arg in type_args:
                if (
                    (inspect.isclass(arg) and issubclass(arg, DataclassInterface))
                    or get_origin_type(arg) in ITERABLE_TYPES + (Union,)
                    or isinstance(arg, TypeVar)
                ):
                    etype = 'Union' if annotation_is_union(field_type) else 'Tuple'
                    raise TypeError(
                        f"Class {cls} has field {field.name}: {field.type}, but {etype} with members of type "
                        f"DataclassInterface, Dict, List, Set, Tuple, Sequence, Union, Optional, TypeVar "
                        f"are unsafe and not allowed. Remove the Union instead."
                    )
            continue
        if annotation_is_dict(field_type) or annotation_is_list(field_type):
            # Annotation type checks will be run by cls.types below
            continue
        # Forbid non-class type annotations
        if (
            not inspect.isclass(field_type)
            and not isinstance(field_type, dataclasses.InitVar)
            and field_type is not dataclasses.InitVar
            and getattr(field_type, "__origin__", None) is not ClassVar
        ):
            raise TypeError(
                f"Forbidden type annotation in {cls} for field '{field.name}: {field.type}'. "
                f"Expected a class, Optional[class], Dict, List, Tuple or Union"
            )


class DataclassMeta(ABCMeta):
    def __new__(mcs, name, bases, attributes, *args, **kwargs):  # noqa: N804
        new_cls = super().__new__(mcs, name, bases, attributes, *args, **kwargs)
        new_cls._define_dict_and_list_attribute_classes()  # noqa: SLF001
        return new_cls

    def _define_dict_and_list_attribute_classes(cls):  # noqa: N805
        """
        [IMPORTANT]: Suppose we have a dataclass
        ```
        @dataclass
        class MyDataclass(Dataclass):
            my_field: Dict[str, int]
        ```
        The values for my_field will always be cast to `DataclassDict[int]` inside `Dataclass.__init__()`
        However, the type `DataclassDict[int]` doesn't exist before we try to create an instance of
        `MyDataclass`. This is problematic because:
          1. We cannot properly unpickle such classes when using multiprocessing. In particular, the
              class `MyDataclass` will be known to the subprocess since the class will be imported.
              However, `DataclassDict[int]` will be undefined, since the subprocess hasn't created
              an instance of `MyDataclass` before it tries to unpickle it
          2. If the types inside `Dict` are not allowed in `DataclassDict`, error will be raised only
              when we try to create an instance, instead of at class definition time
        To resolve these problems, here we define all the equivalent `DataclassDict` and `DataclassList`
        classes needed for the new dataclass attributes
        """

        # NOTE: At this point, @dataclass hasn't been applied yet, so `__dataclass_fields__` is not the
        # absolute source of truth for dataclass attributes. `__annotations__` gives us the fields added in
        # this class definition, while `__dataclass_fields__` gives us the fields defined by the parents.
        # We check both to also cover any transitive updates on parent class attributes, e.g. if
        # called by `__getitem__`
        annotations = getattr(cls, '__annotations__', {})
        dataclass_fields = getattr(cls, '__dataclass_fields__', {})
        fields = {**annotations, **{field.name: field.type for field in dataclass_fields.values()}}

        # Note that we check only direct attribute types and don't recurse, e.g. into something like
        # `Dict[str, List[str]]`. For now, we care only about the dataclass field type and cast only that
        for field_name, field_type in fields.items():
            if annotation_is_dict(field_type):
                _, DValueT = get_dict_annotation_type_args(field_type)
                try:
                    DataclassDict[DValueT]
                except TypeError as exc:
                    # Provide more informative error message when DValueT not allowed
                    message = f"Cannot cast {cls.__name__}.{field_name} to DataclassDict[{DValueT}].\n{exc}"
                    raise TypeError(message) from exc
            if annotation_is_list(field_type):
                LValueT: Type = get_list_annotation_type_args(field_type)[0]
                try:
                    DataclassList[LValueT]
                except TypeError as exc:
                    # Provide more informative error message when DValueT not allowed
                    message = f"Cannot cast {cls.__name__}.{field_name} to DataclassList[{LValueT}].\n{exc}"
                    raise TypeError(message) from exc


@dataclass
class Dataclass(DataclassInterface, metaclass=DataclassMeta):
    def __post_init__(self):
        self._cast_dict_and_list_attrs()
        self._make_type_checks()

    def _cast_dict_and_list_attrs(self):
        """
        Cast attributes of type Dict or List to their corresponding DataclassDict and DataclassList types
        """
        for field_name, field_type in self._cast_fields.items():
            value = getattr(self, field_name)
            if type(value) is field_type:
                continue
            try:
                object.__setattr__(self, field_name, field_type(value))
            except TypeError:
                # Let self._make_type_checks() raise a more informative error
                pass

    def _make_type_checks(self):
        for field_name, field_type in self.types.items():
            value = getattr(self, field_name)

            if (
                value is None
                and inspect.isclass(field_type)
                and not issubclass(field_type, DataclassInterface)
            ):
                # Here we allow None for any field which is not a subclass of DataclassInterface.
                # Technically, proper type checks should forbid any None value for non-Optional fields,
                # but this is required for Dataclass.make_empty() to work in general.
                # TODO: Figure out how to resolve this predicament in general
                continue

            if not value_type_is_annotation_type(value, field_type):
                raise TypeError(
                    f"Incorrect type provided for {self.__class__.__name__}.{field_name}: "
                    f"expected {self.types[field_name]}, but got value {value} with type {type(value)}"
                )

    @cached_classproperty
    def fields(cls) -> Dict[str, dataclasses.Field]:  # noqa: N805
        # pylint: disable=no-self-argument
        """Returns a frozendict of the Field objects, keyed and sorted by field name"""
        return frozendict({f.name: f for f in sorted(dataclasses.fields(cls), key=lambda x: x.name)})

    @cached_classproperty
    def annotated_types(cls) -> Dict[str, type]:  # noqa: N805
        # pylint: disable=no-self-argument
        return frozendict({name: f.type for name, f in cls.fields.items()})

    @cached_classproperty
    def types(cls) -> Dict[str, type]:  # noqa: N805
        # pylint: disable=no-self-argument
        return frozendict({name: cls._cast_fields.get(name, f.type) for name, f in cls.fields.items()})

    @cached_classproperty
    def _cast_fields(cls) -> Dict[str, Type[DataclassInterface]]:  # noqa: N805
        """
        Returns a frozendict containing only the fields annotated as Dict and List in this `Dataclass`
        and their equivalent `DataclassDict` and `DataclassList` types that values should be cast to.
        Note that DataclassMeta already ensures that all Dict and List annotations can be cast.
        """
        # pylint: disable=no-self-argument
        cast_fields: Dict[str, Type[DataclassInterface]] = {}
        for field in cls.fields.values():
            if annotation_is_dict(field.type):
                _, DValueT = get_dict_annotation_type_args(field.type)
                cast_fields[field.name] = DataclassDict[DValueT]  # type: ignore[valid-type]
            if annotation_is_list(field.type):
                LValueT: Type = get_list_annotation_type_args(field.type)[0]
                cast_fields[field.name] = DataclassList[LValueT]  # type: ignore[valid-type]
        return frozendict(cast_fields)

    @classmethod
    def keys(cls) -> List[str]:
        return list(cls.fields)

    @classmethod
    def from_subclass(cls: Type[DataclassT], subclass_instance: DataclassT) -> DataclassT:
        """Returns an instance of DataclassT, with all subclass properties removed"""
        if not isinstance(subclass_instance, cls):
            raise TypeError(
                f"from_subclass expected an instance of (a subclass of) {cls.__name__}, "
                f"but got {subclass_instance.__class__.__name__}"
            )
        return cls.from_json(
            subclass_instance.as_json(recursive=False), recursive=False, ignore_redundant=True
        )

    @classmethod
    def from_json(cls, data: Dict[str, Any], recursive: bool = True, ignore_redundant: bool = False):
        """Parses a dict to dataclass object"""
        if not isinstance(data, collections.abc.Mapping):
            raise TypeError(
                f"{cls}.from_json() expects input of type dict, but got input of type {type(data)}"
            )
        data = dict(data)

        if ignore_redundant:
            data = cls._ignore_redundant_keys(data)

        if recursive:
            for key, value in data.items():
                field_type = cls.types.get(key, None)
                if field_type is None:
                    # This key is not part of the dataclass fields. Leave it as is.
                    # Dataclass.__init__() will raise error
                    continue
                field_type = get_maybe_optional_type(field_type)
                # If the field annotation is not a class, let the __new__ method deal with it
                if not inspect.isclass(field_type):
                    continue
                # If field is an Enum, convert it to the right type
                if issubclass(field_type, enum.Enum):
                    data[key] = field_type(value)
                if issubclass(field_type, DataclassInterface) and value is not None:
                    data[key] = field_type.from_json(
                        value, recursive=recursive, ignore_redundant=ignore_redundant
                    )

        return cls(**data)

    @classmethod
    def _ignore_redundant_keys(cls, data: dict) -> dict:
        return {key: value for key, value in data.items() if key in cls.types}

    @classmethod
    def from_dict(cls, d: dict, recursive: bool = True, ignore_redundant: bool = False):
        return cls.from_json(d, recursive, ignore_redundant)

    @classmethod
    def make_empty(cls: Type[DataclassT]) -> DataclassT:
        return cls(
            **{
                k: v.make_empty() if inspect.isclass(v) and issubclass(v, DataclassInterface) else None
                for k, v in cls.types.items()
            }
        )

    @property
    def is_empty(self) -> bool:
        for value in self.values():
            if isinstance(value, DataclassInterface):
                if not value.is_empty:
                    return False
            elif value is not None:
                return False
        return True

    def as_json(self, recursive: bool = True) -> dict:
        return {
            k: v.as_json() if isinstance(v, DataclassInterface) and recursive else v for k, v in self.items()
        }

    def as_dict(self, recursive: bool = True) -> dict:
        # cannot use the inbuilt asdict methods as they do a deepcopy of tensors
        return self.as_json(recursive=recursive)

    def values(self):
        return [getattr(self, name) for name in self.fields]

    def items(self, recursive: bool = False):
        for key, value in zip(self.keys(), self.values(), strict=True):
            if recursive and isinstance(value, DataclassInterface):
                for subkey, subvalue in value.items(recursive=True):
                    yield f"{key}.{subkey}", subvalue
            else:
                yield key, value

    def getattr(self, name: str) -> Any:
        """
        Recursive counterpart of the builtin getattr(): `name` can address an attribute of a nested
        dataclass via dot-separation, e.g. 'nested_field.attr'
        """
        value = self
        for attr in name.split("."):
            # Method names are not in scope inside the method body, so this is the builtin getattr()
            value = getattr(value, attr)
        return value

    def replace(self, **kwargs):
        """
        Return a new instance of Dataclass with the kwargs overwritten.
        """
        kwargs = maybe_chained_keys_to_nested_dict(kwargs)

        data = self.as_dict(recursive=False)

        for key, value in kwargs.items():
            if key not in self.types:
                raise KeyError(f"Dataclass {self.__class__} does not have a field {key}")

            value_type = self.types[key]
            value_type = get_maybe_optional_type(value_type)

            # Recurse if a DataclassInterface field
            if inspect.isclass(value_type) and issubclass(value_type, DataclassInterface):
                if isinstance(value, dict):
                    data[key] = data[key].replace(**value)
                else:
                    # If the type is incorrect, delegate to __init__ to handle it
                    data[key] = value
            else:
                data[key] = value

        return self.__class__(**data)

    def apply(self, fcn: Callable, recursive: bool = True, skip_nones: bool = False):
        def fcn_wrapper(value: Any) -> Any:
            if value is None and skip_nones:
                return None
            if isinstance(value, DataclassInterface) and recursive:
                return value.apply(fcn, recursive=True, skip_nones=skip_nones)
            return fcn(value)

        return self.__class__(**{key: fcn_wrapper(value) for key, value in self.items()})

    def __iter__(self):
        for name in self.fields:
            field_value = getattr(self, name)
            if isinstance(field_value, DataclassDict):
                for value in field_value.values():
                    yield value
            elif isinstance(field_value, DataclassInterface):
                for value in field_value:
                    yield value
            else:
                yield field_value

    def shape_repr(self, indent: int = 0, skip_empty=True) -> str:
        DEFAULT_INDENT = 2
        indent_str = " " * indent
        field_indent_str = " " * (indent + DEFAULT_INDENT)
        shape_strings = []

        if self.is_empty and skip_empty:
            return f"{indent_str}{self.__class__.__name__}.empty()"

        for key, value in self.items():
            if value is None:
                shape_strings.append(f"{field_indent_str}{key}: None")
            elif isinstance(value, Dataclass):
                dataclass_repr = value.shape_repr(indent + DEFAULT_INDENT, skip_empty).strip()
                shape_strings.append(f"{field_indent_str}{key}: {dataclass_repr}")
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                shape_strings.append(f"{field_indent_str}{key}: {list(value.shape)}")
            else:
                shape_strings.append(f"{field_indent_str}{key}: -X-")

        return f"{indent_str}{self.__class__.__name__}(\n" + ",\n".join(shape_strings) + f",\n{indent_str})"
