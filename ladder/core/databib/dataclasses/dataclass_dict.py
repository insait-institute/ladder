import inspect
import sys
from typing import Any, Callable, Dict, Tuple, Type, TypeVar

from frozendict import frozendict

from ladder.core.databib.dataclasses.dataclass_interface import DataclassInterface
from ladder.core.databib.template import Template, TemplateMeta
from ladder.core.databib.utils.typing_utils import (
    cast_container_types,
    get_maybe_optional_type,
    substitute_annotation_container_types,
    type_annotation_contains_type_var,
    value_type_is_annotation_type,
)

ValueT = TypeVar("ValueT")
DataclassDictT = TypeVar("DataclassDictT", bound="DataclassDict")


class DataclassDictMeta(TemplateMeta):
    def __getitem__(cls, params):  # noqa: N805
        # Substitute nested List / Dict in the value type with their DataclassList / DataclassDict
        # equivalents (ConfigList / ConfigDict for the Config variants)
        ListType, DictType = cls._container_classes()
        new_cls = super().__getitem__(substitute_annotation_container_types(params, ListType, DictType))
        new_cls._raise_on_forbidden_value_type()  # noqa: SLF001
        return new_cls

    def _raise_on_forbidden_value_type(cls):  # noqa: N805
        if cls.__parameters__:
            return
        assert cls.ValueT is not Ellipsis

        if isinstance(get_maybe_optional_type(cls.ValueT), TypeVar):
            return

        if type_annotation_contains_type_var(cls.ValueT):
            raise TypeError(f"TypeVar annotation allowed only directly or Optional[TypeVar], but got {cls}")


class DataclassDict(DataclassInterface, frozendict, Template[ValueT], metaclass=DataclassDictMeta):
    """
    Immutable Dict container compatible with DataclassInterface API.
    Assumes key type is always a str. Value type provided at runtime.

    Allowed value type annotations (including nested versions of these):
        Any, Dict, List, Optional, Tuple, Union
    TypeVar annotation is also allowed, but it can't be nested under any of the above containers
    """

    @classmethod
    def _container_classes(cls) -> Tuple[Type, Type]:
        """The (ListType, DictType) pair nested containers are cast to. Overridden by ConfigDict."""
        from ladder.core.databib.dataclasses.dataclass_list import DataclassList

        return DataclassList, DataclassDict

    def __new__(cls, *args, **kwargs):
        values = frozendict(*args, **kwargs)
        # Cast value to Dict[cls.ValueT] (note taht ValueT already has plain Dict/Dict container types
        # substituted with their DataclassList / DataclassDict equivalents).
        # NOTE: We cast the outer container to a python list instead of DataclassDict to avoid infinite
        # recursion since we are in DataclassDict.__new__. super().__new__() will cast the outer container
        values = cast_container_types(values, Dict[str, cls.ValueT])
        return super().__new__(cls, values)

    def __init__(self, *args, **kwargs):
        del args, kwargs
        # NOTE: frozendict gets auto initialized in __new__. No need to call here
        DataclassInterface.__init__(self)
        self._check_value_types_at_runtime()

    def _check_value_types_at_runtime(self):
        if value_type_is_annotation_type(self, Dict[str, self.ValueT]):
            return
        # Provide more informative error
        for key, value in self.items():
            if not value_type_is_annotation_type({key: value}, Dict[str, self.ValueT]):
                raise TypeError(
                    f"Incorrect type provided for {self.__class__.__name__}: expected "
                    f"{self.ValueT}, but got {value} of type {type(value)} at key {key}"
                )

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], recursive: bool = True, ignore_redundant: bool = False
    ) -> "DataclassDict[ValueT]":
        if not isinstance(data, dict):
            raise TypeError(
                f"{cls}.from_json() expects input of type dict, but got input of type {type(data)}"
            )
        data = dict(data)

        if recursive and inspect.isclass(cls.ValueT) and issubclass(cls.ValueT, DataclassInterface):
            data = {
                key: cls.ValueT.from_json(value, recursive=recursive, ignore_redundant=ignore_redundant)
                for key, value in data.items()
            }

        return cls(**data)

    def as_json(self, recursive: bool = True) -> Dict[str, Any]:
        return {
            key: value.as_json() if isinstance(value, DataclassInterface) and recursive else value
            for key, value in self.items()
        }

    @classmethod
    def make_empty(cls: Type[DataclassDictT]) -> DataclassDictT:
        return cls()

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def items(self, recursive: bool = False):
        for key in self.keys():
            value = self[key]
            if recursive and isinstance(value, DataclassInterface):
                for subkey, subvalue in value.items(recursive=True):
                    yield f"{key}.{subkey}", subvalue
            else:
                yield key, value

    def replace(self, strict=False, **kwargs) -> "DataclassDict[ValueT]":
        """If strict, no new keys allowed, only existing values can be replaced"""
        extra_keys = kwargs.keys() - self.keys()
        if strict and len(extra_keys) > 0:
            raise KeyError(f"Unexpected extra keys: {extra_keys}")

        data = {**self}

        for key, value in kwargs.items():
            # Recurse if ValueT is a DataclassInterface field
            if inspect.isclass(self.ValueT) and issubclass(self.ValueT, DataclassInterface):
                # Case 1: value is already the correct type, just assign it
                if isinstance(value, self.ValueT):
                    data[key] = value
                # Case 2: value is a plain dict representing the dataclass
                elif isinstance(value, dict):
                    # Case 2.1: new key being added to the dataclass -> cast to ValueT
                    if key in extra_keys:
                        data[key] = self.ValueT(**value)
                    # Case 2.2: existing key being updated -> call replace() on the current instance
                    else:
                        data[key] = data[key].replace(**value)
                # Case 3: value is some other type - delegate to __init__ to cast or raise error
                else:
                    data[key] = value
            else:
                data[key] = value

        return self.__class__(**data)

    def apply(
        self, fcn: Callable, recursive: bool = True, skip_nones: bool = False
    ) -> "DataclassDict[ValueT]":
        def fcn_wrapper(v: Any) -> Any:
            if v is None and skip_nones:
                return None
            if isinstance(v, DataclassInterface) and recursive:
                return v.apply(fcn, recursive=True, skip_nones=skip_nones)
            return fcn(v)

        return self.__class__(**{k: fcn_wrapper(v) for k, v in self.items()})

    def __reduce__(self) -> Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]:
        """
        Needed so we properly support pickling for classes with square brackets in their names.

        Consider a Template subclass `DataclassDict[typing.Any]`. `Template.__getitem__` assigns
        the class at creation time to the module `databib.template.template`. However, directly
        accessing the class via `databib.template.template.DataclassDict[typing.Any]` doesn't work
        due to the name of the class containing square brackets. Because of that, pickle can't access
        the class when it has to pickle it.

        This method gets called when pickling and returns a tuple (callable, args, state), where
        `callable` will be called with the provided `args` to create an initial object instance
        and `state` will be passed to __setstate__ after creation to update its state. Note that
        the returned tuple must be pickle-able and we can't return a lambda or a local function.

        NOTE: Not yet clear why only DataclassDict and DataclassList need __reduce__, but Template doesn't
        """
        return (
            get_dynamic_class,
            (self.__class__.__module__, self.__class__.__name__, self.as_json()),
            self.__dict__,
        )


def get_dynamic_class(module_name: str, classname: str, data: Dict[str, Any]):
    return getattr(sys.modules[module_name], classname).from_json(data)
