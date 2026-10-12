import inspect
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type, TypeVar

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
DataclassListT = TypeVar("DataclassListT", bound="DataclassList")


class DataclassListMeta(TemplateMeta):
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


class DataclassList(DataclassInterface, tuple, Template[ValueT], metaclass=DataclassListMeta):
    """
    Immutable List container compatible with DataclassInterface API. Value type provided at runtime.

    Allowed value type annotations (including nested versions of these):
        Any, Dict, List, Optional, Tuple, Union
    TypeVar annotation is also allowed, but it can't be nested under any of the above containers
    """

    @classmethod
    def _container_classes(cls) -> Tuple[Type, Type]:
        """The (ListType, DictType) pair nested containers are cast to. Overridden by ConfigList."""
        from ladder.core.databib.dataclasses.dataclass_dict import DataclassDict

        return DataclassList, DataclassDict

    def __new__(cls, *args, **kwargs):
        value = tuple(*args, **kwargs)
        # Cast value to List[cls.ValueT] (note that ValueT already has plain List/Dict container types
        # substituted with their DataclassList / DataclassDict equivalents).
        # NOTE: We cast the outer container to a python list instead of DataclassList to avoid infinite
        # recursion since we are in DataclassList.__new__. super().__new__() will cast the outer container
        value = cast_container_types(value, List[cls.ValueT])
        return super().__new__(cls, value)

    def __init__(self, *args, **kwargs):
        del args, kwargs
        # NOTE: tuple gets auto initialized in __new__. No need to call here
        DataclassInterface.__init__(self)
        self._check_value_types_at_runtime()

    def _check_value_types_at_runtime(self):
        if value_type_is_annotation_type(self, List[self.ValueT]):
            return
        # Provide more informative error
        for index, value in enumerate(self):
            if not value_type_is_annotation_type([value], List[self.ValueT]):
                raise TypeError(
                    f"Incorrect type provided for {self.__class__.__name__}: expected "
                    f"{self.ValueT}, but got {value} of type {type(value)} at index {index}"
                )

    @classmethod
    def from_json(
        cls, data: Sequence[Any], recursive: bool = True, ignore_redundant: bool = False
    ) -> "DataclassList[ValueT]":
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"{cls}.from_json() expects input of type list or tuple, but got input of type {type(data)}"
            )
        data = list(data)

        if recursive and inspect.isclass(cls.ValueT) and issubclass(cls.ValueT, DataclassInterface):
            data = [
                cls.ValueT.from_json(value, recursive=recursive, ignore_redundant=ignore_redundant)
                for value in data
            ]

        return cls(data)

    def as_json(self, recursive: bool = True) -> List:
        return [
            value.as_json() if isinstance(value, DataclassInterface) and recursive else value
            for value in self
        ]

    @classmethod
    def make_empty(cls: Type[DataclassListT]) -> DataclassListT:
        return cls([])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def items(self, recursive: bool = False):
        for i, value in enumerate(self):
            if recursive and isinstance(value, DataclassInterface):
                for subkey, subvalue in value.items(recursive=True):
                    yield f"{i}.{subkey}", subvalue
            else:
                yield f"{i}", value

    def replace(self, **kwargs) -> "DataclassList[ValueT]":  # noqa: ARG002
        raise RuntimeError(f"{self.__class__.__name__}.replace() is not supported")

    def apply(
        self, fcn: Callable, recursive: bool = True, skip_nones: bool = False
    ) -> "DataclassList[ValueT]":
        def fcn_wrapper(v: Any) -> Any:
            if v is None and skip_nones:
                return None
            if isinstance(v, DataclassInterface) and recursive:
                return v.apply(fcn, recursive=True, skip_nones=skip_nones)
            return fcn(v)

        return self.__class__([fcn_wrapper(value) for value in self])

    def __repr__(self) -> str:
        indent = " " * 2

        fields_reprs = [
            f"{indent}{repr(value)}".replace('\n', f'\n{indent}') for _, value in self.items(recursive=False)
        ]
        fields_repr = ("\n" + ",\n".join(fields_reprs) + f",\n") if len(fields_reprs) > 0 else ""

        return f"{self.__class__.__name__}({fields_repr})"

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
