import functools
from dataclasses import dataclass
from typing import Any, TypeVar

from ladder.core.databib.utils.typing_utils import (
    annotation_is_dict,
    annotation_is_list,
    get_dict_annotation_type_args,
    get_list_annotation_type_args,
    maybe_update_type_annotation,
)

from .template import Template, TemplateMeta
from .template_utils import resolve_type_var_attrs_chain


class TemplateDataclassMeta(TemplateMeta):
    """
    Metaclass for templated dataclasses.
    Updates TypeVars in type annotations and fields with specific type arguments.

    Functionality:

    1. Overwrites dataclass field types with specific type arguments (instead of TypeVar). Example:
        ```
        @dataclass
        class Base(Template[T]):
            data: T
        @dataclass
        class IntData(Base[int]):
            pass
        ```
        Both IntData.__annotations__['data'] and IntData.fields['data'].type are int as opposed to T

    2. Makes sure that TypeVars used in dataclass fields are included in the Template arguments. Example:
        ```
        @dataclass
        class A(TemplateDataclass, Template[T1]):
            field_1: T1
            fiedl_2: T2
        ```
        Raises TypeError as T2 is not included in Template arguments
    """

    def __new__(mcs, name, bases, attributes, *args, **kwargs):  # noqa: N804
        new_cls = super().__new__(mcs, name, bases, attributes, *args, **kwargs)

        # If the call to __new__ is from Template.__getitem__(), then we don't need
        # to run checks on this class. In the example below, both Template[T1] and A
        # will call __new__(), but we only want to run checks on A
        # @dataclass
        # class A(TemplateDataclass, Template[T1]):
        #     field_1: T1
        #     field_2: T2
        if attributes.get('__module__', None) is not Template.__module__:
            new_cls._raise_on_missing_template_type_vars(name, bases)  # noqa: SLF001

        return new_cls

    def _raise_on_missing_template_type_vars(cls, name, bases):  # noqa: N805
        """
        Prevent cases where a TypeVar is used in a dataclass field, but is not included in the Template arguments
        Example:
            @dataclass
            class A(TemplateDataclass, Template[T1]):
                field_1: T1
                field_2: T2
                field_3: Dict[str, T3]
                field_4: List[T4]
        """

        # NOTE: At this point, @dataclass hasn't been applied yet, so __dataclass_fields__ might be empty.
        # __annotations__ gives us the fields defined by this class (A in example above), while
        # __dataclass_fields__ gives us the fields defined by the parents
        annotations = getattr(cls, '__annotations__', {})
        dataclass_fields = getattr(cls, '__dataclass_fields__', {})
        fields = {**annotations, **{field.name: field.type for field in dataclass_fields.values()}}
        parameters = set(getattr(cls, '__parameters__', ()))

        for field_name, field_type in fields.items():
            if isinstance(field_type, TypeVar) and field_type not in parameters:
                # pylint: disable=no-value-for-parameter
                cls._raise_missing_type_var_error(name, bases, field_name, field_type, field_type)
            if annotation_is_dict(field_type):
                dict_vtype = get_dict_annotation_type_args(field_type)[1]
                if isinstance(dict_vtype, TypeVar) and dict_vtype not in parameters:
                    # pylint: disable=no-value-for-parameter
                    cls._raise_missing_type_var_error(name, bases, field_name, field_type, dict_vtype)
            if annotation_is_list(field_type):
                list_vtype = get_list_annotation_type_args(field_type)[0]
                if isinstance(list_vtype, TypeVar) and list_vtype not in parameters:
                    # pylint: disable=no-value-for-parameter
                    cls._raise_missing_type_var_error(name, bases, field_name, field_type, list_vtype)

    def _raise_missing_type_var_error(
        cls, name, bases, field_name: str, field_type: Any, type_var: TypeVar  # noqa: N805
    ):
        suggest_bases = [
            base.__name__
            if getattr(base, "__origin__", None) is not Template
            else f"Template[{', '.join([p.__name__ for p in base.__parameters__] + [type_var.__name__])}]"
            for base in bases
        ]
        raise TypeError(
            f"Dataclass {cls} has field {field_name}: {field_type}, where {field_type} "
            f"contains a TypeVar that is not included in Template arguments. "
            f"The correct definition should be: {name}({', '.join(suggest_bases)})"
        )

    @functools.lru_cache(maxsize=None, typed=True)  # pylint:disable=cache-max-size-none  # noqa: B019
    def __getitem__(cls, params):  # noqa: N805
        params = params if type(params) is tuple else (params,)

        new_cls = super().__getitem__(params)
        new_cls = new_cls._update_dataclass_fields()  # noqa: SLF001

        return new_cls

    def _update_dataclass_fields(cls):  # noqa: N805
        """
        Update __dataclass_fields__ and __annotations__ attributes.
        Note that while A[int] creates a new class, A[int].__dataclass_fields__ and A[int].__annotations__ are actually
        proxies for A.__dataclass_fields__ and A.__annotations__. Here we make a copy of A[int].__dataclass_fields__
        and A[int].__annotations__ and update them with the new type annotations
        """
        type_updates = resolve_type_var_attrs_chain(cls.__type_var_attrs__)

        if getattr(cls, '__dataclass_fields__', None) is None:
            return cls

        # NOTE: __dataclass_fields__ is maintained by the dataclasses package and is the only correct source for
        # dataclass fields as it includes fields from base classes. __annotations__ does not necessarily contain
        # keys from bases and might include type annotations which are not intended as dataclass fields

        # Construct __annotations__ using the updated types
        type_annotations = {}
        for key, field in cls.__dataclass_fields__.items():
            type_annotations[key] = maybe_update_type_annotation(field.type, type_updates)

        # Overwrite __annotations__. Note this is completely safe to do as there is
        # no way to provide extra type annotations when calling cls.__getitem__()
        cls.__annotations__ = type_annotations

        # Wrap in dataclass so that __dataclass_fields__ is updated with the new types
        new_cls = dataclass(
            cls,
            frozen=cls.__dataclass_params__.frozen,
            eq=cls.__dataclass_params__.eq,
            init=cls.__dataclass_params__.init,
            order=cls.__dataclass_params__.order,
            repr=cls.__dataclass_params__.repr,
            unsafe_hash=cls.__dataclass_params__.unsafe_hash,
        )

        return new_cls


class TemplateDataclass(metaclass=TemplateDataclassMeta):
    def __new__(cls, *args, **kwargs):
        # object.__new_()_ does not take extra arguments, which the __init__ of subclasses usually take
        if super().__new__ is object.__new__:
            obj = super().__new__(cls)
        else:
            obj = super().__new__(cls, *args, **kwargs)
        return obj
