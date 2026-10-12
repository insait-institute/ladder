import functools
import sys
from abc import ABCMeta
from typing import Dict, Generic, Tuple, TypeVar

from .template_utils import (
    TypeVarAttrsT,
    TypeVarT,
    check_type_var_replacements_are_valid,
    get_type_name,
    resolve_type_var_attrs_chain,
)


class TemplateMeta(ABCMeta):
    """
    Metaclass for Template. Performs safety checks and saves TypeVar parameters as class attributes

    TemplateMeta uses the following internal attributes for bookkeeping:
        __parameters__: Tuple[TypeVar, ...] = tuple([])
            A tuple of TypeVars that a class expects.
            Example: A[T1, T2].__parameters__ == (T1, T2)
        __origin__: Optional[Type] = None
            Keeps a reference to a type that was subscripted (otherwise is None).
            Example: A[T1, T2].__origin__ == A
        __args__: Optional[Tuple[Union[Type, TypeVar], ...]] = None
            A tuple of all arguments passed when subscripting the class
            Example: Dict[T, int].__args__ == (T, int)
        __type_var_attrs__: Dict[TypeVar, Union[Type, TypeVar, str]] = {}
            Keeps the entire chain of TypeVar replacements for this class
            Example:
                class A(Template[T1, T2]): pass; class B(A[int, T2]): pass
                B.__type_var_attrs__ == {T1: int, T2: T2}
    """

    __parameters__ = ()  # : Tuple[TypeVar, ...]; Help mypy
    __origin__ = None  # : type; Help mypy

    def __new__(mcs, name, bases, attributes, *args, **kwargs):  # noqa: N804
        """
        There are two cases when this method gets called:
            1. When a statement such as `Template[T]` is interpreted (or a subclass of Template)
                Note that this will call Template.__getitem__(), which will call this method in turn
            2. When a class inherits from Template, e.g. `class A(Template[T]): pass`
                Note that in this case we refer to the declaration of A, not Template[T]
        """

        # Case 2 - called when a class inherits from Template
        if "__origin__" not in attributes:
            # Prevent falling back to the values of '__origin__' and '__args__' from the MRO.
            # These should be attributes only when the class was subscripted.
            attributes["__origin__"] = None
            attributes["__args__"] = None
            mcs._raise_on_invalid_inheritance(name, bases)
            mcs._raise_on_missing_type_vars_from_template(name, bases)

        # If not called from __getitem__, determine the value for __parameters__
        if "__parameters__" not in attributes:
            assert attributes["__origin__"] is None
            attributes["__parameters__"] = mcs._determine_parameters_from_bases(bases)

        new_cls = super().__new__(mcs, name, bases, attributes, *args, **kwargs)

        new_cls._create_type_var_attrs_on_class(bases)  # noqa: SLF001

        return new_cls

    @functools.lru_cache(maxsize=None, typed=True)  # pylint:disable=cache-max-size-none  # noqa: B019
    def __getitem__(cls, params) -> type:
        """Gets called when a class is subscripted. Returns a new class, with type variables replaced."""
        params = params if type(params) is tuple else (params,)

        if cls.__origin__ is not None:
            raise TypeError(
                f"You called {cls.__name__}.__getitem__{params}, but recursive calls to Template.__getitem__() "
                f"are not encouraged. Create a new class which inherits from {cls.__name__} and Template instead."
            )

        if cls is not Template and len(cls.__parameters__) != len(params):
            raise ValueError(
                f"{cls.__name__}.__getitem__ expects {len(cls.__parameters__)} parameters: "
                f"{cls.__parameters__}, but got {len(params)} parameters: {params}"
            )

        if cls is Template and not all(isinstance(param, TypeVar) for param in params):
            raise TypeError(f"{cls.__name__}[] accepts only TypeVar parameters, but got {params}")

        # Get the parameters the subscripted class expects
        __parameters__: Tuple[TypeVarT, ...] = tuple(p for p in params if isinstance(p, TypeVar))  # type: ignore

        # Create the new class, for example A[int]
        name = f"{cls.__name__}[{', '.join(get_type_name(p) for p in params)}]"
        bases = (cls,)
        attributes = {
            "__origin__": cls,
            "__parameters__": __parameters__,
            "__args__": params,
            "__module__": __name__,
        }
        new_cls = type(cls)(name, bases, attributes)

        # Make new_cls globally visible by adding it as attribute to this module
        # (together with setting `"__module__": __name__` in attributes)
        sys.modules[__name__].__dict__[new_cls.__qualname__] = new_cls

        replacements: TypeVarAttrsT = {  # type: ignore
            # NOTE: strict=False is required: when `cls is Template`, `cls.__parameters__` is empty while
            # `params` holds the TypeVars, and zip must truncate to yield no replacements (see length guard above).
            type_var: type_value
            for type_var, type_value in zip(cls.__parameters__, params, strict=False)  # type: ignore
        }

        check_type_var_replacements_are_valid(replacements, module=cls.__module__)

        new_cls._update_type_var_attrs_on_class(  # noqa: SLF001
            update_values=replacements,
            type_var_attrs=new_cls.__type_var_attrs__,
        )

        return new_cls

    @staticmethod
    def _determine_parameters_from_bases(bases: Tuple[type, ...]) -> Tuple[TypeVarT, ...]:
        for base in bases:
            if getattr(base, "__origin__", None) is Template:
                assert isinstance(base, TemplateMeta)
                return tuple(base.__parameters__)
        return tuple()

    def _create_type_var_attrs_on_class(cls, bases: Tuple[type, ...]):
        params = getattr(cls, "__parameters__", ())
        new_type_var_attrs: Dict[TypeVarT, TypeVarT] = {type_var: type_var for type_var in params}  # type: ignore

        type_var_attrs = cls._get_type_var_attrs_from_bases(bases)

        for type_var in new_type_var_attrs:  # type: ignore
            type_value_in_base = type_var_attrs.get(type_var, None)
            if type_value_in_base is not None and not isinstance(type_value_in_base, TypeVar):  # type: ignore
                raise TypeError(
                    f"Cannot redefine type var {type_var} on class {cls} as it was previously "
                    f"bound to {type_value_in_base} in a superclass"
                )

        # Ignore pylint bug
        # pylint: disable=no-value-for-parameter
        cls._update_type_var_attrs_on_class(
            update_values=new_type_var_attrs,
            type_var_attrs=type_var_attrs,
        )

    def _update_type_var_attrs_on_class(
        cls,
        update_values: TypeVarAttrsT,
        type_var_attrs: TypeVarAttrsT,
    ):
        type_var_attrs = {**type_var_attrs, **update_values}

        for type_var, type_value in type_var_attrs.items():
            # Special case to support CRTP forward references
            if isinstance(type_value, str) and type_value == cls.__name__:
                type_var_attrs[type_var] = cls

        # Ignore pylint bug
        # pylint: disable=no-value-for-parameter
        cls._set_type_var_attrs(type_var_attrs)

    @staticmethod
    def _get_type_var_attrs_from_bases(bases: Tuple[type, ...]) -> TypeVarAttrsT:
        type_var_attrs_bases = [getattr(base, "__type_var_attrs__", {}) for base in bases]
        type_var_attrs = {
            key: value for base_attrs in reversed(type_var_attrs_bases) for key, value in base_attrs.items()
        }
        return type_var_attrs

    def _set_type_var_attrs(cls, type_var_attrs: TypeVarAttrsT):
        recursive_mapping = resolve_type_var_attrs_chain(type_var_attrs)

        for type_var, type_value in recursive_mapping.items():
            setattr(cls, type_var.__name__, type_value)

        cls.__type_var_attrs__ = type_var_attrs

    @staticmethod
    def _raise_on_invalid_inheritance(name: str, bases: Tuple[type, ...]):
        template_in_bases: bool = False
        for base in bases:
            # Prevent class A(Template)
            if base is Template:
                raise TypeError("Cannot inherit from plain Template class.")
            # Prevent class A(Template[T1], Template[T2])
            if getattr(base, "__origin__", None) is Template:
                if not template_in_bases:
                    template_in_bases = True
                else:
                    raise TypeError(f"Class {name} cannot inherit from Template[...] multiple times")
            # Prevent inherence from Generic
            if Generic in base.__mro__:
                raise TypeError(
                    f"Class {name} is inheriting from Template and typing.Generic, which is forbidden. "
                    f"Remove inheritance from typing.Generic."
                )

    @staticmethod
    def _raise_on_missing_type_vars_from_template(name: str, bases: Tuple[type, ...]):
        template_params = ()
        for base in bases:
            if getattr(base, "__origin__", base) == Template:
                template_params = getattr(base, "__parameters__", ())

        # If some of the TypeVars to are not provided explicitly to Template, raise an error
        # For example, class A(Template[T1, T2]); class B(Template[T1], A[T1, T2])
        for base in bases:
            if getattr(base, "__origin__", base) != Template:
                params = getattr(base, "__parameters__", ())
                if not set(params).issubset(template_params):
                    raise TypeError(
                        f"{name} cannot inherit from {base.__name__} "
                        f"because not all parameters are included in Template[]:\n"
                        f"{base.__name__}.__parameters__ = {params}, Template.__parameters__ = {template_params}"
                    )


class Template(metaclass=TemplateMeta):
    def __new__(cls, *args, **kwargs):
        if cls is Template:
            raise TypeError("Class Template cannot be instantiated, it can be used only as a base class")
        cls._raise_if_class_expects_parameters()
        cls._raise_if_template_args_are_forward_refs()

        # object.__new_()_ does not take extra arguments, which the __init__ of subclasses usually take
        if super().__new__ is object.__new__:
            obj = super().__new__(cls)
        else:
            obj = super().__new__(cls, *args, **kwargs)
        return obj

    @classmethod
    def _raise_if_class_expects_parameters(cls):
        if cls.__parameters__:
            raise TypeError(
                f"Cannot instantiate {cls.__name__} because of unfilled parameters: {cls.__parameters__}"
            )

    @classmethod
    def _raise_if_template_args_are_forward_refs(cls):
        for type_var, type_value in cls.__type_var_attrs__.items():
            if isinstance(type_value, str):
                raise TypeError(
                    f"Cannot create an instance of class {cls} as the parameter for TypeVar "
                    f"{type_var.__name__} is an unresolved forward reference of value {type_value}"
                )

    @classmethod
    def __class_getitem__(cls, params) -> type:
        """Define to help pyright. Delegate to metaclass"""
        return super().__class_getitem__(params)  # type: ignore[misc]
