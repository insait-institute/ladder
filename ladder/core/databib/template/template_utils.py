import sys
from typing import ForwardRef  # type: ignore
from typing import Dict, Type, TypeVar, Union

from ladder.core.databib.utils.typing_utils import annotation_is_union, get_annotation_type_args

TypeVarT = TypeVar('TypeVarT')  # mypy does not like annotations with TypeVar
TypeValueT = Union[type, TypeVarT, str]  # str is used for forward references
TypeVarAttrsT = Dict[TypeVarT, TypeValueT]


def check_type_var_replacements_are_valid(replacements: TypeVarAttrsT, module: str):
    for type_var, type_val in replacements.items():
        check_type_var_replacement_is_valid(type_var, type_val, module)


def get_type_name(type_value: Union[Type, TypeVarT]) -> str:
    if isinstance(type_value, type):
        return type_value.__name__
    return repr(type_value)


def check_type_var_replacement_is_valid(type_var: TypeVarT, type_value: TypeValueT, module: str):
    # We can't check against forward references
    if isinstance(type_value, str):
        return

    bound = resolve_forward_reference(type_var.__bound__, module=module)  # type: ignore
    if bound:
        if isinstance(type_value, TypeVar):  # type: ignore
            try:
                replacement_bound = resolve_forward_reference(type_value.__bound__, module=module)  # type: ignore
            except NameError:
                # If we are replacing with a forward type reference, it's possible the target type hasn't
                # been defined yet, in which case we get a NameError.
                return

            if not replacement_bound:
                raise TypeError(
                    f'TypeVar {type_var} specifies a bound, but replacement type {type_value} does not'
                )

            if not issubclass(replacement_bound, bound):
                raise TypeError(
                    f"Replacement TypeVar {type_value} with bound {replacement_bound} does not satisfy bound of "
                    f"parent TypeVar {type_var} with bound {bound}"
                )
        elif annotation_is_union(type_value):
            for type_value_in_union in get_annotation_type_args(type_value):
                check_type_var_replacement_is_valid(type_var, type_value_in_union, module=module)
        elif not issubclass(type_value, bound):
            raise TypeError(
                f'Type {type_value.__name__} does not satisfy bound of {type_var}: {bound.__name__}'
            )


def resolve_forward_reference(type_value: Union[type, TypeVarT], module) -> Union[type, TypeVarT]:
    if isinstance(type_value, ForwardRef):
        global_namespace = sys.modules[module].__dict__
        result = type_value._evaluate(  # noqa: SLF001
            global_namespace, {}, frozenset()
        )  # pylint: disable=protected-access
        assert result is not None
        return result
    return type_value


def resolve_type_var_attrs_chain(type_var_attrs: TypeVarAttrsT) -> TypeVarAttrsT:
    def is_last_in_chain(type_value: TypeValueT):
        if isinstance(type_value, TypeVar):  # type: ignore
            return type_var_attrs[type_value] is type_value
        return True

    recursive_mapping: TypeVarAttrsT = {}

    for type_var in type_var_attrs.keys():
        chain = [type_var]
        last_in_chain = chain[-1]

        while last_in_chain not in recursive_mapping.keys() and not is_last_in_chain(last_in_chain):
            last_in_chain = type_var_attrs[last_in_chain]
            chain.append(last_in_chain)

        type_value = last_in_chain if is_last_in_chain(last_in_chain) else recursive_mapping[last_in_chain]

        for type_var_in_chain in chain[:-1]:
            recursive_mapping[type_var_in_chain] = type_value

        if isinstance(type_value, TypeVar):  # type: ignore
            recursive_mapping[type_value] = type_value

    return recursive_mapping
