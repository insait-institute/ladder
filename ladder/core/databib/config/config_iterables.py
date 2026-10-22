from typing import Dict, List, Tuple, Type

# Import as modules to avoid potential circular import problems
import ladder.core.databib.config.auto_config as auto_config
from ladder.core.databib.dataclasses import DataclassDict, DataclassList
from ladder.core.databib.utils.typing_utils import value_type_is_annotation_type

def _auto_config_isinstance(value, field_type) -> bool:
    """
    Run type checks by also accounting for AutoConfig
    """
    # Check AutoConfig type
    if issubclass(field_type, auto_config.AutoConfig):
        return isinstance(value, field_type)
    return type(value) is field_type


class ConfigDict(DataclassDict):
    @classmethod
    def _container_classes(cls) -> Tuple[Type, Type]:
        return ConfigList, ConfigDict

    def _check_value_types_at_runtime(self):
        """Override to allow for AutoConfig checks"""
        if value_type_is_annotation_type(self, Dict[str, self.ValueT], _auto_config_isinstance):
            return
        # Provide more informative error
        for key, value in self.items():
            if not value_type_is_annotation_type(
                {key: value}, Dict[str, self.ValueT], _auto_config_isinstance
            ):
                raise TypeError(
                    f"Incorrect type provided for {self.__class__.__name__}: expected "
                    f"{self.ValueT}, but got {value} of type {type(value)} at key {key}"
                )

    def replace(self, strict: bool = False, **kwargs):
        """
        Overload to change the default value of strict.
        In ConfigDict, adding new attributes is common behavior
        """
        return super().replace(strict=strict, **kwargs)


class ConfigList(DataclassList):
    @classmethod
    def _container_classes(cls) -> Tuple[Type, Type]:
        return ConfigList, ConfigDict

    def _check_value_types_at_runtime(self):
        """Override to allow for AutoConfig checks"""
        if value_type_is_annotation_type(self, List[self.ValueT], _auto_config_isinstance):
            return
        # Provide more informative error
        for index, value in enumerate(self):
            if not value_type_is_annotation_type([value], List[self.ValueT], _auto_config_isinstance):
                raise TypeError(
                    f"Incorrect type provided for {self.__class__.__name__}: expected "
                    f"{self.ValueT}, but got {value} of type {type(value)} at index {index}"
                )
