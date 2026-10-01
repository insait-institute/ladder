import abc
from typing import Any, Callable, Iterator, Tuple, Union


class DataclassInterface(abc.ABC):
    """
    Interface for dataclasses and the supported methods.
    All dataclasses are immutable and all methods return new instances.
    """

    @classmethod
    @abc.abstractmethod
    def from_json(cls, data, recursive: bool = True, ignore_redundant: bool = False):
        pass

    @classmethod
    @abc.abstractmethod
    def make_empty(cls):
        pass

    @abc.abstractmethod
    def as_json(self, recursive: bool = True) -> Union[dict, list]:
        pass

    @abc.abstractmethod
    def replace(self, **kwargs):
        """Returns a new instance of the dataclass with the given fields replaced by the given values"""

    @abc.abstractmethod
    def apply(self, fcn: Callable, recursive: bool = True, skip_nones: bool = False):
        pass

    @abc.abstractmethod
    def items(self, recursive: bool = False) -> Iterator[Tuple[str, Any]]:
        pass

    def __repr__(self) -> str:
        indent = " " * 2

        fields_reprs = [
            f"{indent}{key}={repr(value)}".replace('\n', f'\n{indent}')
            for key, value in self.items(recursive=False)
        ]
        fields_repr = ("\n" + ",\n".join(fields_reprs) + f",\n") if len(fields_reprs) > 0 else ""

        return f"{self.__class__.__name__}({fields_repr})"
