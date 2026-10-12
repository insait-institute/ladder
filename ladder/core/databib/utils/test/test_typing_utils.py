import unittest
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

from ladder.core.databib.config.config_iterables import ConfigDict, ConfigList
from ladder.core.databib.dataclasses.dataclass_dict import DataclassDict
from ladder.core.databib.dataclasses.dataclass_list import DataclassList
from ladder.core.databib.utils.typing_utils import (
    annotation_is_dict,
    annotation_is_list,
    annotation_is_optional,
    annotation_is_tuple,
    annotation_is_union,
    get_maybe_optional_type,
    maybe_update_type_annotation,
    substitute_annotation_container_types,
    value_type_is_annotation_type,
)

T = TypeVar('T')


class TestTypingUtils(unittest.TestCase):
    def test_annotation_is_union(self):
        self.assertTrue(annotation_is_union(Union[int, str]))
        self.assertTrue(annotation_is_union(int | str))

    def test_annotation_is_optional(self):
        self.assertTrue(annotation_is_optional(Union[int, None]))
        self.assertTrue(annotation_is_optional(Optional[int]))
        self.assertTrue(annotation_is_optional(int | None))

    def test_get_maybe_optional_type(self):
        self.assertEqual(int, get_maybe_optional_type(Optional[int]))
        self.assertEqual(int, get_maybe_optional_type(int | None))
        self.assertEqual(int, get_maybe_optional_type(None | int))
        self.assertEqual(int, get_maybe_optional_type(int))

    def test_annotation_is_list(self):
        self.assertTrue(annotation_is_list(List[int]))
        self.assertTrue(annotation_is_list(list[int]))
        self.assertTrue(annotation_is_list(list))

    def test_annotation_is_dict(self):
        self.assertTrue(annotation_is_dict(Dict[str, int]))
        self.assertTrue(annotation_is_dict(dict[str, int]))
        self.assertTrue(annotation_is_dict(dict))

    def test_annotation_is_tuple(self):
        self.assertTrue(annotation_is_tuple(Tuple[int, str]))
        self.assertTrue(annotation_is_tuple(tuple[int, str]))
        self.assertTrue(annotation_is_tuple(tuple))

    def test_value_type_is_annotation_type_flat(self):
        # Any
        self.assertTrue(value_type_is_annotation_type(None, Any))
        self.assertTrue(value_type_is_annotation_type(6, Any))
        self.assertTrue(value_type_is_annotation_type(6.0, Any))
        self.assertTrue(value_type_is_annotation_type('6.0', Any))
        self.assertTrue(value_type_is_annotation_type(6, Any))
        self.assertTrue(value_type_is_annotation_type([6], Any))
        self.assertTrue(value_type_is_annotation_type((6,), Any))
        self.assertTrue(value_type_is_annotation_type({6: 6}, Any))

        # Optional
        self.assertTrue(value_type_is_annotation_type(None, Optional[int]))
        self.assertTrue(value_type_is_annotation_type(6, Optional[int]))

        self.assertFalse(value_type_is_annotation_type(6, Optional[str]))  # Error: int

        # Union
        self.assertTrue(value_type_is_annotation_type(None, Union[int, None]))
        self.assertTrue(value_type_is_annotation_type(6, Union[int, None]))
        self.assertTrue(value_type_is_annotation_type(None, int | None))
        self.assertTrue(value_type_is_annotation_type(6, int | None))
        self.assertTrue(value_type_is_annotation_type(6, Union[int, str, float]))
        self.assertTrue(value_type_is_annotation_type(6.0, Union[int, str, float]))
        self.assertTrue(value_type_is_annotation_type('6', Union[int, str, float]))

        self.assertFalse(value_type_is_annotation_type(None, Union[int, str, float]))  # Error: None
        self.assertFalse(value_type_is_annotation_type([6], Union[int, str, float]))  # Error: list
        self.assertFalse(value_type_is_annotation_type((6,), Union[int, str, float]))  # Error: tuple
        self.assertFalse(value_type_is_annotation_type({6: 6}, Union[int, str, float]))  # Error: dict

        # Dict
        self.assertTrue(value_type_is_annotation_type({'a': 5}, Dict[str, int]))
        self.assertTrue(value_type_is_annotation_type({'a': 5}, Dict[str, Any]))
        self.assertTrue(value_type_is_annotation_type({'a': 5, 'b': '6'}, Dict[str, Any]))

        self.assertFalse(value_type_is_annotation_type({'a': 5, 'b': '6'}, Dict[str, int]))  # Error: str
        self.assertFalse(value_type_is_annotation_type({'a': 5, 7: 6}, Dict[str, int]))  # Error: int
        self.assertFalse(value_type_is_annotation_type(None, Dict[str, int]))  # Error: None

        # List
        self.assertTrue(value_type_is_annotation_type([5, 6, 7], List[int]))
        self.assertTrue(value_type_is_annotation_type([5, 6, 7], List[Any]))
        self.assertTrue(value_type_is_annotation_type([5, '6', 7], List[Any]))

        self.assertFalse(value_type_is_annotation_type([5, '6', 7], List[int]))  # Error: str
        self.assertFalse(value_type_is_annotation_type(None, List[int]))  # Error: None

        # Tuple
        self.assertTrue(value_type_is_annotation_type((5, 6, 7), Tuple[int, ...]))
        self.assertTrue(value_type_is_annotation_type((5, 6, 7), Tuple[Any, ...]))
        self.assertTrue(value_type_is_annotation_type((5, 6, 7), Tuple[int, int, int]))
        self.assertTrue(value_type_is_annotation_type((5, 6, 7), Tuple[Any, Any, Any]))
        self.assertTrue(value_type_is_annotation_type((5, '6'), Tuple[int, str]))

        self.assertFalse(value_type_is_annotation_type((5, 6, 7), Tuple[int, int]))  # Error: len
        self.assertFalse(value_type_is_annotation_type((5, '6'), Tuple[str, int]))  # Error: order

        # Numpy types
        self.assertTrue(value_type_is_annotation_type(np.uint8(3), int))
        self.assertTrue(value_type_is_annotation_type(np.int16(3), int))
        self.assertTrue(value_type_is_annotation_type(np.int32(3), int))
        self.assertTrue(value_type_is_annotation_type(np.int64(3), int))
        self.assertTrue(value_type_is_annotation_type(np.float16(6.0), float))
        self.assertTrue(value_type_is_annotation_type(np.float32(6.0), float))
        self.assertTrue(value_type_is_annotation_type(np.float64(6.0), float))

        # int -> float
        self.assertTrue(value_type_is_annotation_type(5, float))

        self.assertFalse(value_type_is_annotation_type(5.0, int))
        self.assertFalse(value_type_is_annotation_type(5.0, np.int32))

    def test_maybe_update_type_annotation(self):
        update_map = {T: str}

        with self.subTest("Union"):
            self.assertEqual(maybe_update_type_annotation(Union[int, T], update_map), Union[int, str])
            self.assertEqual(maybe_update_type_annotation(int | T, update_map), Union[int, str])

        with self.subTest("Optional"):
            self.assertEqual(maybe_update_type_annotation(Optional[T], update_map), Optional[str])

        with self.subTest("Tuple"):
            self.assertEqual(maybe_update_type_annotation(Tuple[int, T], update_map), Tuple[int, str])
            self.assertEqual(maybe_update_type_annotation(Tuple[T, ...], update_map), Tuple[str, ...])

        with self.subTest("nested"):
            self.assertEqual(
                maybe_update_type_annotation(Dict[str, List[Union[int, T]]], update_map),
                Dict[str, List[Union[int, str]]],
            )

    def test_substitute_annotation_container_types(self):
        # (annotation, ListType, DictType, expected). The (ListType, DictType) pair selects the
        # container family every List / Dict is rewritten to.
        cases = [
            # Leaves are returned unchanged
            (int, DataclassList, DataclassDict, int),
            (Any, DataclassList, DataclassDict, Any),
            # Flat containers
            (List[int], DataclassList, DataclassDict, DataclassList[int]),
            (Dict[str, int], DataclassList, DataclassDict, DataclassDict[int]),
            # Nested containers recurse all the way down to the leaves
            (List[Dict[str, int]], DataclassList, DataclassDict, DataclassList[DataclassDict[int]]),
            (
                Dict[str, Dict[str, List[int]]],
                DataclassList,
                DataclassDict,
                DataclassDict[DataclassDict[DataclassList[int]]],
            ),
            # Optional / Tuple are preserved with their members rewritten
            (Optional[List[int]], DataclassList, DataclassDict, Optional[DataclassList[int]]),
            (Tuple[int, List[str]], DataclassList, DataclassDict, Tuple[int, DataclassList[str]]),
            # The Config family is selected by passing ConfigList / ConfigDict
            (List[Dict[str, int]], ConfigList, ConfigDict, ConfigList[ConfigDict[int]]),
        ]
        for annotation, list_type, dict_type, expected in cases:
            with self.subTest(annotation=annotation, list_type=list_type):
                self.assertEqual(
                    substitute_annotation_container_types(annotation, list_type, dict_type), expected
                )

        # Identity (not just equality) is preserved when nothing nested changed, so callers can
        # cheaply detect a no-op via `is`.
        self.assertIs(
            substitute_annotation_container_types(Optional[int], DataclassList, DataclassDict),
            Optional[int],
        )

    def test_value_type_is_annotation_type_nested(self):
        # Dict
        self.assertTrue(value_type_is_annotation_type({'a': {'b': 5}}, Dict[str, Dict[str, int]]))
        self.assertTrue(value_type_is_annotation_type({'a': [5]}, Dict[str, List[int]]))
        self.assertTrue(value_type_is_annotation_type({'a': (5,)}, Dict[str, Tuple[int, ...]]))
        self.assertTrue(value_type_is_annotation_type({'a': 5, 'b': '6'}, Dict[str, Union[int, str]]))
        self.assertTrue(value_type_is_annotation_type({'a': 5, 'b': None}, Dict[str, Optional[int]]))

        # List
        self.assertTrue(value_type_is_annotation_type([{'b': 5}], List[Dict[str, int]]))
        self.assertTrue(value_type_is_annotation_type([[5]], List[List[int]]))
        self.assertTrue(value_type_is_annotation_type([(5,)], List[Tuple[int, ...]]))
        self.assertTrue(value_type_is_annotation_type([5, '6'], List[Union[int, str]]))
        self.assertTrue(value_type_is_annotation_type([5, None], List[Optional[int]]))

        # Tuple
        self.assertTrue(value_type_is_annotation_type(({'b': 5},), Tuple[Dict[str, int], ...]))
        self.assertTrue(value_type_is_annotation_type(([5],), Tuple[List[int], ...]))
        self.assertTrue(value_type_is_annotation_type(((5,),), Tuple[Tuple[int, ...], ...]))
        self.assertTrue(
            value_type_is_annotation_type(
                (
                    5,
                    '6',
                ),
                Tuple[Union[int, str], ...],
            )
        )
        self.assertTrue(
            value_type_is_annotation_type(
                (
                    5,
                    None,
                ),
                Tuple[Optional[int], ...],
            )
        )
