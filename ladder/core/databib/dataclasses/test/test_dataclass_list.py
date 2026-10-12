import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

from ladder.core.databib.dataclasses.dataclass import Dataclass, dataclass
from ladder.core.databib.dataclasses.dataclass_dict import DataclassDict
from ladder.core.databib.dataclasses.dataclass_list import DataclassList


@dataclass
class InnerDataclass(Dataclass):
    c: int


@dataclass
class OuterDataclass(Dataclass):
    a: DataclassList[InnerDataclass]
    b: DataclassList[int]


@dataclass
class SimpleClass(Dataclass):
    b: DataclassList[int]


@dataclass
class RecursiveListDataclass(Dataclass):
    ll: List[List[int]]
    ld: List[Dict[str, int]]


class TestDataclassList(unittest.TestCase):
    def test_type_arguments(self):
        # pylint: disable=pointless-statement
        DataclassList[Any]
        DataclassList[Optional[int]]

        DataclassList[Dict[str, int]]

        DataclassList[List[str]]

        DataclassList[Tuple[str]]

        DataclassList[Optional[Dict[str, int]]]

        DataclassList[Union[str, int]]

    def test_from_json_bad_input(self):
        data = {"b": {'x': 'y'}}
        with self.assertRaises(TypeError):
            SimpleClass.from_json(data)

        data = {"b": ['x', 'y']}
        with self.assertRaises(TypeError):
            SimpleClass.from_json(data)

    def test_from_json(self):
        data = {
            "a": (
                {"c": 0},
                {"c": 1},
                {"c": 2},
            ),
            "b": (3, 4),
        }

        outer = OuterDataclass.from_json(data)

        self.assertIsInstance(outer.a[0], InnerDataclass)
        self.assertIsInstance(outer.a[1], InnerDataclass)
        self.assertIsInstance(outer.a[2], InnerDataclass)

        self.assertEqual(outer.a[0].c, 0)
        self.assertEqual(outer.a[1].c, 1)
        self.assertEqual(outer.a[2].c, 2)
        self.assertEqual(outer.b[0], 3)
        self.assertEqual(outer.b[1], 4)

    def test_as_json(self):
        outer = OuterDataclass(
            a=DataclassList[InnerDataclass](
                [
                    InnerDataclass(c=0),
                    InnerDataclass(c=1),
                    InnerDataclass(c=2),
                ]
            ),
            b=DataclassList[int](
                [
                    3,
                    4,
                ]
            ),
        )

        self.assertDictEqual(
            outer.as_json(),
            {
                "a": [
                    {"c": 0},
                    {"c": 1},
                    {"c": 2},
                ],
                "b": [3, 4],
            },
        )

    def test_make_empty(self):
        empty = OuterDataclass.make_empty()

        self.assertIsInstance(empty.a, DataclassList[InnerDataclass])
        self.assertIsInstance(empty.b, DataclassList[int])

        self.assertListEqual(empty.a.as_json(), [])
        self.assertListEqual(empty.b.as_json(), [])

    def test_apply(self):
        outer = OuterDataclass(
            a=DataclassList[InnerDataclass](
                [
                    InnerDataclass(c=0),
                    InnerDataclass(c=1),
                    InnerDataclass(c=2),
                ]
            ),
            b=DataclassList[int](
                [
                    3,
                    4,
                ]
            ),
        )

        outer = outer.apply(lambda x: x + 1)

        self.assertDictEqual(
            outer.as_json(),
            {
                "a": [
                    {"c": 1},
                    {"c": 2},
                    {"c": 3},
                ],
                "b": [4, 5],
            },
        )

    def test_items(self):
        data = DataclassList[InnerDataclass](
            [
                InnerDataclass(c=0),
                InnerDataclass(c=1),
                InnerDataclass(c=2),
            ]
        )

        self.assertDictEqual(
            {
                "0": InnerDataclass(c=0),
                "1": InnerDataclass(c=1),
                "2": InnerDataclass(c=2),
            },
            {key: value for key, value in data.items()},
        )

        data = OuterDataclass(
            a=DataclassList[InnerDataclass](
                [
                    InnerDataclass(c=0),
                    InnerDataclass(c=1),
                    InnerDataclass(c=2),
                ]
            ),
            b=DataclassList[int](
                [
                    3,
                    4,
                ]
            ),
        )

        self.assertDictEqual(
            {
                "a.0.c": 0,
                "a.1.c": 1,
                "a.2.c": 2,
                "b.0": 3,
                "b.1": 4,
            },
            {key: value for key, value in data.items(recursive=True)},
        )

    def test_recursive_list_of_list(self):
        outer = RecursiveListDataclass.from_json({"ll": [[1, 2], [3]], "ld": []})

        # The plain List[List[int]] field is wrapped recursively: both the outer field and the
        # nested level become DataclassList, not plain lists
        self.assertIsInstance(outer.ll, DataclassList)
        self.assertIsInstance(outer.ll[0], DataclassList)
        # apply traverses all the way to the leaves
        self.assertListEqual(outer.ll.apply(lambda x: x * 2).as_json(), [[2, 4], [6]])
        # Round-trips back to plain nested lists
        self.assertListEqual(outer.as_json()["ll"], [[1, 2], [3]])

    def test_recursive_list_of_dict(self):
        outer = RecursiveListDataclass.from_json({"ll": [], "ld": [{"a": 1}, {"b": 2}]})

        self.assertIsInstance(outer.ld, DataclassList)
        self.assertIsInstance(outer.ld[0], DataclassDict)
        self.assertListEqual(outer.ld.apply(lambda x: x + 1).as_json(), [{"a": 2}, {"b": 3}])
        self.assertListEqual(outer.as_json()["ld"], [{"a": 1}, {"b": 2}])

    def test_value_t_is_substituted(self):
        self.assertIs(DataclassList[List[int]].ValueT, DataclassList[int])
        self.assertIs(DataclassList[Dict[str, int]].ValueT, DataclassDict[int])

    def test_direct_construction_recurses(self):
        nested = DataclassList[Dict[str, int]]([{"a": 1}])
        self.assertIsInstance(nested[0], DataclassDict)

    def test_repr(self):
        self.assertEqual(
            f"{DataclassList[int]([5, 6, 7])}",
            # fmt: off
            "DataclassList[int]("
            "\n  5,"
            "\n  6,"
            "\n  7,"
            "\n)"
            # fmt: on
        )

        self.assertEqual(
            f"{DataclassList[int]([])}",
            "DataclassList[int]()",
        )


if __name__ == '__main__':
    unittest.main()
