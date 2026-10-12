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
    a: DataclassDict[InnerDataclass]
    b: DataclassDict[int]


@dataclass
class RecursiveDictDataclass(Dataclass):
    dd: Dict[str, Dict[str, int]]
    dl: Dict[str, List[int]]


class TestDataclassDict(unittest.TestCase):
    def test_type_arguments(self):
        # pylint: disable=pointless-statement
        DataclassDict[Any]

        DataclassDict[Optional[int]]

        DataclassDict[Dict[str, int]]

        DataclassDict[List[str]]

        DataclassDict[Tuple[str]]

        DataclassDict[Optional[Dict[str, int]]]

        DataclassDict[Union[str, int]]

    def test_from_json_bad_messages(self):
        data = {
            "a": {
                "d": {"c": 0},
            },
            "b": {
                "g": "3",
                "h": 4,
            },
        }
        with self.assertRaises(TypeError):
            OuterDataclass.from_json(data)

        data = {
            "a": {
                "d": {"c": 0},
            },
            "b": "hello",
        }
        with self.assertRaises(TypeError):
            OuterDataclass.from_json(data)

    def test_from_json(self):
        data = {
            "a": {
                "d": {"c": 0},
                "e": {"c": 1},
                "f": {"c": 2},
            },
            "b": {
                "g": 3,
                "h": 4,
            },
        }

        outer = OuterDataclass.from_json(data)

        self.assertIsInstance(outer.a["d"], InnerDataclass)
        self.assertIsInstance(outer.a["e"], InnerDataclass)
        self.assertIsInstance(outer.a["f"], InnerDataclass)

        self.assertEqual(outer.a["d"].c, 0)
        self.assertEqual(outer.a["e"].c, 1)
        self.assertEqual(outer.a["f"].c, 2)
        self.assertEqual(outer.b["g"], 3)
        self.assertEqual(outer.b["h"], 4)

    def test_as_json(self):
        outer = OuterDataclass(
            a=DataclassDict[InnerDataclass](
                d=InnerDataclass(c=0),
                e=InnerDataclass(c=1),
                f=InnerDataclass(c=2),
            ),
            b=DataclassDict[int](
                g=3,
                h=4,
            ),
        )

        self.assertDictEqual(
            outer.as_json(),
            {
                "a": {
                    "d": {"c": 0},
                    "e": {"c": 1},
                    "f": {"c": 2},
                },
                "b": {
                    "g": 3,
                    "h": 4,
                },
            },
        )

    def test_replace(self):
        outer = OuterDataclass(
            a=DataclassDict[InnerDataclass](
                d=InnerDataclass(c=0),
                e=InnerDataclass(c=1),
                f=InnerDataclass(c=2),
            ),
            b=DataclassDict[int](
                g=3,
                h=4,
            ),
        )

        outer = outer.replace(
            a=dict(
                e=dict(c=6),
            ),
            b=dict(
                g=8,
            ),
        )

        self.assertEqual(outer.a["e"].c, 6)
        self.assertEqual(outer.b["g"], 8)

        # A dict has no fixed schema, so replace ADDS keys it does not already have by default
        added = outer.replace(a=dict(h=dict(c=3)))
        self.assertEqual(added.a["h"].c, 3)
        self.assertEqual(added.a["e"].c, 6)  # existing entries survive

        # strict=True is still available where only existing values may be overwritten
        with self.assertRaises(KeyError):
            outer.a.replace(strict=True, h=dict(c=3))

    def test_replace_casts_plain_containers(self):
        # A Dict[str, List[...]] field holds DataclassList values, so a plain list is the natural thing
        # to pass. replace() has to cast it exactly as the constructor does
        outer = RecursiveDictDataclass.from_json({"dd": {"a": {"x": 1}}, "dl": {"b": [1, 2]}})

        outer = outer.replace(dl=dict(b=[3, 4]))

        self.assertIsInstance(outer.dl["b"], DataclassList)
        self.assertEqual(outer.dl["b"].as_json(), [3, 4])

    def test_make_empty(self):
        empty = OuterDataclass.make_empty()

        self.assertIsInstance(empty.a, DataclassDict[InnerDataclass])
        self.assertIsInstance(empty.b, DataclassDict[int])

        self.assertDictEqual(empty.a.as_json(), dict())
        self.assertDictEqual(empty.b.as_json(), dict())

    def test_apply(self):
        outer = OuterDataclass(
            a=DataclassDict[InnerDataclass](
                d=InnerDataclass(c=0),
                e=InnerDataclass(c=1),
                f=InnerDataclass(c=2),
            ),
            b=DataclassDict[int](
                g=3,
                h=4,
            ),
        )

        outer = outer.apply(lambda x: x + 1)

        self.assertDictEqual(
            outer.as_json(),
            {
                "a": {
                    "d": {"c": 1},
                    "e": {"c": 2},
                    "f": {"c": 3},
                },
                "b": {
                    "g": 4,
                    "h": 5,
                },
            },
        )

    def test_items(self):
        data = DataclassDict[InnerDataclass](
            d=InnerDataclass(c=0),
            e=InnerDataclass(c=1),
            f=InnerDataclass(c=2),
        )

        self.assertDictEqual(
            {
                "d": InnerDataclass(c=0),
                "e": InnerDataclass(c=1),
                "f": InnerDataclass(c=2),
            },
            {key: value for key, value in data.items()},
        )

        data = OuterDataclass(
            a=DataclassDict[InnerDataclass](
                d=InnerDataclass(c=0),
                e=InnerDataclass(c=1),
                f=InnerDataclass(c=2),
            ),
            b=DataclassDict[int](
                g=3,
                h=4,
            ),
        )

        self.assertDictEqual(
            {
                "a.d.c": 0,
                "a.e.c": 1,
                "a.f.c": 2,
                "b.g": 3,
                "b.h": 4,
            },
            {key: value for key, value in data.items(recursive=True)},
        )

    def test_frozen(self):
        data = {
            "a": {
                "d": {"c": 0},
                "e": {"c": 1},
                "f": {"c": 2},
            },
            "b": {
                "g": 3,
                "h": 4,
            },
        }

        outer = OuterDataclass.from_json(data)

        with self.assertRaises(TypeError):
            outer.a["d"] = InnerDataclass(c=5)

    def test_recursive_dict_of_dict(self):
        outer = RecursiveDictDataclass.from_json({"dd": {"a": {"x": 1}}, "dl": {}})

        # The plain Dict[str, Dict[str, int]] field is wrapped recursively: both the outer field and
        # the nested level become DataclassDict, not plain dicts
        self.assertIsInstance(outer.dd, DataclassDict)
        self.assertIsInstance(outer.dd["a"], DataclassDict)
        # apply traverses all the way to the leaves
        self.assertDictEqual(outer.dd.apply(lambda x: x * 2).as_json(), {"a": {"x": 2}})
        # Round-trips back to plain nested dicts
        self.assertDictEqual(outer.as_json()["dd"], {"a": {"x": 1}})

    def test_recursive_dict_of_list(self):
        outer = RecursiveDictDataclass.from_json({"dd": {}, "dl": {"a": [1, 2], "b": [3]}})

        self.assertIsInstance(outer.dl, DataclassDict)
        self.assertIsInstance(outer.dl["a"], DataclassList)
        self.assertDictEqual(outer.dl.apply(lambda x: x + 1).as_json(), {"a": [2, 3], "b": [4]})
        self.assertDictEqual(outer.as_json()["dl"], {"a": [1, 2], "b": [3]})

    def test_value_t_is_substituted(self):
        self.assertIs(DataclassDict[Dict[str, int]].ValueT, DataclassDict[int])
        self.assertIs(DataclassDict[List[int]].ValueT, DataclassList[int])

    def test_direct_construction_recurses(self):
        nested = DataclassDict[List[int]]({"a": [1, 2]})
        self.assertIsInstance(nested["a"], DataclassList)

    def test_repr(self):
        self.assertEqual(
            f"{DataclassDict[int]({'a': 5, 'b': 6, 'c': 7})}",
            # fmt: off
            "DataclassDict[int]("
            "\n  a=5,"
            "\n  b=6,"
            "\n  c=7,"
            "\n)",
            # fmt: on
        )

        self.assertEqual(
            f"{DataclassDict[int]([])}",
            "DataclassDict[int]()",
        )


if __name__ == '__main__':
    unittest.main()
