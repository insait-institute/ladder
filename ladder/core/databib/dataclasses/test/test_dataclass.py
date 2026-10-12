import os
import pickle
import tempfile
import unittest
from typing import Optional

import numpy as np
import torch

from ladder.core.databib.dataclasses.dataclass import Dataclass, dataclass

from .helper import MemPickleDataclass, PickleDataclass, worker_pickle_dump


@dataclass
class MockDataclass(Dataclass):
    a: str
    b: str


@dataclass
class MockNestedDataclass(Dataclass):
    c: MockDataclass
    d: MockDataclass


@dataclass
class MockDoubleNestedDataclass(Dataclass):
    e: MockNestedDataclass
    f: MockNestedDataclass


@dataclass
class MockNumpyDataclass(Dataclass):
    a: int
    b: float


class DataclassTest(unittest.TestCase):
    def test_from_dict(self):
        dc = MockNestedDataclass.from_dict({"c": {"a": "c.a", "b": "c.b"}, "d": {"a": "d.a", "b": "d.b"}})
        self.assertEqual(dc.c.a, "c.a")
        self.assertEqual(dc.c.b, "c.b")
        self.assertEqual(dc.d.a, "d.a")
        self.assertEqual(dc.d.b, "d.b")

    def test_as_dict(self):
        d = {"c": {"a": "c.a", "b": "c.b"}, "d": {"a": "d.a", "b": "d.b"}}
        self.assertDictEqual(MockNestedDataclass.from_dict(d).as_dict(), d)

    def test_from_dict_throws_with_too_few_parameters(self):
        with self.assertRaises(Exception):  # noqa: B017
            MockDataclass.from_dict({"a": "A"})

    def test_from_dict_throws_with_too_many_parameters(self):
        with self.assertRaises(Exception):  # noqa: B017
            MockDataclass.from_dict({"a": "A", "b": "B", "c": "C"})

    def test_comparison(self):
        self.assertNotEqual(
            MockDataclass(a='A', b='B'),
            MockDataclass(a='A', b='B2'),
        )

    def test_replace(self):
        data = MockNestedDataclass(
            c=MockDataclass(a='A', b='B'),
            d=MockDataclass(a='A', b='B2'),
        )

        # Test replacing via dict
        replaced = data.replace(c={'b': 'B3'})
        self.assertEqual(replaced.c.b, 'B3')
        self.assertEqual(replaced.c.a, 'A')

        # Test replacing via dataclass
        replaced = data.replace(d=MockDataclass(a='A4', b='B4'))
        self.assertEqual(replaced.d.b, 'B4')
        self.assertEqual(replaced.d.a, 'A4')

        data = MockDoubleNestedDataclass(
            e=MockNestedDataclass(
                c=MockDataclass(a='A1', b='B1'),
                d=MockDataclass(a='A2', b='B2'),
            ),
            f=MockNestedDataclass(
                c=MockDataclass(a='A3', b='B3'),
                d=MockDataclass(a='A4', b='B4'),
            ),
        )

        # Test replacing via chained arguments and dicts
        replaced = data.replace(
            **{
                "e.c.a": "A11",
                "e": {"d": {"b": "B12"}},
                "f.c.a": "A13",
                "f.c": {"b": "B13"},
                "f": {"d": {"a": "A14"}},
            },
        )
        self.assertEqual(replaced.e.c.a, 'A11')
        self.assertEqual(replaced.e.d.b, 'B12')
        self.assertEqual(replaced.f.c.a, 'A13')
        self.assertEqual(replaced.f.c.b, 'B13')
        self.assertEqual(replaced.f.d.a, 'A14')

        # Test replacing via chained arguments and mixture of dicts and dataclass
        replaced = data.replace(
            **{
                "f.c": MockDataclass(a='A23', b='B23'),
                "f": {"d": {"a": "A24"}},
            },
        )
        self.assertEqual(replaced.f.c.a, 'A23')
        self.assertEqual(replaced.f.c.b, 'B23')
        self.assertEqual(replaced.f.d.a, 'A24')

        # Test that passing multiple values for the same key throws an error
        with self.assertRaises(ValueError):
            data.replace(
                **{
                    "f.c.a": "A33",
                    "f.c": MockDataclass(a='A23', b='B23'),
                },
            )

        with self.assertRaises(ValueError):
            data.replace(
                **{
                    "f.c": MockDataclass(a='A23', b='B23'),
                    "f.c.a": "A33",
                },
            )

        with self.assertRaises(ValueError):
            data.replace(
                **{
                    "f": {"c": {"a": "A44"}},
                    "f.c.a": "A33",
                },
            )

    def test_apply(self):
        data = MockNestedDataclass(
            c=MockDataclass(a='A', b='B'),
            d=MockDataclass(a='A1', b='B1'),
        )

        def add_str(value):
            return value + '2'

        applied = data.apply(add_str)

        self.assertEqual(applied.c.a, 'A2')
        self.assertEqual(applied.c.b, 'B2')
        self.assertEqual(applied.d.a, 'A12')
        self.assertEqual(applied.d.b, 'B12')

    def test_nested_optional(self):
        @dataclass
        class Nested(Dataclass):
            a: str

        @dataclass
        class Outer(Dataclass):
            optional_nested: Optional[Nested]

        outer = Outer.from_dict({'optional_nested': {'a': 'test'}})

        self.assertEqual(outer.optional_nested.a, 'test')

        outer_without_optional = Outer.from_dict({'optional_nested': None})
        self.assertEqual(outer_without_optional.optional_nested, None)

    def test_items(self):
        data = MockNestedDataclass(
            c=MockDataclass(a='A', b='B'),
            d=MockDataclass(a='A', b='B2'),
        )

        self.assertDictEqual(
            {"c": MockDataclass(a='A', b='B'), "d": MockDataclass(a='A', b='B2')},
            {key: value for key, value in data.items()},
        )

        self.assertDictEqual(
            {"c.a": "A", "c.b": "B", "d.a": "A", "d.b": "B2"},
            {key: value for key, value in data.items(recursive=True)},
        )

    def test_getattr(self):
        data = MockDoubleNestedDataclass(
            e=MockNestedDataclass(
                c=MockDataclass(a='A1', b='B1'),
                d=MockDataclass(a='A2', b='B2'),
            ),
            f=MockNestedDataclass(
                c=MockDataclass(a='A3', b='B3'),
                d=MockDataclass(a='A4', b='B4'),
            ),
        )

        self.assertEqual(data.getattr('e'), data.e)
        self.assertEqual(data.getattr('e.c'), MockDataclass(a='A1', b='B1'))
        self.assertEqual(data.getattr('e.c.a'), 'A1')
        self.assertEqual(data.getattr('f.d.b'), 'B4')

        with self.assertRaises(AttributeError):
            data.getattr('e.missing')

    def test_is_empty(self):
        # Test single dataclass
        data = MockDataclass.make_empty()
        self.assertTrue(data.is_empty)
        data = MockDataclass(a=None, b='hello')
        self.assertFalse(data.is_empty)

        # Test nested
        data = MockNestedDataclass.make_empty()
        self.assertTrue(data.is_empty)
        data = MockNestedDataclass(c=MockDataclass.make_empty(), d=MockDataclass(a=None, b='hello'))
        self.assertFalse(data.is_empty)

    def test_numpy_types(self):
        data1 = MockNumpyDataclass(a=1, b=2.0)
        data2 = MockNumpyDataclass(a=np.int64(1), b=np.float64(2.0))
        self.assertEqual(data1, data2)

    def test_shape_repr(self):
        @dataclass
        class MockNpDataclass(Dataclass):
            a: str
            b: np.ndarray
            c: np.ndarray

        @dataclass
        class MockNestedNpDataclass(Dataclass):
            e: MockNpDataclass
            f: MockNpDataclass
            g: int

        data = MockNestedNpDataclass(
            e=MockNpDataclass(
                a='A',
                b=np.zeros((2, 4)),
                c=np.ones((2, 3)),
            ),
            f=MockNpDataclass(
                a='B',
                b=np.zeros((3, 3)),
                c=np.ones((3, 5)),
            ),
            g=5,
        )

        self.assertEqual(
            data.shape_repr(),
            "MockNestedNpDataclass(\n"
            "  e: MockNpDataclass(\n"
            "    a: -X-,\n"
            "    b: [2, 4],\n"
            "    c: [2, 3],\n"
            "  ),\n"
            "  f: MockNpDataclass(\n"
            "    a: -X-,\n"
            "    b: [3, 3],\n"
            "    c: [3, 5],\n"
            "  ),\n"
            "  g: -X-,\n"
            ")",
        )

        self.assertEqual(
            data.replace(f=MockNpDataclass.make_empty()).shape_repr(),
            "MockNestedNpDataclass(\n"
            "  e: MockNpDataclass(\n"
            "    a: -X-,\n"
            "    b: [2, 4],\n"
            "    c: [2, 3],\n"
            "  ),\n"
            "  f: MockNpDataclass.empty(),\n"
            "  g: -X-,\n"
            ")",
        )

        self.assertEqual(
            MockNestedNpDataclass.make_empty().shape_repr(),
            "MockNestedNpDataclass.empty()",
        )

    def test_dataclass_pickles_single_process(self):
        # Make sure that Dataclass with Dict and List fields can pickle in a single process
        data = PickleDataclass(
            str_name='test',
            dict_int={'1': 2, '3': '4'},
            list_int=[1, 2, '3', 4],
            dict_dict={'1': {'2': 3}, '4': {'5': 6}},
            list_list=[[5.6, 7.8], [10.11, 13.14]],
            p_data=MemPickleDataclass[int](value=10),
        )

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, 'output.pickle')

            with open(file_path, 'wb') as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

            with open(file_path, 'rb') as file:
                p_data = pickle.load(file)

            assert data == p_data

    def test_dataclass_pickles_multiprocessing(self):
        # Make sure that Dataclass with Dict and List fields can pickle when using torch
        data = PickleDataclass(
            str_name='test',
            dict_int={'1': 2, '3': '4'},
            list_int=[1, 2, '3', 4],
            dict_dict={'1': {'2': 3}, '4': {'5': 6}},
            list_list=[[5.6, 7.8], [10.11, 13.14]],
            p_data=MemPickleDataclass[int](value=10),
        )

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, 'output.pickle')

            torch.multiprocessing.spawn(worker_pickle_dump, args=(file_path, data), nprocs=1, join=True)

            with open(file_path, 'rb') as file:
                worker_data = pickle.load(file)

            assert data == worker_data


if __name__ == '__main__':
    unittest.main()
