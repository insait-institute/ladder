import unittest
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

from ladder.core.databib.dataclasses import Dataclass, DataclassDict, DataclassList, dataclass
from ladder.core.databib.template import Template

AnyT = TypeVar("AnyT")


class TemplatedClass(Template[AnyT]):
    pass


class ConcreteClass(TemplatedClass[str]):
    pass


@dataclass
class DataclassInner(Dataclass):
    pass


@dataclass
class DataclassInnerSubclass(DataclassInner):
    pass


class MyString(str):
    pass


@dataclass
class Dataclass1(Dataclass):
    a: str
    b: ConcreteClass
    c: DataclassInner


@dataclass
class Dataclass2(Dataclass):
    a: float


class DataclassTest(unittest.TestCase):
    def test_init(self):
        # Correct instantiation
        Dataclass1(a="test", b=ConcreteClass(), c=DataclassInner())

        # Incorrect string field
        with self.assertRaises(TypeError):
            Dataclass1(a=5, b=ConcreteClass(), c=DataclassInner())

        # Incorrect templated class field
        with self.assertRaises(TypeError):
            Dataclass1(a="test", b=5, c=DataclassInner())

        # Incorrect dataclass field
        with self.assertRaises(TypeError):
            Dataclass1(a="test", b=ConcreteClass(), c=5)

        # Incorrect dataclass field with subclass
        with self.assertRaises(TypeError):
            Dataclass1(a="test", b=ConcreteClass(), c=DataclassInnerSubclass())

        # Incorrect string field with subclass
        with self.assertRaises(TypeError):
            Dataclass1(a=MyString(), b=ConcreteClass(), c=DataclassInner())

    def test_replace(self):
        # Correct instantiation and replacement
        obj = Dataclass1(a="test", b=ConcreteClass(), c=DataclassInner())
        obj.replace(a="test", b=ConcreteClass(), c=DataclassInner())

        # Incorrect string field
        with self.assertRaises(TypeError):
            obj.replace(a=5)

        # Incorrect templated class field
        with self.assertRaises(TypeError):
            obj.replace(b=5)

        # Incorrect dataclass field
        with self.assertRaises(TypeError):
            obj.replace(c=5)

        # Incorrect dataclass field with subclass
        with self.assertRaises(TypeError):
            obj.replace(c=DataclassInnerSubclass())

        # Incorrect string field with subclass
        with self.assertRaises(TypeError):
            obj.replace(a=MyString())

    def test_int_downcasting(self):
        # Works with floats
        obj = Dataclass2(a=2.0)
        obj = obj.replace(a=3.0)

        # Works with ints
        obj2 = Dataclass2(a=2)
        obj2 = obj2.replace(a=3)

        # Doesn't work with other types
        with self.assertRaises(TypeError):
            Dataclass2(a="wrong")

    def test_dataclass_allows_np_values_for_int_and_float(self):
        @dataclass
        class A(Dataclass):
            a: int
            b: float

        A(a=np.int64(10), b=np.float32(5.0))
        A(a=np.int64(10.0), b=np.float32(5))

    def test_dataclass_casts_dicts_and_lists(self):
        @dataclass
        class SimpleData(Dataclass):
            data: float

        @dataclass
        class MyData(Dataclass):
            a: int
            b: Dict[str, str]
            c: Dict[str, SimpleData]
            d: List[int]
            e: List[SimpleData]
            f: Tuple[int, float]

        my_data = MyData(
            a=5,
            b={"ba": "bb", "bc": "bd"},
            c={"ca": SimpleData(data=10.0)},
            d=[7, 8, 9],
            e=[SimpleData(data=11.0), SimpleData(data=12.0)],
            f=(5, 8.0),
        )
        self.assertIsInstance(my_data.b, DataclassDict)
        self.assertIsInstance(my_data.c, DataclassDict)
        self.assertIsInstance(my_data.d, DataclassList)
        self.assertIsInstance(my_data.e, DataclassList)

        self.assertEqual(my_data.b, DataclassDict[str]({"ba": "bb", "bc": "bd"}))
        self.assertEqual(my_data.c, DataclassDict[SimpleData]({"ca": SimpleData(data=10.0)}))
        self.assertEqual(my_data.d, DataclassList[int]([7, 8, 9]))
        self.assertEqual(my_data.e, DataclassList[SimpleData]([SimpleData(data=11.0), SimpleData(data=12.0)]))

        my_data = MyData.from_json(my_data.as_json())
        self.assertIsInstance(my_data.b, DataclassDict)
        self.assertIsInstance(my_data.c, DataclassDict)
        self.assertIsInstance(my_data.d, DataclassList)
        self.assertIsInstance(my_data.e, DataclassList)

        self.assertEqual(my_data.b, DataclassDict[str]({"ba": "bb", "bc": "bd"}))
        self.assertEqual(my_data.c, DataclassDict[SimpleData]({"ca": SimpleData(data=10.0)}))
        self.assertEqual(my_data.d, DataclassList[int]([7, 8, 9]))
        self.assertEqual(my_data.e, DataclassList[SimpleData]([SimpleData(data=11.0), SimpleData(data=12.0)]))

    def test_dataclass_decorator_type_checks(self):
        # Make sure TypeVars are allowed
        @dataclass
        class A(Dataclass, Template[AnyT]):
            a: int
            b: float
            c: AnyT

        # Make sure Union is allowed
        @dataclass
        class B(Dataclass):
            a: Union[int, str]

        # Make sure Dict, List and Tuple annotations are allowed
        @dataclass
        class D(Dataclass):
            a: Dict[str, str]
            b: Dict[str, Dataclass]
            c: List[str]
            d: List[Dataclass]
            e: Tuple[str, str]

        # Make sure forbidden Dict key types are not allowed
        with self.assertRaises(TypeError):

            @dataclass
            class K(Dataclass):
                a: Dict[int, Dataclass]

        # Make sure DataclassDict and DataclassList are allowed
        @dataclass
        class L(Dataclass):
            a: DataclassDict[int]
            b: DataclassList[int]

    def test_forbidden_union_and_tuple_annotations(self):
        # Mirrors the forbidden annotations listed in docs/databib/dataclasses.md. A Union of Union is not
        # part of the list since `int | Union[str, float]` flattens to `Union[int, str, float]`.
        @dataclass
        class Inner(Dataclass):
            a: int

        forbidden_annotations = {
            # Tuple with a Dataclass, iterable or TypeVar member
            "Tuple[Dataclass, ...]": Tuple[Inner, ...],
            "Tuple[str, Dataclass]": Tuple[str, Inner],
            "Tuple[str, Dict]": Tuple[str, Dict[str, Any]],
            "Tuple[str, List]": Tuple[str, List[Any]],
            "Tuple[str, Tuple]": Tuple[str, Tuple[str, str]],
            "Tuple[str, TypeVar]": Tuple[str, AnyT],
            # Union with a Dataclass, iterable or TypeVar member
            "Union[Dataclass, Dict]": Union[Inner, Dict[str, Any]],
            "Union[Dataclass, List]": Union[Inner, List[Any]],
            "Union[Dataclass, int]": Union[Inner, int],
            "Union[Dataclass, TypeVar]": Union[Inner, AnyT],
            "Union[Dataclass, Set]": Union[Inner, Set],
            "Union[int, TypeVar]": Union[int, AnyT],
            "Union[Dict, List]": Union[Dict[str, Any], List[Any]],
            # Optional does not lift the restriction
            "Optional[Union[Dataclass, int]]": Optional[Union[Inner, int]],
            "Optional[Tuple[Dataclass, ...]]": Optional[Tuple[Inner, ...]],
        }
        for name, annotation in forbidden_annotations.items():
            with self.subTest(name), self.assertRaisesRegex(TypeError, "unsafe and not allowed"):

                @dataclass
                class Forbidden(Dataclass):
                    a: annotation


if __name__ == '__main__':
    unittest.main()
