# ruff: noqa: E721 - these tests intentionally assert exact type-object equality (e.g. resolved
# template TypeVars == a concrete type), for which `==` is the clearest assertion.
import pickle
from dataclasses import dataclass, fields
from typing import Dict, List, TypeVar, get_type_hints

import pytest

from ladder.core.databib.template.template import Template
from ladder.core.databib.template.template_dataclass import TemplateDataclass

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@dataclass
class ExampleDataClass:
    field_1: int


@dataclass
class TemplatedClass(TemplateDataclass, Template[T1]):
    a: T1


class ActualClass(TemplatedClass[str]):
    pass


class TestTemplateDataclassMeta:
    def test_overwriting_dataclass_fields(self):
        @dataclass
        class B1(TemplateDataclass, Template[T1, T2]):
            a: T1
            b: T2

        @dataclass
        class B2(B1[T1, T3], Template[T1, T3]):
            pass

        @dataclass
        class AnActualClass(B2[ExampleDataClass, float]):
            pass

        # Make sure all __dataclass_field__ are correctly updated
        assert {f.name: f.type for f in fields(AnActualClass)} == {"a": ExampleDataClass, "b": float}

        assert AnActualClass.T1 == ExampleDataClass
        assert AnActualClass.T2 == float
        assert AnActualClass.T3 == float

        # Make sure __annotations__ gets correctly updated
        assert B1[T1, T3].__annotations__["a"] == T1
        assert B1[T1, T3].__annotations__["b"] == T3

        assert B2[ExampleDataClass, float].__annotations__["a"] == ExampleDataClass
        assert B2[ExampleDataClass, float].__annotations__["b"] == float

    def test_overwriting_dataclass_fields_for_dict_and_list(self):
        @dataclass
        class B1(TemplateDataclass, Template[T1, T2]):
            a: Dict[str, T1]
            b: List[T2]

        @dataclass
        class C1(B1[int, float]):
            pass

        # Make sure all __dataclass_field__ are correctly updated
        assert {f.name: f.type for f in fields(C1)} == {"a": Dict[str, int], "b": List[float]}

        assert C1.T1 == int
        assert C1.T2 == float

        @dataclass
        class B2(B1[T1, T3], Template[T1, T3]):
            pass

        @dataclass
        class C2(B2[ExampleDataClass, float]):
            pass

        # Make sure all __dataclass_field__ are correctly updated
        assert {f.name: f.type for f in fields(C2)} == {"a": Dict[str, ExampleDataClass], "b": List[float]}

        assert C2.T1 == ExampleDataClass
        assert C2.T2 == float

        # Make sure __annotations__ gets correctly updated
        assert B1[T1, T3].__annotations__["a"] == Dict[str, T1]
        assert B1[T1, T3].__annotations__["b"] == List[T3]

        assert B2[ExampleDataClass, float].__annotations__["a"] == Dict[str, ExampleDataClass]
        assert B2[ExampleDataClass, float].__annotations__["b"] == List[float]

    def test_inheritance_from_non_templated_base(self):
        @dataclass
        class BaseClass(TemplateDataclass, Template[T1, T2]):
            a: T1
            b: T2

        @dataclass
        class AnActualClass(BaseClass[ExampleDataClass, float]):
            pass

        # Make sure AnActualClass.__bases__[0] is BaseClass[ExampleDataClass, float] rather than BaseClass
        assert AnActualClass.__bases__[0].__parameters__ == tuple()

    def test_incorrect_dataclass_field(self):
        BoundT = TypeVar("BoundT", bound=int)

        @dataclass
        class BaseClass(TemplateDataclass, Template[BoundT]):
            a: BoundT

        with pytest.raises(TypeError):

            @dataclass
            class AnActualClass(BaseClass[str]):  # type:ignore
                pass

    def test_overwriting_works_with_inheritance(self):
        @dataclass
        class BaseClass(TemplateDataclass, Template[T1]):
            a: T1

            def some_method(self):
                return "ohai"

        @dataclass
        class ATemplatedClass(BaseClass[T1], Template[T1, T2]):
            b: T2

        AnActualClass = ATemplatedClass[ExampleDataClass, float]

        assert {f.name: f.type for f in fields(AnActualClass)} == {"a": ExampleDataClass, "b": float}
        assert get_type_hints(AnActualClass)["a"] == ExampleDataClass
        assert AnActualClass.__annotations__["b"] == float

        instance = AnActualClass(a=None, b=None)
        assert instance.some_method() == "ohai"

    def test_isinstance_works_with_concrete_type(self):
        @dataclass
        class ATemplatedClass(TemplateDataclass, Template[T1]):
            a: T1

        class AnActualClass(ATemplatedClass[str]):
            pass

        instance = AnActualClass(a="hello")
        assert isinstance(instance, ATemplatedClass)

    def test_dataclass_with_templated_and_non_templated_fields(self):
        @dataclass
        class TemplatedClass1(TemplateDataclass, Template[T1]):
            a: T1

        @dataclass
        class ActualClass1(TemplatedClass1[str]):
            b: int

        assert {f.name: f.type for f in fields(ActualClass1)} == {"a": str, "b": int}

        @dataclass
        class TemplatedClass2(TemplateDataclass, Template[T1]):
            a: T1
            b: float

        @dataclass
        class ActualClass2(TemplatedClass2[str]):
            pass

        assert {f.name: f.type for f in fields(ActualClass2)} == {"a": str, "b": float}

    def test_instances_are_pickleable(self):
        instance = ActualClass(a="hello")
        pickled = pickle.dumps(instance)
        unpickled = pickle.loads(pickled)
        assert unpickled.a == "hello"

    def test_dynamic_instances_are_pickleable(self):
        instance = TemplatedClass[str](a="hello")
        pickled = pickle.dumps(instance)
        unpickled = pickle.loads(pickled)
        assert unpickled.a == "hello"

    def test_templated_dataclasses_are_instances_of_object(self):
        assert issubclass(TemplatedClass[float], object)
        assert issubclass(ActualClass, object)
        assert isinstance(TemplatedClass[float](a=0.5), object)
        assert isinstance(ActualClass(a="test"), object)

    def test_multiple_subclasses_of_templated_dataclass(self):
        @dataclass
        class TemplatedBase(TemplateDataclass, Template[T1]):
            c: T1
            d: str

        class A:
            pass

        class B:
            pass

        SubclassA = TemplatedBase[A]
        SubclassB = TemplatedBase[B]

        assert TemplatedBase.__dataclass_fields__["c"].type == T1
        assert TemplatedBase.__dataclass_fields__["d"].type == str
        assert SubclassA.__dataclass_fields__["c"].type == A
        assert SubclassA.__dataclass_fields__["d"].type == str
        assert SubclassB.__dataclass_fields__["c"].type == B
        assert SubclassB.__dataclass_fields__["d"].type == str

    def test_caching_template_getitem(self):
        @dataclass
        class TemplatedClass1(TemplateDataclass, Template[T1, T2]):
            field_1: T1
            field_2: T2

        actual_classes = [
            TemplatedClass1[T1, T2],
            TemplatedClass1[T1, str],
            TemplatedClass1[int, T2],
            TemplatedClass1[int, str],
        ]

        another_actual_classes = [
            TemplatedClass1[T1, T2],
            TemplatedClass1[T1, str],
            TemplatedClass1[int, T2],
            TemplatedClass1[int, str],
        ]

        for actual_class, another_actual_class in zip(actual_classes, another_actual_classes, strict=True):
            assert another_actual_class is actual_class

    def test_missing_template_args(self):
        @dataclass
        class A(TemplateDataclass, Template[T1, T2]):
            field_1: T1
            field_2: T2

        @dataclass
        class B(TemplateDataclass, Template[T1, T2]):
            field_1: T1

        with pytest.raises(TypeError):

            @dataclass
            class C(TemplateDataclass, Template[T1]):
                field_1: T1
                field_2: T2

        with pytest.raises(TypeError):

            @dataclass
            class D(A, Template[T1, T2]):
                field_3: T3

        with pytest.raises(TypeError):

            @dataclass
            class E(A, Template[T1, T3]):
                field_3: T3

        # Test Dict fields
        with pytest.raises(TypeError):

            @dataclass
            class F(A, Template[T1, T2]):
                field_3: Dict[str, T3]

        # Test List fields
        with pytest.raises(TypeError):

            @dataclass
            class G(A, Template[T1, T2]):
                field_3: List[T3]
