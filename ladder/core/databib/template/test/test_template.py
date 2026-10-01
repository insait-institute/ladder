# ruff: noqa: E721 - these tests intentionally assert exact type-object equality (e.g. resolved
# template TypeVars == a concrete type), for which `==` is the clearest assertion.
import pickle
from dataclasses import dataclass
from typing import Generic, TypeVar

import pytest

from ladder.core.databib.template.template import Template
from ladder.core.databib.utils.classproperty import cached_classproperty, classproperty

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
FwdBoundT = TypeVar("FwdBoundT", bound="Bound")


class Bound:
    pass


class TestTemplateMeta:
    # pylint:disable=no-self-argument,pointless-statement

    def test_inheritance_from_template_and_generic_is_forbidden(self):
        with pytest.raises(TypeError):

            class TemplateBase(Template[T1]):
                pass

            class TemplatedClass(TemplateBase[T1], Template[T1], Generic[T1]):
                pass

        with pytest.raises(TypeError):

            class AnotherTemplatedClass(Template[T1], Generic[T1]):
                pass

    def test_raises_on_missing_type_vars_from_template(self):
        class A(Template[T1, T2]):
            pass

        with pytest.raises(TypeError):

            class B(A[T1, T2], Template[T1]):
                pass

    def test_raises_on_incorrect_inheritance(self):
        with pytest.raises(TypeError):

            class A(Template):
                pass

        with pytest.raises(TypeError):

            class B(Template[T1], Template[T2]):
                pass

    def test_template_attributes_are_correctly_set(self):
        class TemplatedClass(Template[T1, T2]):
            pass

        assert TemplatedClass.__parameters__ == (T1, T2)
        assert TemplatedClass.__type_var_attrs__ == {T1: T1, T2: T2}

        Derived1 = TemplatedClass[str, int]

        assert Derived1.__parameters__ == tuple()
        assert Derived1.__args__ == (str, int)
        assert Derived1.__origin__ == TemplatedClass
        assert Derived1.__type_var_attrs__ == {T1: str, T2: int}

        class Derived2(TemplatedClass[str, int]):
            pass

        assert Derived2.__parameters__ == tuple()
        assert Derived2.__args__ is None
        assert Derived2.__origin__ is None
        assert Derived2.__type_var_attrs__ == {T1: str, T2: int}

        Derived3 = TemplatedClass[T3, T3]

        assert Derived3.__parameters__ == (T3, T3)
        assert Derived3.__args__ == (T3, T3)
        assert Derived3.__origin__ == TemplatedClass
        assert Derived3.__type_var_attrs__ == {T1: T3, T2: T3, T3: T3}

    def test_type_variable_instances_are_saved_on_class(self):
        class TemplatedClass(Template[T1, T2]):
            a: T1
            b: T2

            @classproperty
            def type_of_b(cls):  # noqa: N805
                return cls.T2

        StrClass = TemplatedClass[str, bytes]
        FloatClass = TemplatedClass[float, bytes]

        assert StrClass is not FloatClass
        assert StrClass.T1 is str
        assert StrClass.T2 is bytes
        assert FloatClass.T1 is float
        assert FloatClass.T2 is bytes

    def test_type_variable_instances_are_saved_on_instances(self):
        class TemplatedBase1(Template[T1]):
            a: T1

        StrClass = TemplatedBase1[str]
        str_instance = StrClass()

        assert str_instance.T1 is str

        class TemplatedBase2(Template[T1]):
            a: T1

        class TemplatedClass2(TemplatedBase2[T1], Template[T1]):
            a: T1

        assert TemplatedClass2[str]().T1 is str

        class TemplatedBase3(Template[T1]):
            pass

        class TemplatedClass3(TemplatedBase3[T1], Template[T1]):
            pass

        assert TemplatedClass3[str]().T1 is str

    def test_type_variables_are_saved_transitively(self):
        class TemplatedClass(Template[T1]):
            a: T1

        class TemplatedSubclass(TemplatedClass[T2], Template[T2]):
            pass

        StrClass = TemplatedSubclass[str]
        FloatClass = TemplatedSubclass[float]

        assert StrClass.T1 is str
        assert FloatClass.T1 is float
        assert StrClass.T2 is str
        assert FloatClass.T2 is float

        class TemplatedSubclass2(TemplatedSubclass[T3], Template[T3]):
            pass

        StrClass2 = TemplatedSubclass2[str]
        assert StrClass2.T1 is str
        assert StrClass2.T2 is str
        assert StrClass2.T3 is str

    def test_type_variables_can_be_replaced_with_same_name(self):
        class TemplatedClass(Template[T1]):
            a: T1

        NewT1 = TypeVar("NewT1")

        class TemplatedSubclass(TemplatedClass[NewT1], Template[NewT1]):
            pass

        StrClass = TemplatedSubclass[str]
        assert StrClass.T1 is str

    def test_instantiating_templated_classes(self):
        class TemplatedClass(Template[T1]):
            pass

        with pytest.raises(TypeError):
            TemplatedClass()

        with pytest.raises(TypeError):

            class InvalidClass(TemplatedClass):
                pass

    def test_that_redefining_already_bound_type_vars_is_not_allowed(self):
        class Unbound(Template[T1]):
            pass

        with pytest.raises(TypeError):

            class Rebound(Unbound[str], Template[T1]):
                pass

    def test_passing_types_to_template(self):
        with pytest.raises(TypeError):
            Template[int]

        with pytest.raises(TypeError):
            Template[T1, str]

    def test_class_properties_on_superclasses_work(self):
        class TemplatedClass(Template[T1]):
            pass

        class TemplatedSubClass1(TemplatedClass[T1], Template[T1, T2]):
            @classproperty
            def type_1(cls):  # noqa: N805
                return cls.T1

            @classproperty
            def type_2(cls):  # noqa: N805
                return cls.T2

        class TemplatedSubClass2(TemplatedSubClass1[T1, T2], Template[T1, T2]):
            pass

        class TemplatedSubClassChangingSuperclassTypes(TemplatedSubClass1[T3, T3], Template[T3]):
            pass

        class ActualClass(TemplatedSubClass2[str, int]):
            pass

        assert ActualClass.type_1 == str
        assert ActualClass.type_2 == int

    def test_class_properties_work_when_subclassing_with_a_new_type(self):
        class TemplatedClass(Template[T1]):
            pass

        class TemplatedSubClass(TemplatedClass[int], Template[T2]):
            @classproperty
            def type_2(cls):  # noqa: N805
                return cls.T2

        class ActualClass(TemplatedSubClass[str]):
            pass

        assert ActualClass.T1 == int
        assert ActualClass.type_2 == str

    def test_bounds_check_works_with_forward_declarations(self):
        class TemplatedClass(Template[FwdBoundT]):
            pass

        class Derived(Bound):
            pass

        with pytest.raises(TypeError):
            TemplatedClass[int]

        TemplatedClass[Derived]

    def test_crtp_works_with_forward_declarations(self):
        class TemplatedClass(Template[T1]):
            pass

        class CrtpSubclass(TemplatedClass["CrtpSubclass"]):
            pass

        assert isinstance(CrtpSubclass.T1(), CrtpSubclass)

    def test_template_prevents_instances_of_classes_with_forward_ref_args(self):
        class TemplatedClass(Template[T1]):
            pass

        class InvalidClass(TemplatedClass["NonClass"]):
            pass

        with pytest.raises(TypeError):
            InvalidClass()

    def test_template_works_with_nested_attributes(self):
        class AttributeClass:
            nested_attribute = 0

        class AttributeSubclass(AttributeClass):
            nested_attribute = 1

        AttributeClassT = TypeVar("AttributeClassT", bound=AttributeClass)

        class TemplatedClass(Template[T1]):
            pass

        class TemplatedSubclass(TemplatedClass[int], Template[AttributeClassT]):
            @classproperty
            def value_of_nested_attribute(cls):  # noqa: N805
                return cls.AttributeClassT.nested_attribute

        class ActualClass(TemplatedSubclass[AttributeClass]):
            pass

        class AnotherActualClass(TemplatedSubclass[AttributeSubclass]):
            pass

        assert ActualClass.value_of_nested_attribute == 0
        assert AnotherActualClass.value_of_nested_attribute == 1

    def test_cached_classproperty_on_template(self):
        data = {"key": 5}

        T = TypeVar("T")

        @dataclass
        class TestDataClassBase(Template[T]):
            a: T

            @cached_classproperty
            def test_property(cls):  # noqa: N805
                return data["key"]

        @dataclass
        class TestDataClass(TestDataClassBase[float]):
            pass

        assert TestDataClassBase.test_property == 5

        data["key"] = 6

        # Make sure the cached_classproperty wasn't called by TemplateMeta when creating the class
        assert TestDataClass.test_property == 6

    def test_recursive_calls_to_getitem_are_forbidden(self):
        class A(Template[T1, T2, T3]):
            pass

        B = A[int, T2, T3]

        with pytest.raises(TypeError):
            B[float, str]

        with pytest.raises(TypeError):
            B[T2, str]

    def test_pickle_template(self):
        class TemplateBase(Template[T1]):
            def __init__(self, i: T1):
                self.i = i

        TemplateInt = TemplateBase[int]

        template_int_pickled = pickle.dumps(TemplateInt)
        TemplateIntUnpickled = pickle.loads(template_int_pickled)

        assert TemplateIntUnpickled.T1 is int

        template_int_instance = TemplateInt(4)
        template_int_instance_pickled = pickle.dumps(template_int_instance)
        template_int_instance_unpickled = pickle.loads(template_int_instance_pickled)

        assert template_int_instance.i == template_int_instance_unpickled.i

        A = TemplateBase[float]
        TemplateA = TemplateBase[A]

        template_a_pickled = pickle.dumps(TemplateA)
        TemplateAUnpickled = pickle.loads(template_a_pickled)

        assert TemplateAUnpickled.T1 is A

        template_a_instance = TemplateA(A(4))
        template_a_instance_pickled = pickle.dumps(template_a_instance)
        template_a_instance_unpickled = pickle.loads(template_a_instance_pickled)

        assert template_a_instance.i.i == template_a_instance_unpickled.i.i

    def test_caching_template_getitem(self):
        class TemplatedClass(Template[T1, T2]):
            pass

        actual_classes = [
            TemplatedClass[T1, T2],
            TemplatedClass[T1, str],
            TemplatedClass[int, T2],
            TemplatedClass[int, str],
        ]

        another_actual_classes = [
            TemplatedClass[T1, T2],
            TemplatedClass[T1, str],
            TemplatedClass[int, T2],
            TemplatedClass[int, str],
        ]

        for actual_class, another_actual_class in zip(actual_classes, another_actual_classes, strict=True):
            assert another_actual_class is actual_class
