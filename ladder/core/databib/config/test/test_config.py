import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeVar

import pytest
from backports.strenum import StrEnum
from frozendict import frozendict

from ladder.core.databib.config.auto_config import AutoConfig
from ladder.core.databib.config.config import (
    Config,
    Configurable,
    InvalidConfigAttributeTypeError,
    MissingConfigAttributeError,
    UnexpectedConfigAttributeError,
)
from ladder.core.databib.config.yaml_util import _load_raw_yaml_with_import_tree
from ladder.core.databib.template import Template

CONFIG_FILES_DIR = os.path.join(os.path.dirname(__file__), 'config_files')


class ExampleConfig(Config):
    a: int
    b: int = 0
    c: int


class ExampleNestedConfig(Config):
    inner: ExampleConfig
    a_nullable_value: Optional[ExampleConfig] = None
    inner_list: List[int]


class ExampleDoubleNestedConfig(Config):
    outer: ExampleNestedConfig
    root_value_1: int
    root_value_2: int


def test_too_few_arguments_at_init():
    with pytest.raises(Exception):  # noqa: B017
        ExampleConfig(a=1)

    with pytest.raises(MissingConfigAttributeError):
        ExampleConfig(b=2, c=3)


def test_too_many_arguments_at_init():
    with pytest.raises(Exception):  # noqa: B017
        ExampleConfig(a=1, c=3, d=4)


def test_default_values():
    config = ExampleConfig(a=1, c=3)
    assert config.a == 1
    assert config.b == 0
    assert config.c == 3

    config = ExampleConfig(a=1, b=2, c=3)
    assert config.a == 1
    assert config.b == 2
    assert config.c == 3


def test_as_dict():
    assert ExampleConfig(a=1, c=3).as_dict() == {'a': 1, 'b': 0, 'c': 3}
    assert ExampleConfig(a=1, b=2, c=3).as_dict() == {'a': 1, 'b': 2, 'c': 3}


def test_post_init():
    class ExamplePostInitConfig(ExampleConfig):
        def __post_init__(self):
            super().__post_init__()
            self.a = 2 * self.b
            self.c = self.b + 1

    assert ExamplePostInitConfig(a=1, c=5).as_dict() == {'a': 0, 'b': 0, 'c': 1}


def test_non_idempotent_post_init():
    class ExampleInvalidPostInitConfig:
        a: 0

        def __post_init__(self):
            super().__post_init__()
            self.a = self.a + 1

    with pytest.raises(Exception):  # noqa: B017
        ExampleInvalidPostInitConfig(a=1, c=3)

    with pytest.raises(Exception):  # noqa: B017
        ExampleInvalidPostInitConfig(a=1, b=2, c=3)


def test_frozen():
    config = ExampleConfig(a=1, c=3)

    with pytest.raises(Exception):  # noqa: B017
        config.a = 2

    with pytest.raises(Exception):  # noqa: B017
        config.b = 3


def test_from_yaml():
    config = ExampleDoubleNestedConfig.from_yaml(os.path.join(CONFIG_FILES_DIR, 'a.yaml'), save_path=False)
    expected = {
        'outer': {
            'inner': {'a': 1, 'b': 1, 'c': 1},
            'inner_list': [7, 8, 9],
            'a_nullable_value': None,
        },
        'root_value_1': 10,
        'root_value_2': 8,
    }
    assert config.as_dict() == expected


def test_dependency_tree():
    _, dep_tree = _load_raw_yaml_with_import_tree(os.path.join(CONFIG_FILES_DIR, 'a.yaml'))
    assert {
        os.path.join(CONFIG_FILES_DIR, 'b.yaml'): {
            os.path.join(CONFIG_FILES_DIR, 'c.yaml'): {},
        },
    } == dep_tree


def test_generic_config_is_invalid():
    T = TypeVar('T')

    with pytest.raises(TypeError):

        class TestGenericConfig(Config, Template[T]):  # pylint: disable=unused-variable
            my_list: List[T] = []


def test_mutable_default_attribute_gets_frozen():
    class TestDictConfig(Config):
        a: Dict[str, int] = {}

    assert isinstance(TestDictConfig().a, frozendict)
    assert isinstance(TestDictConfig(a={'b': 1}).a, frozendict)


def test_init_subconfigs():
    class ExampleDefaultInnerConfig(Config):
        a: int = 0

    class ExampleNonDefaultInnerConfig(Config):
        a: int

    class ExampleConfigWithSubConfigs(Config):
        default: ExampleDefaultInnerConfig
        non_default: ExampleNonDefaultInnerConfig

    assert ExampleConfigWithSubConfigs(non_default={"a": 1}).as_dict() == {
        "default": {"a": 0},
        "non_default": {"a": 1},
    }

    with pytest.raises(Exception):  # noqa: B017
        ExampleConfigWithSubConfigs(default={"a": 1})


def test_invalid_attributes_in_post_init():
    class TestInvalidConfig(Config):
        a: int

        def __post_init__(self):
            super().__post_init__()
            self.b = self.a + 1

    with pytest.raises(UnexpectedConfigAttributeError):
        TestInvalidConfig(a=1)


def test_inheritance_from_frozen_dataclass():
    @dataclass(frozen=True)
    class DataClass:
        a: str
        b: str = 'B'

    class ConfigClass(Config, DataClass):
        pass

    config = ConfigClass(a='A')
    assert config.a == 'A'
    assert config.b == 'B'


def test_overloading_post_init_raises_error_if_super_not_called():
    with pytest.raises(RuntimeError):

        class ConfigClass(Config):
            a: str

            def __post_init__(self):
                pass


def test_config_attribute_errors():
    with pytest.raises(MissingConfigAttributeError) as first_error_info:
        ExampleConfig(c=1)

    with pytest.raises(MissingConfigAttributeError) as second_error_info:
        ExampleConfig(a=2)

    assert (
        str(first_error_info.value)
        == "Fields ['a'] of ExampleConfig were not provided\n\nExampleConfig\n\ta: MissingConfigAttributeError"
    )
    assert (
        str(second_error_info.value)
        == "Fields ['c'] of ExampleConfig were not provided\n\nExampleConfig\n\tc: MissingConfigAttributeError"
    )


def test_missing_attr_error_if_attr_is_missing():
    """
    If there is a non-trivial postinit code, the config construction might have trigger a cryptic error before
    because of a missing argument in yaml.
    """

    class ExampleConfigWithPostinit(Config):
        a: int
        b: int = 0
        c: int

        def __post_init__(self):
            super().__post_init__()
            assert self.a != self.b

    with pytest.raises(MissingConfigAttributeError) as excinfo:
        ExampleConfigWithPostinit()

    assert "Fields ['a', 'c'] of ExampleConfigWithPostinit were not provided" in str(excinfo.value)


def test_config_subclass_type_check():
    class BaseConfig(Config):
        pass

    class InheritedConfig(BaseConfig):
        pass

    # Exact type matching is enforced: a plain Config-typed field rejects a subclass instance.
    class ExampleConfigSubclass(Config):
        a: BaseConfig

    with pytest.raises(InvalidConfigAttributeTypeError):
        ExampleConfigSubclass(a=InheritedConfig())

    # AutoConfig is the supported way to allow subclass polymorphism.
    class ExampleAutoConfigSubclass(Config):
        a: AutoConfig[BaseConfig]

    ExampleAutoConfigSubclass(a=InheritedConfig())

    # AutoConfig still enforces its bound: an unrelated Config is rejected.
    class UnrelatedConfig(Config):
        pass

    with pytest.raises(InvalidConfigAttributeTypeError):
        ExampleAutoConfigSubclass(a=UnrelatedConfig())


def test_config_autocasts_basic_numeric_types():
    class NumericConfig(Config):
        i: int
        f: float

    # An int provided for a float field is cast to float; the int field stays int.
    config = NumericConfig(i=5, f=3)
    assert type(config.i) is int and config.i == 5
    assert type(config.f) is float and config.f == 3.0

    # A float with no fractional part provided for an int field is cast to int.
    config = NumericConfig(i=5.0, f=2.5)
    assert type(config.i) is int and config.i == 5
    assert type(config.f) is float and config.f == 2.5

    # A float with a fractional part for an int field is not cast and fails the type check.
    with pytest.raises(InvalidConfigAttributeTypeError):
        NumericConfig(i=5.5, f=1.0)

    # bool is never cast to a numeric field, even though it subclasses int.
    with pytest.raises(InvalidConfigAttributeTypeError):
        NumericConfig(i=True, f=1.0)


def test_config_autocasts_list_to_tuple():
    class TupleConfig(Config):
        pair: Tuple[float, float]
        variadic: Tuple[str, ...] = ()
        maybe_pair: Optional[Tuple[int, int]] = None

    # A list provided for a Tuple field is cast to a tuple (YAML has no tuple literal).
    config = TupleConfig(pair=[0.01, 1.0], variadic=['a', 'b'])
    assert type(config.pair) is tuple and config.pair == (0.01, 1.0)
    assert type(config.variadic) is tuple and config.variadic == ('a', 'b')

    # An empty list for a variadic Tuple becomes an empty tuple.
    assert TupleConfig(pair=[1.0, 2.0], variadic=[]).variadic == ()

    # Tuple elements are recursively coerced (int -> float here).
    config = TupleConfig(pair=[1, 2])
    assert config.pair == (1.0, 2.0) and all(type(v) is float for v in config.pair)

    # A wrong-length list for a fixed-size Tuple is not silently accepted.
    with pytest.raises(InvalidConfigAttributeTypeError):
        TupleConfig(pair=[1.0])

    # Optional[Tuple]: a list is cast to a tuple, while None stays None.
    config = TupleConfig(pair=[1.0, 2.0], maybe_pair=[3, 4])
    assert type(config.maybe_pair) is tuple and config.maybe_pair == (3, 4)
    assert TupleConfig(pair=[1.0, 2.0], maybe_pair=None).maybe_pair is None


def test_config_scalar_type_mismatch():
    class ScalarConfig(Config):
        a: int
        b: str

    # Wrong scalar types are rejected by the now-enabled type checks.
    with pytest.raises(InvalidConfigAttributeTypeError):
        ScalarConfig(a="not an int", b="ok")

    with pytest.raises(InvalidConfigAttributeTypeError):
        ScalarConfig(a=1, b=2)

    # None for a non-Optional field is rejected.
    with pytest.raises(InvalidConfigAttributeTypeError):
        ScalarConfig(a=None, b="ok")


def test_enum_attributes():
    class MyEnum(StrEnum):  # pylint: disable=invalid-enum-extension
        VALUE_1 = "value_1"
        VALUE_2 = "value_2"
        VALUE_3 = "value_3"
        NONE = 'none'

    class EnumConfig(Config):
        a: int
        b: MyEnum

    config = EnumConfig(a=1, b=MyEnum.VALUE_1)

    assert isinstance(config.b, MyEnum), type(config.b)

    config = EnumConfig(a=1, b='value_1')
    assert isinstance(config.b, MyEnum), type(config.b)

    assert config.as_json() == {'a': 1, 'b': 'value_1'}


def test_union_of_enum_attributes():
    class MyEnum(StrEnum):  # pylint: disable=invalid-enum-extension
        VALUE_1 = "value_1"
        VALUE_2 = "value_2"
        VALUE_3 = "value_3"
        NONE = 'none'

    class EnumConfig(Config):
        a: int
        b: MyEnum | float

    config = EnumConfig(a=1, b=MyEnum.VALUE_1)

    assert isinstance(config.b, MyEnum), type(config.b)

    config = EnumConfig(a=1, b='value_1')
    assert isinstance(config.b, MyEnum), type(config.b)

    assert config.as_json() == {'a': 1, 'b': 'value_1'}

    assert config == EnumConfig.from_json(config.as_json())

    config = EnumConfig(a=1, b=3.0)
    assert isinstance(config.b, float), type(config.b)

    class AnotherEnum(StrEnum):  # pylint: disable=invalid-enum-extension
        VALUE_1 = "value_1"
        NONE = 'none'

    with pytest.raises(TypeError):

        class InvalidEnumConfig(Config):
            a: int
            b: MyEnum | AnotherEnum


def test_optional_config_attribute():
    class ConfigWithOptional(Config):
        a: int
        b: Optional[ExampleConfig] = None

    config = ConfigWithOptional(a=1, b=None)

    assert config.b is None, config.b

    config = ConfigWithOptional(a=1, b=ExampleConfig(a=2, c=3))
    assert isinstance(config.b, ExampleConfig), type(config.b)

    config = ConfigWithOptional(a=1, b={'a': 2, 'c': 3})
    assert isinstance(config.b, ExampleConfig), type(config.b)


def test_forbidden_type_annotations():
    with pytest.raises(TypeError):

        class UnionOfConfigs(Config):
            b: ExampleConfig | ExampleNestedConfig

    with pytest.raises(TypeError):

        class UnionOfConfigAndInt(Config):
            b: ExampleConfig | int

    with pytest.raises(TypeError):

        class TupleOfConfigAndEllipsis(Config):
            b: Tuple[ExampleConfig, ...]

    with pytest.raises(TypeError):

        class TupleOfConfigAndInt(Config):
            b: Tuple[ExampleConfig, int]

    with pytest.raises(TypeError):

        class TupleOfConfigs(Config):
            b: Tuple[ExampleConfig, ExampleNestedConfig]

    class MyEnum(StrEnum):  # pylint: disable=invalid-enum-extension
        VALUE_1 = "value_1"

    with pytest.raises(TypeError):

        class TupleOfEnumAndEllipsis(Config):
            b: Tuple[MyEnum, ...]

    TypeVarT = TypeVar('TypeVarT')
    with pytest.raises(TypeError):

        class ConfigWithTypeVar(Config):
            b: TypeVarT


def test_default_subconfigs_are_respected():
    class NestedConfig(Config):
        inner: ExampleConfig = ExampleConfig(a=1, c=3)
        inner_list: List[int]

    class DoubleNestedConfig(Config):
        outer: NestedConfig = NestedConfig(inner=ExampleConfig(a=5, b=6, c=8), inner_list=[1, 2, 3])
        root_value: int = 10

    nested_config = NestedConfig(inner_list=[])
    assert nested_config.inner.as_dict() == {'a': 1, 'b': 0, 'c': 3}

    nested_config = NestedConfig(inner_list=[], inner={'c': 10})
    assert nested_config.inner.as_dict() == {'a': 1, 'b': 0, 'c': 10}

    double_nested_config = DoubleNestedConfig(root_value=1)
    assert double_nested_config.outer.as_dict() == {
        'inner': {'a': 5, 'b': 6, 'c': 8},
        'inner_list': [1, 2, 3],
    }

    double_nested_config = DoubleNestedConfig(
        root_value=1, outer={'inner': {'a': 7, 'b': 12}, 'inner_list': [4, 5]}
    )
    assert double_nested_config.outer.as_dict() == {'inner': {'a': 7, 'b': 12, 'c': 8}, 'inner_list': [4, 5]}


class ConfigA(Config):
    base_field: int = 10


class ConfigB(ConfigA):
    common_field: int


class ConfigC(ConfigA):
    common_field: int
    only_c: int


class ConfigurableB(Configurable[ConfigB]):
    pass


class ConfigurableC(Configurable[ConfigC]):
    pass


auto_path_b = "ladder.core.databib.config.test.test_config.ConfigurableB"
auto_path_c = "ladder.core.databib.config.test.test_config.ConfigurableC"
auto_path_nested = "ladder.core.databib.config.test.test_config.MyNestedConfigurable"
auto_path_nested_more = "ladder.core.databib.config.test.test_config.MyNestedConfigurableWithMore"


class MyNestedConfig(Config):
    inner_1: AutoConfig = ConfigB(base_field=11, common_field=12)  # type: ignore
    inner_2: AutoConfig = AutoConfig(autoclass=auto_path_b, base_field=13, common_field=14)  # type: ignore
    field: float = 0.5


class MyNestedConfigWithMore(Config):
    inner_1: AutoConfig = ConfigB(base_field=11, common_field=12)  # type: ignore
    inner_2: AutoConfig = AutoConfig(autoclass=auto_path_b, base_field=13, common_field=14)  # type: ignore
    field: float = 0.5
    extra_field: int = 10


class MyNestedConfigurable(Configurable[MyNestedConfig]):
    pass


class MyNestedConfigurableWithMore(Configurable[MyNestedConfigWithMore]):
    pass


class MyDoubleNestedConfig(Config):
    outer_1: AutoConfig = MyNestedConfig(  # type: ignore
        inner_1=AutoConfig(autoclass=auto_path_b, base_field=15, common_field=16),
        inner_2=AutoConfig(autoclass=auto_path_b, base_field=17, common_field=18),
        field=1.75,
    )
    outer_2: AutoConfig = AutoConfig(  # type: ignore
        autoclass=auto_path_nested,
        inner_1=AutoConfig(autoclass=auto_path_b, base_field=25, common_field=26),
        inner_2=AutoConfig(autoclass=auto_path_b, base_field=27, common_field=28),
        field=0.75,
    )
    root_value: int = 10


def test_default_subconfigs_with_autoconfig():
    # TEST 1: Make sure default value is correctly initialized
    nested_config = MyNestedConfig()
    expected = {
        'inner_1': {'base_field': 11, 'common_field': 12},
        'inner_2': {'autoclass': auto_path_b, 'base_field': 13, 'common_field': 14},
        'field': 0.5,
    }
    assert nested_config.as_json() == expected

    # TEST 2: Make sure we can correctly overwrite inner_1 and inner_2 via `autoclass`
    nested_config = MyNestedConfig(
        inner_1={'autoclass': auto_path_c, 'common_field': 21, 'only_c': 22},
        inner_2={'autoclass': auto_path_c, 'common_field': 23, 'only_c': 24},
        field=1.5,
    )
    expected = {
        'inner_1': {'autoclass': auto_path_c, 'base_field': 11, 'common_field': 21, 'only_c': 22},
        'inner_2': {'autoclass': auto_path_c, 'base_field': 13, 'common_field': 23, 'only_c': 24},
        'field': 1.5,
    }
    assert type(nested_config.inner_1) is ConfigC
    assert type(nested_config.inner_2) is ConfigC
    assert nested_config.as_dict() == expected

    # TEST 3: Make sure we can overwrite inner_1 and inner_2 via specific Config instance instead of autoclass dict
    nested_config = MyNestedConfig(
        inner_1=ConfigC(common_field=21, only_c=22),
        inner_2=ConfigC(common_field=23, only_c=24),
        field=1.5,
    )
    expected = {
        'inner_1': {'base_field': 10, 'common_field': 21, 'only_c': 22},
        'inner_2': {'base_field': 10, 'common_field': 23, 'only_c': 24},
        'field': 1.5,
    }
    assert type(nested_config.inner_1) is ConfigC
    assert nested_config.as_dict() == expected

    # TEST 4: Make sure default value is correctly initialized
    double_nested_config = MyDoubleNestedConfig()
    expected = {
        'outer_1': {
            'field': 1.75,
            'inner_1': {'autoclass': auto_path_b, 'base_field': 15, 'common_field': 16},
            'inner_2': {'autoclass': auto_path_b, 'base_field': 17, 'common_field': 18},
        },
        'outer_2': {
            'autoclass': auto_path_nested,
            'field': 0.75,
            'inner_1': {'autoclass': auto_path_b, 'base_field': 25, 'common_field': 26},
            'inner_2': {'autoclass': auto_path_b, 'base_field': 27, 'common_field': 28},
        },
        'root_value': 10,
    }
    assert type(double_nested_config.outer_1) is MyNestedConfig
    assert type(double_nested_config.outer_2) is MyNestedConfig
    assert type(double_nested_config.outer_1.inner_1) is ConfigB
    assert type(double_nested_config.outer_1.inner_2) is ConfigB
    assert type(double_nested_config.outer_2.inner_1) is ConfigB
    assert type(double_nested_config.outer_2.inner_2) is ConfigB
    assert double_nested_config.as_dict() == expected

    # TEST 5: Make sure we can correctly overwrite inner_1 and inner_2 via `autoclass`
    double_nested_config = MyDoubleNestedConfig(
        outer_1={
            'autoclass': auto_path_nested_more,
            'field': 3.0,
            'inner_1': {'autoclass': auto_path_c, 'base_field': 31, 'common_field': 32, 'only_c': 33},
            'inner_2': {'autoclass': auto_path_c, 'base_field': 35, 'common_field': 36, 'only_c': 37},
        },
        outer_2={
            'autoclass': auto_path_nested_more,
            'field': 4.0,
            'extra_field': 40,
            'inner_1': {'autoclass': auto_path_b, 'base_field': 41, 'common_field': 42},
            'inner_2': {'autoclass': auto_path_b, 'base_field': 45, 'common_field': 46},
        },
    )
    expected = {
        'outer_1': {
            'autoclass': auto_path_nested_more,
            'field': 3.0,
            'extra_field': 10,
            'inner_1': {'autoclass': auto_path_c, 'base_field': 31, 'common_field': 32, 'only_c': 33},
            'inner_2': {'autoclass': auto_path_c, 'base_field': 35, 'common_field': 36, 'only_c': 37},
        },
        'outer_2': {
            'autoclass': auto_path_nested_more,
            'field': 4.0,
            'extra_field': 40,
            'inner_1': {'autoclass': auto_path_b, 'base_field': 41, 'common_field': 42},
            'inner_2': {'autoclass': auto_path_b, 'base_field': 45, 'common_field': 46},
        },
        'root_value': 10,
    }
    assert type(double_nested_config.outer_1) is MyNestedConfigWithMore
    assert type(double_nested_config.outer_2) is MyNestedConfigWithMore
    assert type(double_nested_config.outer_1.inner_1) is ConfigC
    assert type(double_nested_config.outer_1.inner_2) is ConfigC
    assert type(double_nested_config.outer_2.inner_1) is ConfigB
    assert type(double_nested_config.outer_2.inner_2) is ConfigB
    assert double_nested_config.as_dict() == expected

    # TEST 6: Make sure we can overwrite inner_1 and inner_2 via specific Config instance instead of autoclass dict
    double_nested_config = MyDoubleNestedConfig(
        outer_1=MyNestedConfig(inner_1=ConfigC(base_field=21, common_field=22, only_c=23), field=1.0)
    )
    double_nested_config = MyDoubleNestedConfig(
        outer_1={
            'field': 3.0,
            'inner_1': ConfigC(base_field=31, common_field=32, only_c=33),
            'inner_2': ConfigC(base_field=35, common_field=36, only_c=37),
        },
        outer_2=MyNestedConfigWithMore(
            field=4.0,
            extra_field=40,
            inner_1=ConfigB(base_field=41, common_field=42),
            inner_2=ConfigB(base_field=45, common_field=46),
        ),
    )
    expected = {
        'outer_1': {
            'field': 3.0,
            'inner_1': {'base_field': 31, 'common_field': 32, 'only_c': 33},
            'inner_2': {'base_field': 35, 'common_field': 36, 'only_c': 37},
        },
        'outer_2': {
            'field': 4.0,
            'extra_field': 40,
            'inner_1': {'base_field': 41, 'common_field': 42},
            'inner_2': {'base_field': 45, 'common_field': 46},
        },
        'root_value': 10,
    }
    assert type(double_nested_config.outer_1) is MyNestedConfig
    assert type(double_nested_config.outer_2) is MyNestedConfigWithMore
    assert type(double_nested_config.outer_1.inner_1) is ConfigC
    assert type(double_nested_config.outer_1.inner_2) is ConfigC
    assert type(double_nested_config.outer_2.inner_1) is ConfigB
    assert type(double_nested_config.outer_2.inner_2) is ConfigB
    assert double_nested_config.as_dict() == expected
