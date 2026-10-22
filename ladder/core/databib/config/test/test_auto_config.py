import unittest
from typing import Any, Dict, List, TypeVar

from ladder.core.databib.config.auto_config import AutoConfig, AutoConfigurable
from ladder.core.databib.config.config import (
    AutoRecordsManager,
    Config,
    ConfigT,
    Configurable,
    MissingConfigAttributeError,
)

from .helper import ConfigJ, ConfigurableJ as ConfigurableJ1


class ConfigA(Config):
    pass


class ConfigurableA(Configurable[ConfigA]):
    pass


class ConfigurableA2(Configurable[ConfigA]):
    pass


class ConfigB(Config):
    a: int = 1
    b: float = 3.0


class ConfigurableB(Configurable[ConfigB]):
    pass


class ConfigSA(ConfigA):
    sa: int = 42


class ConfigSB(ConfigB):
    sb: int = 50


class ConfigurableSA(Configurable[ConfigSA]):
    pass


class ConfigurableSB(Configurable[ConfigSB]):
    pass


class ConfigK(Config):
    pass


class ConfigurableK1(Configurable[ConfigK]):
    pass


class ConfigurableK2(Configurable[ConfigK]):
    pass


class ConfigurableJ(Configurable[ConfigJ]):
    pass


class TestAutoConfig(unittest.TestCase):
    def test_configurable_registers_subclasses(self):
        class_path = "ladder.core.databib.config.test.test_auto_config.ConfigurableA"

        self.assertIn(class_path, AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIG_CLS)
        self.assertIn(class_path, AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIGURABLE_CLS)
        self.assertIs(AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIG_CLS[class_path], ConfigA)
        self.assertIs(AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIGURABLE_CLS[class_path], ConfigurableA)

        class ConfigC(Config):
            pass

        class ConfigurableC(Configurable[ConfigC]):
            pass

        class_path = "ladder.core.databib.config.test.test_auto_config.TestAutoConfig.test_configurable_registers_subclasses.<locals>.ConfigurableC"

        self.assertIn(class_path, AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIG_CLS)
        self.assertIn(class_path, AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIGURABLE_CLS)
        self.assertIs(AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIG_CLS[class_path], ConfigC)
        self.assertIs(AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIGURABLE_CLS[class_path], ConfigurableC)

    def test_generic_configurables_are_not_registered(self):
        class ConfigurableC(Configurable[ConfigT]):
            pass

        class_path = "ladder.core.databib.config.test.test_auto_config.TestAutoConfig.test_generic_configurables_are_not_registered.<locals>.ConfigurableC"

        self.assertNotIn(class_path, AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIG_CLS)
        self.assertNotIn(class_path, AutoRecordsManager.CONFIGURABLE_CLS_PATH_TO_CONFIGURABLE_CLS)

    def test_config_prevents_overwriting_autoclass_annotation(self):
        with self.assertRaises(ValueError):

            class ConfigC(Config):
                autoclass: str

    def test_auto_init_with_full_specs(self):
        class_path = "ladder.core.databib.config.test.test_auto_config.ConfigurableA"

        config = AutoConfig(autoclass=class_path)
        self.assertIsInstance(config, ConfigA)
        self.assertEqual(config.autoclass, class_path)

        configurable = AutoConfigurable(config)
        self.assertIsInstance(configurable, ConfigurableA)

        # Test auto can still initialize another type
        class_path = "ladder.core.databib.config.test.test_auto_config.ConfigurableB"

        config = AutoConfig(autoclass=class_path)
        self.assertIsInstance(config, ConfigB)
        self.assertEqual(config.autoclass, class_path)

        configurable = AutoConfigurable(config)
        self.assertIsInstance(configurable, ConfigurableB)

    def test_unambiguous_full_specs(self):
        class_path_1 = "ladder.core.databib.config.test.test_auto_config.ConfigurableK1"
        class_path_2 = "ladder.core.databib.config.test.test_auto_config.ConfigurableK2"

        config = AutoConfig(autoclass=class_path_1)
        self.assertIsInstance(config, ConfigK)
        self.assertEqual(config.autoclass, class_path_1)

        configurable = AutoConfigurable(config)
        self.assertIsInstance(configurable, ConfigurableK1)

        config = AutoConfig(autoclass=class_path_2)
        self.assertIsInstance(config, ConfigK)
        self.assertEqual(config.autoclass, class_path_2)

        configurable = AutoConfigurable(config)
        self.assertIsInstance(configurable, ConfigurableK2)

    def test_auto_config_raises_on_invalid_autoclass_keys(self):
        with self.assertRaises(MissingConfigAttributeError):
            AutoConfig()

        with self.assertRaises(ValueError):
            AutoConfig(autoclass="")

        with self.assertRaises(ModuleNotFoundError):
            AutoConfig(autoclass="module.does.not.exist")

    def test_auto_config_init_modes(self):
        configurable_class_path = "ladder.core.databib.config.test.test_auto_config.ConfigurableA"

        # Full Configurable path
        config = AutoConfig(autoclass=configurable_class_path)
        self.assertIsInstance(config, ConfigA)
        self.assertEqual(config.autoclass, configurable_class_path)

        # Config path instead of Configurable
        with self.assertRaises(TypeError):
            AutoConfig(autoclass="ladder.core.databib.config.test.test_auto_config.ConfigA")

        # Incomplete Configurable class path
        with self.assertRaises(ValueError):
            AutoConfig(autoclass="ConfigurableA")

    def test_auto_configurable_init_modes(self):
        # Full Configurable path
        configurable_class_path = "ladder.core.databib.config.test.test_auto_config.ConfigurableA"
        configurable = AutoConfigurable(AutoConfig(autoclass=configurable_class_path))
        self.assertIsInstance(configurable, ConfigurableA)
        self.assertEqual(configurable.config.autoclass, configurable_class_path)

    def test_auto_configurable_raises_on_ambiguous_config(self):
        config_class_path = "ladder.core.databib.config.test.test_auto_config.ConfigA"

        # Config doesn't contain autoclass
        with self.assertRaises(ValueError):
            AutoConfigurable(config=ConfigA())

        # Config class is ambiguous, need to use Configurable instead
        with self.assertRaises(ValueError):
            AutoConfigurable(config=ConfigA(autoclass="ConfigA"))

        # Config class is ambiguous, need to use Configurable instead
        with self.assertRaises(TypeError):
            AutoConfigurable(config=ConfigA(autoclass=config_class_path))

        # Ambiguous Configurable class name - also imported from helper.py
        with self.assertRaises(ValueError):
            AutoConfigurable(config=ConfigJ(autoclass="ConfigurableJ"))

        # Full path spec must work
        configurable_class_path = "ladder.core.databib.config.test.test_auto_config.ConfigurableJ"
        configurable = AutoConfigurable(AutoConfig(autoclass=configurable_class_path))
        self.assertIsInstance(configurable, ConfigurableJ)
        self.assertEqual(configurable.config.autoclass, configurable_class_path)

        configurable_class_path = "ladder.core.databib.config.test.helper.ConfigurableJ"
        configurable = AutoConfigurable(AutoConfig(autoclass=configurable_class_path))
        self.assertIsInstance(configurable, ConfigurableJ1)
        self.assertEqual(configurable.config.autoclass, configurable_class_path)

    def test_auto_config_ignore_redundant(self):
        AutoConfig.from_json(
            {
                "autoclass": "ladder.core.databib.config.test.test_auto_config.ConfigurableB",
                "c": 8,
                "d": 9,
            },
            ignore_redundant=True,
        )

    def test_config_with_autoconfig_attribute(self):
        class ConfigC(Config):
            d: int
            e: List[int]

        class ConfigurableC(Configurable[ConfigC]):
            pass

        class ConfigD(Config):
            a: int
            b: str
            c: AutoConfig

        data = {
            "a": 5,
            "b": "test",
            "c": {
                "autoclass": "ladder.core.databib.config.test.test_auto_config.TestAutoConfig.test_config_with_autoconfig_attribute.<locals>.ConfigurableC",
                "d": 6,
                "e": [7, 8, 9],
            },
        }

        config = ConfigD.from_json(data)
        self.assertIsInstance(config.c, ConfigC)

        ConfigD(a=6, b="test", c=AutoConfig(**data["c"]))
        self.assertIsInstance(config.c, ConfigC)

        data['c']['autoclass'] = ''

        with self.assertRaises(ValueError):
            ConfigD.from_json(data)

    def test_autoconfig_with_config_class_arg(self):
        # Test parametrized AutoConfig classes can be defined
        class ConfigC(Config):
            a: AutoConfig[ConfigA]
            b: AutoConfig[ConfigB]
            c: AutoConfig
            d: int

        class ConfigD(Config):
            a: int
            b: str
            c: AutoConfig[ConfigB]

        data = {
            "a": 5,
            "b": "test",
            "c": {
                "autoclass": "ladder.core.databib.config.test.test_auto_config.ConfigurableB",
                "a": 10,
                "b": 13.0,
            },
        }

        # Test parametrized AutoConfig can be instantiated
        config = ConfigD.from_json(data)
        self.assertEqual(config.c.a, 10)
        self.assertEqual(config.c.b, 13.0)

        # Test parametrized AutoConfig respects bounds
        with self.assertRaises(TypeError):
            data['c']['autoclass'] = "ladder.core.databib.config.test.test_auto_config.ConfigurableA"
            ConfigD.from_json(data)

    def test_config_dict_list_with_autoconfig_values(self):
        class MegaConfig(Config):
            unbounded_dict: Dict[str, AutoConfig]
            bounded_dict: Dict[str, AutoConfig[ConfigB]]
            unbounded_list: List[AutoConfig]
            bounded_list: List[AutoConfig[ConfigA]]

        data = {
            'unbounded_dict': {
                'c1': {'autoclass': 'ladder.core.databib.config.test.test_auto_config.ConfigurableA'},
                'c2': {'autoclass': 'ladder.core.databib.config.test.test_auto_config.ConfigurableB'},
                'c3': ConfigSA(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableSA'),
                'c4': ConfigSB(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableSB'),
            },
            'bounded_dict': {
                'c1': {'autoclass': 'ladder.core.databib.config.test.test_auto_config.ConfigurableB'},
                'c2': {'autoclass': 'ladder.core.databib.config.test.test_auto_config.ConfigurableSB'},
                'c3': ConfigB(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableB'),
                'c4': ConfigSB(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableSB'),
            },
            'unbounded_list': [
                {'autoclass': 'ladder.core.databib.config.test.test_auto_config.ConfigurableA'},
                {'autoclass': 'ladder.core.databib.config.test.test_auto_config.ConfigurableB'},
                ConfigSA(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableSA'),
                ConfigSB(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableSB'),
            ],
            'bounded_list': [
                {'autoclass': 'ladder.core.databib.config.test.test_auto_config.ConfigurableA'},
                {'autoclass': 'ladder.core.databib.config.test.test_auto_config.ConfigurableSA'},
                ConfigA(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableA'),
                ConfigSA(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableSA'),
            ],
        }

        # Make sure we can put any config in unbounded Dict/List and subclasses in bounded list
        mega_config = MegaConfig(**data)

        self.assertIsInstance(mega_config.unbounded_dict['c1'], ConfigA)
        self.assertIsInstance(mega_config.unbounded_dict['c2'], ConfigB)
        self.assertIsInstance(mega_config.unbounded_dict['c3'], ConfigSA)
        self.assertIsInstance(mega_config.unbounded_dict['c4'], ConfigSB)

        self.assertIsInstance(mega_config.bounded_dict['c1'], ConfigB)
        self.assertIsInstance(mega_config.bounded_dict['c2'], ConfigSB)
        self.assertIsInstance(mega_config.bounded_dict['c3'], ConfigB)
        self.assertIsInstance(mega_config.bounded_dict['c4'], ConfigSB)

        self.assertIsInstance(mega_config.unbounded_list[0], ConfigA)
        self.assertIsInstance(mega_config.unbounded_list[1], ConfigB)
        self.assertIsInstance(mega_config.unbounded_list[2], ConfigSA)
        self.assertIsInstance(mega_config.unbounded_list[3], ConfigSB)

        self.assertIsInstance(mega_config.bounded_list[0], ConfigA)
        self.assertIsInstance(mega_config.bounded_list[1], ConfigSA)
        self.assertIsInstance(mega_config.bounded_list[2], ConfigA)
        self.assertIsInstance(mega_config.bounded_list[3], ConfigSA)

        # Make sure AutoConfig bounds are respected in ConfigDict
        with self.assertRaises(TypeError):
            MegaConfig(
                unbounded_dict={},
                bounded_dict={
                    'a': ConfigA(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableA')
                },
                unbounded_list=[],
                bounded_list=[],
            )

        # Make sure AutoConfig bounds are respected in ConfigList
        with self.assertRaises(TypeError):
            MegaConfig(
                unbounded_dict={},
                bounded_dict={},
                unbounded_list=[],
                bounded_list=[
                    ConfigB(autoclass='ladder.core.databib.config.test.test_auto_config.ConfigurableB')
                ],
            )

    def test_auto_config_raises_on_non_type_arg(self):
        T = TypeVar('T')

        with self.assertRaises(TypeError):
            AutoConfig[Any]  # pylint: disable=pointless-statement

        with self.assertRaises(TypeError):
            AutoConfig[T]  # pylint: disable=pointless-statement

    def test_auto_config_instance_check(self):
        self.assertIsInstance(ConfigA(), AutoConfig[ConfigA])
        self.assertNotIsInstance(ConfigB(), AutoConfig[ConfigA])
        self.assertIsInstance(ConfigA(), AutoConfig)

    def test_auto_config_with_auto_import(self):
        configurable_class_path = "ladder.core.databib.config.test.import_helper.ImportConfigurable"
        AutoConfigurable(AutoConfig(autoclass=configurable_class_path))
