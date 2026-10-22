import unittest
from tempfile import NamedTemporaryFile
from typing import Dict, List

import yaml
from frozendict import frozendict

from ladder.core.databib.config.config import Config, UnexpectedConfigAttributeError
from ladder.core.databib.config.config_parser import ConfigParser


class FakeNestedConfig(Config):
    fake_int_field: int = 0
    fake_float_field: float = 0.05
    fake_str_field: str = "default"
    fake_dict_field: Dict[str, int] = {}
    fake_list_field: List[int] = []

    default_int_field: int = 10


class FakeConfig(Config):
    nested: FakeNestedConfig
    fake_int_field: int = 5
    default_int_field: int = 20


class TestConfigParser(unittest.TestCase):
    def setUp(self):
        self.config_file = NamedTemporaryFile()  # Autodeletes
        self.config_path = self.config_file.name

    def test_extra_args_are_correctly_parsed(self):
        config = ConfigParser(FakeConfig).parse_args(
            [
                "--config",
                self.config_path,
                "--fake_int_field",
                "2",
                "--nested.fake_int_field",
                "1",
                "--nested.fake_float_field",
                "0.00008",
                "--nested.fake_str_field",
                "non-default",
                "--nested.fake_dict_field",
                "{'fake_key_1': 1, 'fake_key_2': 2}",
                "--nested.fake_list_field",
                "[1, 2, 3, 4]",
            ]
        )

        # Make sure arguments are parsed to their correct types
        self.assertIsInstance(config.fake_int_field, int)
        self.assertIsInstance(config.nested.fake_int_field, int)
        self.assertIsInstance(config.nested.fake_float_field, float)
        self.assertIsInstance(config.nested.fake_str_field, str)
        self.assertIsInstance(config.nested.fake_dict_field, frozendict)
        self.assertIsInstance(config.nested.fake_list_field, tuple)

        # Make sure arguments are parsed to their correct values
        self.assertEqual(config.fake_int_field, 2)
        self.assertEqual(config.nested.fake_int_field, 1)
        self.assertEqual(config.nested.fake_float_field, 0.00008)
        self.assertEqual(config.nested.fake_str_field, "non-default")
        self.assertEqual(config.nested.fake_dict_field, {"fake_key_1": 1, "fake_key_2": 2})
        self.assertEqual(config.nested.fake_list_field, (1, 2, 3, 4))

        # Make sure other values are not affected
        self.assertEqual(config.default_int_field, 20)
        self.assertEqual(config.nested.default_int_field, 10)

    def test_raises_on_incorrect_syntax(self):
        args = ["--config", self.config_path]

        # Make sure an error is raised when incorrect argument syntax
        with self.assertRaises(ValueError):
            ConfigParser(FakeConfig).parse_args(args + ['--fake_int_field'])

        with self.assertRaises(ValueError):
            ConfigParser(FakeConfig).parse_args(args + ['--fake_int_field', '--default_int_field', '0'])

        with self.assertRaises(ValueError):
            ConfigParser(FakeConfig).parse_args(args + ['--fake_int_field', '--default_int_field=0'])

        with self.assertRaises(ValueError):
            ConfigParser(FakeConfig).parse_args(args + ['--fake_int_field=5=6', '--default_int_field=0'])

        # Make sure an error is raised when non-existing arguments are provided
        with self.assertRaises(ValueError):
            ConfigParser(FakeConfig).parse_args(args + ['--fake_float_field'])

        with self.assertRaises(UnexpectedConfigAttributeError):
            ConfigParser(FakeConfig).parse_args(args + ['--nested.fake_non_existent_field', '0'])

        # Make sure error is raises when the argument is provided twice
        with self.assertRaises(ValueError):
            ConfigParser(FakeConfig).parse_args(args + ['--fake_float_field', '0', '--fake_float_field', '0'])

    def test_command_line_overrides_yaml_config(self):
        yaml_config = {
            "fake_int_field": 3,
            "nested": {
                "fake_int_field": 2,
                "fake_float_field": 0.00006,
                "fake_str_field": "random",
                "fake_dict_field": {"fake_key_3": 3},
                "fake_list_field": [5, 6, 7],
            },
        }

        # Dump the yaml config
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(yaml_config, f)

        config = ConfigParser(FakeConfig).parse_args(
            [
                "--config",
                self.config_path,
                "--fake_int_field",
                "2",
                "--nested.fake_int_field",
                "1",
                "--nested.fake_float_field",
                "0.00008",
                "--nested.fake_str_field",
                "non-default",
                "--nested.fake_dict_field",
                "{fake_key_1: 1, fake_key_2: 2}",
                "--nested.fake_list_field",
                "[1, 2, 3, 4]",
            ]
        )

        self.assertEqual(config.fake_int_field, 2)
        self.assertEqual(config.nested.fake_int_field, 1)
        self.assertEqual(config.nested.fake_float_field, 0.00008)
        self.assertEqual(config.nested.fake_str_field, "non-default")
        self.assertEqual(config.nested.fake_dict_field, {"fake_key_3": 3, "fake_key_1": 1, "fake_key_2": 2})
        self.assertEqual(config.nested.fake_list_field, (1, 2, 3, 4))

    def test_no_yaml_config(self):
        config = ConfigParser(FakeConfig).parse_args([])
        self.assertEqual(config.fake_int_field, 5)
        self.assertEqual(config.default_int_field, 20)
        self.assertEqual(config.nested.fake_int_field, 0)
        self.assertEqual(config.nested.fake_float_field, 0.05)
        self.assertEqual(config.nested.fake_str_field, "default")
        self.assertEqual(config.nested.fake_dict_field, {})
        self.assertEqual(config.nested.fake_list_field, ())
        self.assertEqual(config.nested.default_int_field, 10)

        config = ConfigParser(FakeConfig).parse_args(
            [
                "--config",
                self.config_path,
                "--fake_int_field",
                "2",
                "--nested.fake_int_field",
                "1",
                "--nested.fake_float_field",
                "0.00008",
                "--nested.fake_str_field",
                "non-default",
                "--nested.fake_dict_field",
                "{'fake_key_1': 1, 'fake_key_2': 2}",
                "--nested.fake_list_field",
                "[1, 2, 3, 4]",
            ]
        )
        self.assertEqual(config.fake_int_field, 2)
        self.assertEqual(config.nested.fake_int_field, 1)
        self.assertEqual(config.nested.fake_float_field, 0.00008)
        self.assertEqual(config.nested.fake_str_field, "non-default")
        self.assertEqual(config.nested.fake_dict_field, {"fake_key_1": 1, "fake_key_2": 2})
        self.assertEqual(config.nested.fake_list_field, (1, 2, 3, 4))
        self.assertEqual(config.default_int_field, 20)
        self.assertEqual(config.nested.default_int_field, 10)

    def test_config_parser_with_defaultconfig(self):
        default_config = FakeConfig(fake_int_field=10, nested=FakeNestedConfig())

        with self.subTest(name='test_config_parser_with_default_no_yaml'):
            config = ConfigParser(FakeConfig).parse_args([], config=default_config)
            self.assertEqual(config.fake_int_field, 10)
            self.assertEqual(config.default_int_field, 20)
            self.assertEqual(config.nested.fake_int_field, 0)
            self.assertEqual(config.nested.fake_float_field, 0.05)
            self.assertEqual(config.nested.fake_str_field, "default")
            self.assertEqual(config.nested.fake_dict_field, {})
            self.assertEqual(config.nested.fake_list_field, ())
            self.assertEqual(config.nested.default_int_field, 10)

        with self.subTest(name='test_config_parser_with_default_and_yaml'):
            config = ConfigParser(FakeConfig).parse_args(
                [
                    "--config",
                    self.config_path,
                    "--fake_int_field",
                    "2",
                    "--nested.fake_int_field",
                    "1",
                    "--nested.fake_float_field",
                    "0.00008",
                    "--nested.fake_str_field",
                    "non-default",
                    "--nested.fake_dict_field",
                    "{'fake_key_1': 1, 'fake_key_2': 2}",
                    "--nested.fake_list_field",
                    "[1, 2, 3, 4]",
                ]
            )
            self.assertEqual(config.fake_int_field, 2)
            self.assertEqual(config.nested.fake_int_field, 1)
            self.assertEqual(config.nested.fake_float_field, 0.00008)
            self.assertEqual(config.nested.fake_str_field, "non-default")
            self.assertEqual(config.nested.fake_dict_field, {"fake_key_1": 1, "fake_key_2": 2})
            self.assertEqual(config.nested.fake_list_field, (1, 2, 3, 4))
            self.assertEqual(config.default_int_field, 20)
            self.assertEqual(config.nested.default_int_field, 10)


if __name__ == '__main__':
    unittest.main()
