import argparse
import ast
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import yaml

from .config import Config
from .yaml_util import load_config, update_config_dict


class ConfigParser(argparse.ArgumentParser):
    def __init__(self, config_class: Type[Config], config_path_key: Optional[str] = None):
        super().__init__()
        self.config_class = config_class
        self.config_path_key = config_path_key
        self.add_argument("--config", type=str, default=None)

    def _process_known_args(self, known_args: Dict[str, Any]) -> Optional[str]:
        config_path = known_args.pop('config', None)
        return config_path

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        config = load_config(config_path)

        if self.config_path_key is not None:
            config = assign_dict_value_by_recursive_access(config, self.config_path_key, config_path)

        return config

    @staticmethod
    def _set_recursive_arg_value(args_dict: Dict[str, Any], arg_path: str, arg_value: Any) -> Dict[str, Any]:
        """
        Update a nested dictionary with a key and value in place
        arg_path: Can be a nested path separated with `.`
        """
        # Cast value to the correct type
        try:
            value: Any = ast.literal_eval(arg_value)
        except (ValueError, SyntaxError):
            value = arg_value

        name_path: List[str] = arg_path.split('.')
        parents_path, base_name = name_path[:-1], name_path[-1]

        base_dict = args_dict
        for key in parents_path:
            if key not in base_dict:
                base_dict[key] = {}
            base_dict = base_dict[key]
        base_dict[base_name] = value

        return args_dict

    @staticmethod
    def _extra_args_name_value_iter(extra_args: List[str]) -> Iterator[Tuple[str, Any]]:
        names = set()
        i = 0
        while i < len(extra_args):
            arg = extra_args[i]
            if not arg.startswith('--'):
                raise ValueError(
                    f"Expected argument starting with --, but got only {arg}. "
                    f"Argument specification should follow the syntax '--name value' or '--name=value'."
                )

            # Arg provided with syntax '--arg_name=value'
            if '=' in arg:
                data: List[str] = arg[2:].split('=')
                if len(data) != 2:
                    raise ValueError(f"Argument name or value cannot contain '=', but got {arg}")

                name, value = data[0], data[1]
                i += 1

            # Arg provided with syntax '--arg_name value'
            else:
                if i == len(extra_args) - 1 or extra_args[i + 1].startswith('--'):
                    raise ValueError(
                        f"No value provided for argument {arg[2:]}. Expected {arg} <value> or {arg}=<value>."
                    )

                name, value = arg[2:], extra_args[i + 1]
                i += 2

            if name in names:
                raise ValueError(f"Duplicate arguments provided for {name}")

            names.add(name)

            yield name, yaml.safe_load(value)

    def _parse_cmdline_config_args(self, extra_args: List[str]) -> Dict[str, Any]:
        cmdline_args: Dict[str, Any] = {}

        for name, value in self._extra_args_name_value_iter(extra_args):
            cmdline_args = self._set_recursive_arg_value(cmdline_args, name, value)

        return cmdline_args

    def _parse_args_to_dict(self, args=None, namespace=None) -> Dict[str, Any]:
        parsed_args: Tuple[argparse.Namespace, List[str]] = super().parse_known_args(args, namespace)
        known_args: dict = vars(parsed_args[0])
        extra_args: List[str] = parsed_args[1]

        config_path: Optional[str] = self._process_known_args(known_args)
        if config_path is not None:
            yml_args: Dict[str, Any] = self._load_config(config_path)
        else:
            yml_args = {}
        cmdline_args: Dict[str, Any] = self._parse_cmdline_config_args(extra_args)
        config_args: Dict[str, Any] = update_config_dict(yml_args, cmdline_args)

        return config_args

    def _parse_config(self, config_args: Dict[str, Any], config: Optional[Config] = None) -> Config:
        if config is None:
            return self.config_class(**config_args)
        return config.replace(**config_args)

    def _parse_args_to_config(self, args=None, namespace=None, config=None) -> Config:
        config_args: Dict[str, Any] = self._parse_args_to_dict(args=args, namespace=namespace)
        return self._parse_config(config_args, config)

    def parse_args(self, args=None, namespace=None, config=None):
        config = self._parse_args_to_config(args=args, namespace=namespace, config=config)
        return config

    def parse_args_to_dict(self, args=None, namespace=None) -> Dict[str, Any]:
        """Parse config to a dict instead of a Config object"""
        return self._parse_args_to_dict(args=args, namespace=namespace)

    def parse_from_path(self, config_path: str) -> Config:
        config_dict = self._load_config(config_path)
        return self._parse_config(config_dict)


def assign_dict_value_by_recursive_access(data: Dict[str, Any], full_key: str, value: Any) -> Dict[str, Any]:
    keys: List[str] = full_key.split('.')

    inner_dict = data
    for key in keys[:-1]:
        if key not in inner_dict:
            inner_dict[key] = {}
        inner_dict = inner_dict[key]

    inner_dict[keys[-1]] = value

    return data
