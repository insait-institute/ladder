import os
import pathlib
import re
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union, cast

import yaml

# Type alias representing a config "dependency" tree (to trace which configs are imported)
ConfigImportTree = Dict[str, Any]


class _OverwriteDict(dict):
    """A custom dict class that marks nodes that overwrite existing nodes."""

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


class _OverwriteList(list):
    """A custom dict class that marks nodes that overwrite existing nodes."""

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


def load_config(config_path: Union[str, TextIO, pathlib.Path]) -> Dict[str, Any]:
    config, _ = _load_raw_yaml_with_import_tree(config_path)
    config = _expand_env_vars_recursive(config)  # type: ignore[assignment]
    config = _strip_overwrite_types(config)  # type: ignore[assignment]
    return config


def _strip_overwrite_types(config: Dict[str, Any] | List[Any] | Any) -> Dict[str, Any] | List[Any] | Any:
    if isinstance(config, dict):
        return {key: _strip_overwrite_types(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_strip_overwrite_types(item) for item in config]
    return config


def _expand_env_vars_recursive(node: Dict[str, Any] | List[Any] | Any) -> Dict[str, Any] | List[Any] | Any:
    if isinstance(node, dict):
        return type(node)({key: _expand_env_vars_recursive(value) for key, value in node.items()})
    if isinstance(node, list):
        return [_expand_env_vars_recursive(item) for item in node]
    if isinstance(node, str):
        return os.path.expandvars(node)
    return node


def _merge_or_overwrite_value(
    base_value: Dict[str, Any] | List[Any], merge_value: Dict[str, Any] | List[Any]
) -> Dict[str, Any] | List[Any]:
    """
    Merge `base_value` and `merge_value` respecing the !overwrite tag, encoded as `_OverwriteDict` or
    `_OverwriteList`. If merge_value is of type `_OverwriteDict` or `_OverwriteList`, completely overwrites
    `base_value`. Otherwise merges `base_value` and `merge_value`
    """
    if isinstance(merge_value, (_OverwriteDict, _OverwriteList)):
        return merge_value

    # fmt: off
    if (
        isinstance(base_value, dict) != isinstance(merge_value, dict) or
        isinstance(base_value, list) != isinstance(merge_value, list)
    ):
        raise TypeError(f"Can't merge entry {base_value} with entry {merge_value}")
    # fmt: on

    if isinstance(base_value, dict):
        result = type(base_value)({key: value for key, value in base_value.items()})

        for key, value in merge_value.items():  # type: ignore[union-attr]
            # Overwrite local keys pointing to dict
            if isinstance(value, _OverwriteDict):
                result[key] = dict(value)  # Convert _OverwriteDict to regular dict
            # Overwrite local keys pointing to list
            elif isinstance(value, _OverwriteList):
                result[key] = list(value)  # Convert _OverwriteList to regular list
            # Merge nested dicts
            elif isinstance(value, dict) and key in result:
                if not isinstance(result[key], dict):
                    raise ValueError(
                        f"Cannot merge values for key {key} of types dict and {type(result[key])}"
                    )
                result[key] = _merge_or_overwrite_value(result[key], value)
            elif isinstance(value, list) and key in result:
                if not isinstance(result[key], list):
                    raise ValueError(
                        f"Cannot merge values for key {key} of types list and {type(result[key])}"
                    )
                # NOTE: We merge lists by direct appends
                result[key] = [*result[key], *value]
            else:
                result[key] = value
    elif isinstance(base_value, list):
        result = type(base_value)([*base_value, *merge_value])  # type: ignore[assignment]
    else:
        result = merge_value

    return result


def _load_raw_yaml_with_import_tree(
    config_path: Union[str, TextIO, pathlib.Path]
) -> Tuple[Dict[str, Any], ConfigImportTree]:
    if isinstance(config_path, pathlib.Path):
        config_path = str(config_path)

    loader = _Loader(config_path)
    try:
        # Note: get_single_data() returns None for an empty config, we want an empty dict
        config = loader.get_single_data() or {}
        import_tree = loader.import_tree
        return config, import_tree
    finally:
        loader.dispose()


def _ordered_dict_representer(dumper, data):
    return yaml.nodes.MappingNode(
        yaml.SafeDumper.DEFAULT_MAPPING_TAG,
        [(dumper.represent_data(k), dumper.represent_data(v)) for k, v in data.items()],
    )


class _Loader(yaml.SafeLoader):
    config_path: Optional[str]

    def __init__(self, config_path_or_file: Union[str, TextIO]):
        if type(config_path_or_file) is str:
            self.config_path = cast(str, config_path_or_file)
            config_file = open(self.config_path, 'r')
        else:
            self.config_path = None
            config_file = cast(TextIO, config_path_or_file)  # type: ignore

        super().__init__(config_file)

        self.add_implicit_resolver(
            'tag:yaml.org,2002:float',
            re.compile(
                '''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''',
                re.X,
            ),
            list('-+0123456789.'),
        )

        self.add_constructor(self.DEFAULT_MAPPING_TAG, self.mapping_constructor)
        self.add_constructor('!overwrite', self.overwrite_constructor)

        # Keep track of the !import dependency tree. Note that we only support importing when reading
        # from a path on disk. !import doesn't work with file descripors or buffers
        self._dep_tree: ConfigImportTree = {}

    def overwrite_constructor(self, loader, node):
        if isinstance(node, yaml.MappingNode):
            return _OverwriteDict(loader.construct_mapping(node))
        if isinstance(node, yaml.SequenceNode):
            return _OverwriteList(loader.construct_sequence(node))
        raise ValueError("Can only use `!overwrite` tag for dicts or lists")

    @property
    def import_tree(self) -> ConfigImportTree:
        return self._dep_tree

    @staticmethod
    def mapping_constructor(loader, node):
        # Everything here must go through ``loader``, never through the instance that registered this
        # constructor: ``add_constructor`` registers on the Loader CLASS, so that instance is whichever
        # loader was built LAST -- for a nested import, the imported file rather than the one being
        # parsed. Resolving against it made a relative import land in the wrong directory whenever a
        # file's first import pulled in a config from somewhere else, so a path that looked correct
        # from its own location failed, and a wrong one could work by accident.
        if not isinstance(node, yaml.nodes.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, "expected a mapping node, but found %s" % node.id, node.start_mark
            )

        result = {}

        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            value = loader.construct_object(value_node, deep=True)

            if key == "import":
                if loader.config_path is None:
                    raise ValueError('Can only use `!import` tag when loading from a path on disk')
                import_path = os.path.join(os.path.dirname(loader.config_path), os.path.expandvars(value))
                if not os.path.isfile(import_path):
                    raise FileNotFoundError(f'Cannot load referenced yaml at: {import_path}')

                imported_config, imported_dep_tree = _load_raw_yaml_with_import_tree(import_path)
                loader.merge_import_tree(import_path, imported_dep_tree)
                # Merge the imported config immediately. Order of import and !overwrite matters:
                #   - `import` after `!overwrite` merges into `!overwrite`
                #   - `!overwrite` after `import` overwrites whatever is inside the `import`
                result = _merge_or_overwrite_value(result, imported_config)
            else:
                if key not in result:
                    result[key] = value
                else:
                    result[key] = _merge_or_overwrite_value(result[key], value)

        return result

    def merge_import_tree(self, import_path: str, imported_dep_tree: ConfigImportTree):
        # Merge the imported dep tree with the existing one if that config was already imported.
        self._dep_tree[import_path] = {
            **self._dep_tree.get(import_path, {}),
            **imported_dep_tree,
        }

    def dispose(self):
        self.stream.close()
        super().dispose()


def update_config_dict(dict_1: Dict[str, Any], dict_2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates dict_1 with dict_2. Works for nested dictionaries.
    If any value in dict_1 is not a dict, it cannot be replaced with a dict and vice-versa.
    Example 1:
        dict_1: {'a': 5, 'b': {'c': 6, 'd': 7}}
        dict_2: {'b': {'e': 3}}
        result: {'a': 5, 'b': {'c': 6, 'd': 7, 'e': 3}}
    Example 2:
        dict_1: {'a': 5, 'b': {'c': 6, 'd': 7}}
        dict_2: {'b': {'d': {'e': 3}}}
        result: Error: int cannot be replaced by a dict
    """
    dict_1 = dict(dict_1)
    for key, value in dict_2.items():
        if isinstance(value, dict):
            source_value = dict_1.get(key, {})
            if not isinstance(source_value, dict):
                raise TypeError(
                    f'Updating {source_value} of type {type(source_value)} with a dict {value} is not allowed.'
                )
            dict_1[key] = update_config_dict(source_value, value)
        else:
            if isinstance(dict_1.get(key, None), dict):
                raise TypeError(
                    f'Updating {dict_1.get(key)} of type dict with {value} of type {type(value)} is not allowed.'
                )
            dict_1[key] = value
    return dict_1
