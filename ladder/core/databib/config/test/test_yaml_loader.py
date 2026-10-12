import os

from ladder.core.databib.config.yaml_util import load_config

CONFIG_FILES_DIR = os.path.join(os.path.dirname(__file__), 'config_files')


def test_canonic_yaml():
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'canonic.yaml'))
    expected = {
        'outer': {
            'inner': {'a': 1, 'b': 1, 'c': 1},
            'inner_list': [7, 8, 9],
            'a_nullable_value': None,
        }
    }
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"


def test_import_no_overwrite():
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'a.yaml'))
    expected = {
        'outer': {
            'inner': {'a': 1, 'b': 1, 'c': 1},
            'inner_list': [7, 8, 9],
            'a_nullable_value': None,
        },
        'root_value_1': 10,
        'root_value_2': 8,
    }
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"


def test_import_resolves_against_the_file_that_contains_it():
    # Loading an imported file used to hijack resolution for everything parsed afterwards, so a later
    # import in THIS file was looked up under the imported file's directory instead of its own. Here
    # that meant nested/cross_dir_sibling.yaml, which does not exist. Order of the two imports is the
    # whole point: reaching into another directory first is what used to break the sibling.
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'cross_dir_import.yaml'))
    expected = {'from_nested': 1, 'outer': {'from_sibling': 2}}
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"


def test_overwrite_import_inner():
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'overwrite_import_inner.yaml'))
    expected = {
        'outer': {
            'inner': {
                'd': 4,
            },
            'inner_list': [11, 12],
        }
    }
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"


def test_overwrite_import_outer():
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'overwrite_import_outer.yaml'))
    expected = {
        'outer': {
            'inner': {
                'e': 5,
                'f': 6,
            },
            'inner_list': [7, 8],
        }
    }
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"


def test_overwrite_import_nested():
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'overwrite_import_nested.yaml'))
    expected = {
        'outer': {
            'inner': {
                'e': 5,
                'f': 6,
            },
            'inner_list': [7, 8],
        }
    }
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"


def test_import_overwrite_chained():
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'import_overwrite_chain_base.yaml'))
    expected = {
        'untouched': {
            'key1': 'value1',
            'key2': 'value2',
        },
        'overwritten': {
            'key2': {
                'd': 4,
                'e': 5,
                'f': 6,
                'g': 7,
            }
        },
    }
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"


def test_overwrite_import_chained():
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'overwrite_import_chained_base.yaml'))
    expected = {
        'outer': {
            'inner': {
                'e': 5,
                'f': 6,
                'g': 7,
            },
            'inner_list': [7, 8],
        }
    }
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"


def test_expands_env_vars():
    os.environ['MYENVVAR'] = 'MYTESTENVVAR'
    os.environ['MYENVVAR1'] = 'MYTESTENVVAR1'
    config = load_config(os.path.join(CONFIG_FILES_DIR, 'config_with_env_var.yaml'))
    expected = {
        'key_1': 'value_1',
        'key_2': 2,
        'key_3': 'MYTESTENVVAR',
        'key_4': 'MYTESTENVVAR1',
        'nested': {
            'key_5': 'MYTESTENVVAR1',
            'nested': {
                'key_6': 'MYTESTENVVAR',
            },
        },
    }
    assert config == expected, f"Expected:\n{expected}\nActual:\n{config}"

    os.environ.pop('MYENVVAR')
    os.environ.pop('MYENVVAR1')
