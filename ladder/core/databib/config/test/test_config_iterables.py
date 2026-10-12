import os
import pickle
import tempfile
from typing import Any, Dict, List, Tuple

import torch

from ladder.core.databib.config.config import Config
from ladder.core.databib.config.config_iterables import ConfigDict, ConfigList

from .helper import PickleConfig, worker_pickle_dump


class SimpleConfig(Config):
    data: float


class MyConfig(Config):
    a: int
    b: Dict[str, str]
    c: Dict[str, SimpleConfig]
    d: List[int]
    e: List[SimpleConfig]
    f: Tuple[int, float]


def test_config_init_casts_dicts_and_lists():
    my_data = MyConfig(
        a=5,
        b={"ba": "bb", "bc": "bd"},
        c={"ca": SimpleConfig(data=10.0)},
        d=[7, 8, 9],
        e=[SimpleConfig(data=11.0), SimpleConfig(data=12.0)],
        f=(5, 8.0),
    )
    assert isinstance(my_data.b, ConfigDict)
    assert isinstance(my_data.c, ConfigDict)
    assert isinstance(my_data.d, ConfigList)
    assert isinstance(my_data.e, ConfigList)

    my_data = MyConfig.from_json(my_data.as_json())
    assert isinstance(my_data.b, ConfigDict)
    assert isinstance(my_data.c, ConfigDict)
    assert isinstance(my_data.d, ConfigList)
    assert isinstance(my_data.e, ConfigList)


def test_config_inits_dicts_and_lists_of_configs():
    my_data = MyConfig(
        a=5,
        b={"ba": "bb", "bc": "bd"},
        c={"ca": dict(data=10.0), "cb": SimpleConfig(data=11.0)},
        d=[7, 8, 9],
        e=[dict(data=11.0), dict(data=12.0)],
        f=(5, 8.0),
    )
    assert isinstance(my_data.b, ConfigDict)
    assert isinstance(my_data.c, ConfigDict)
    assert isinstance(my_data.c['ca'], SimpleConfig)
    assert isinstance(my_data.c['cb'], SimpleConfig)
    assert isinstance(my_data.d, ConfigList)
    assert isinstance(my_data.e, ConfigList)
    assert isinstance(my_data.e[0], SimpleConfig)
    assert isinstance(my_data.e[1], SimpleConfig)
    assert my_data.b['ba'] == 'bb'
    assert my_data.b['bc'] == 'bd'
    assert my_data.c['ca'].data == 10.0
    assert my_data.c['cb'].data == 11.0
    assert list(my_data.d) == [7, 8, 9]
    assert my_data.e[0].data == 11.0
    assert my_data.e[1].data == 12.0

    my_data = MyConfig.from_json(my_data.as_json())
    assert isinstance(my_data.b, ConfigDict)
    assert isinstance(my_data.c, ConfigDict)
    assert isinstance(my_data.c['ca'], SimpleConfig)
    assert isinstance(my_data.c['cb'], SimpleConfig)
    assert isinstance(my_data.d, ConfigList)
    assert isinstance(my_data.e, ConfigList)
    assert isinstance(my_data.e[0], SimpleConfig)
    assert isinstance(my_data.e[1], SimpleConfig)
    assert my_data.b['ba'] == 'bb'
    assert my_data.b['bc'] == 'bd'
    assert my_data.c['ca'].data == 10.0
    assert my_data.c['cb'].data == 11.0
    assert list(my_data.d) == [7, 8, 9]
    assert my_data.e[0].data == 11.0
    assert my_data.e[1].data == 12.0


def test_as_json_is_recursive_on_dicts_and_lists():
    my_data = MyConfig(
        a=5,
        b={"ba": "bb", "bc": "bd"},
        c={"ca": SimpleConfig(data=10.0)},
        d=[7, 8, 9],
        e=[SimpleConfig(data=11.0), SimpleConfig(data=12.0)],
        f=(5, 8.0),
    )
    json_data = my_data.as_json()

    assert type(json_data['b']) is dict
    assert type(json_data['c']) is dict
    assert type(json_data['d']) is list
    assert type(json_data['e']) is list


def test_default_non_frozen_values_for_dicts_and_lists():
    class MyDeaultsConfig(Config):
        a: int
        b: Dict[str, str] = {}
        c: Dict[str, SimpleConfig] = {"ca": SimpleConfig(data=10.0)}
        d: List[int] = []
        e: List[SimpleConfig] = [SimpleConfig(data=11.0), SimpleConfig(data=12.0)]
        f: Tuple[int, float]

    my_data = MyDeaultsConfig(a=5, f=(5, 8.0))
    assert isinstance(my_data.b, ConfigDict)
    assert isinstance(my_data.c, ConfigDict)
    assert isinstance(my_data.c['ca'], SimpleConfig)
    assert isinstance(my_data.d, ConfigList)
    assert isinstance(my_data.e, ConfigList)
    assert isinstance(my_data.e[0], SimpleConfig)
    assert isinstance(my_data.e[1], SimpleConfig)
    assert len(my_data.b) == 0
    assert my_data.c['ca'].data == 10.0
    assert len(my_data.d) == 0
    assert my_data.e[0].data == 11.0
    assert my_data.e[1].data == 12.0

    my_data = MyDeaultsConfig(
        a=5,
        b={"ba": "bb", "bc": "bd"},
        c={"ca": dict(data=10.0), "cb": SimpleConfig(data=11.0)},
        d=[7, 8, 9],
        e=[dict(data=11.0), dict(data=12.0)],
        f=(5, 8.0),
    )
    assert isinstance(my_data.b, ConfigDict)
    assert isinstance(my_data.c, ConfigDict)
    assert isinstance(my_data.c['ca'], SimpleConfig)
    assert isinstance(my_data.c['cb'], SimpleConfig)
    assert isinstance(my_data.d, ConfigList)
    assert isinstance(my_data.e, ConfigList)
    assert isinstance(my_data.e[0], SimpleConfig)
    assert isinstance(my_data.e[1], SimpleConfig)
    assert my_data.b['ba'] == 'bb'
    assert my_data.b['bc'] == 'bd'
    assert my_data.c['ca'].data == 10.0
    assert my_data.c['cb'].data == 11.0
    assert list(my_data.d) == [7, 8, 9]
    assert my_data.e[0].data == 11.0
    assert my_data.e[1].data == 12.0


def test_config_dict_replace():
    class MyDictConfig(Config):
        a: int
        b: Dict[str, str]
        c: Dict[str, SimpleConfig]

    my_config = MyDictConfig(
        a=5,
        b={"ba": "bb", "bc": "bd"},
        c={"ca": dict(data=10.0), "cb": SimpleConfig(data=11.0)},
    )

    replaced = my_config.replace(
        b={'ba': 'ba', 'be': 'bf'},
        c={
            'ca': SimpleConfig(data=11.0),
            'cb': dict(data=10.0),
            'cc': SimpleConfig(data=11.0),
            'cd': dict(data=10.0),
        },
    )
    assert len(replaced.b) == 3
    assert len(replaced.c) == 4
    assert replaced.b['ba'] == 'ba'
    assert replaced.b['be'] == 'bf'
    assert replaced.c['ca'].data == 11.0
    assert replaced.c['cb'].data == 10.0
    assert replaced.c['cc'].data == 11.0
    assert replaced.c['cd'].data == 10.0


def test_config_dict_list_with_non_type_arguments():
    class MyDictConfig(Config):
        a: int
        b: Dict[str, Any]

    my_config = MyDictConfig(
        a=5,
        b={'a': 'b', 'c': {'d', 'e'}, 'f': [1, 2, 3], 'g': {'h': 'i'}},
    )
    assert isinstance(my_config.b, ConfigDict)
    # Make sure Any isn't cast to anything
    assert my_config.b == {'a': 'b', 'c': {'d', 'e'}, 'f': [1, 2, 3], 'g': {'h': 'i'}}

    # Test calling replace doesn't cause errors
    replaced = my_config.replace(
        b={
            'a': {'h': 'i'},
            'f': [4, 5],
            'g': 'j',
        }
    )
    # Make sure replaced values are correct
    assert replaced.b == {'a': {'h': 'i'}, 'c': {'d', 'e'}, 'f': [4, 5], 'g': 'j'}
    # Make sure original values are unaffected
    assert my_config.b == {'a': 'b', 'c': {'d', 'e'}, 'f': [1, 2, 3], 'g': {'h': 'i'}}

    class MyListConfig(Config):
        a: int
        b: List[Any]

    my_config = MyListConfig(
        a=5,
        b=['b', {'d', 'e'}, [1, 2, 3], {'h': 'i'}],
    )
    assert isinstance(my_config.b, ConfigList)
    assert list(my_config.b) == ['b', {'d', 'e'}, [1, 2, 3], {'h': 'i'}]


def test_config_pickles_single_process():
    # Make sure that Config with Dict and List fields can pickle in a single process
    config = PickleConfig(
        str_name='test',
        dict_int={'1': 2, '3': '4'},
        list_int=[1, 2, '3', 4],
        dict_dict={'1': {'2': 3}, '4': {'5': 6}},
        list_list=[[5.6, 7.8], [10.11, 13.14]],
    )

    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, 'output.pickle')

        with open(file_path, 'wb') as file:
            pickle.dump(config, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(file_path, 'rb') as file:
            p_config = pickle.load(file)

        assert config == p_config


def test_config_pickles_multiprocessing():
    # Make sure that Config with Dict and List fields can pickle when using torch
    config = PickleConfig(
        str_name='test',
        dict_int={'1': 2, '3': '4'},
        list_int=[1, 2, '3', 4],
        dict_dict={'1': {'2': 3}, '4': {'5': 6}},
        list_list=[[5.6, 7.8], [10.11, 13.14]],
    )

    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, 'output.pickle')

        torch.multiprocessing.spawn(worker_pickle_dump, args=(file_path, config), nprocs=1, join=True)

        with open(file_path, 'rb') as file:
            worker_config = pickle.load(file)

        assert config == worker_config


def test_config_recursive_list_preserves_family():
    class MyConfig(Config):
        a: List[List[int]]

    my_config = MyConfig(a=[[1, 2], [3]])
    assert isinstance(my_config.a, ConfigList)
    # The nested level stays in the Config family (ConfigList, not a plain DataclassList)
    assert isinstance(my_config.a[0], ConfigList)
    # apply traverses all the way to the leaves
    assert my_config.a.apply(lambda x: x * 2).as_json() == [[2, 4], [6]]


def test_config_recursive_dict_preserves_family():
    class MyConfig(Config):
        a: Dict[str, Dict[str, int]]

    my_config = MyConfig(a={"x": {"y": 1}})
    assert isinstance(my_config.a, ConfigDict)
    assert isinstance(my_config.a["x"], ConfigDict)
    assert my_config.a.apply(lambda v: v + 1).as_json() == {"x": {"y": 2}}


def test_config_dict_with_default():
    # Make sure that Dict defaults are merged with the arguments provided

    class MyConfigDefaults(Config):
        data: Dict[str, int] = {'a': 5, 'b': 7}

    config = MyConfigDefaults(data={'c': 2})
    assert config.data == {'a': 5, 'b': 7, 'c': 2}

    config = MyConfigDefaults(data={'c': 2, 'a': 8})
    assert config.data == {'a': 8, 'b': 7, 'c': 2}

    config = MyConfigDefaults(data={})
    assert config.data == {}  # pylint: disable=use-implicit-booleaness-not-comparison
