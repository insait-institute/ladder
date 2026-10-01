import pickle
from typing import Any, Dict, List

from ladder.core.databib.config.config import Config, Configurable


class ConfigJ(Config):
    pass


class ConfigurableJ(Configurable[Config]):
    pass


class PickleConfig(Config):
    str_name: str
    dict_int: Dict[str, Any]
    list_int: List[Any]
    dict_dict: Dict[str, Dict[str, float]]
    list_list: List[List[float]]


def worker_pickle_dump(_, file_path: str, value: Any):
    with open(file_path, 'wb') as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
