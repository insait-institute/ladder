from enum import Enum
from typing import Any, Dict


class RequestType(str, Enum):
    RESET = "reset"
    GET_OBSERVATIONS = "get_observations"
    EXEC_ACTION = "exec_action"
    CONNECT = "connect"
    EXIT = "exit"


def parse_request_type(request: Dict[str, Any]) -> RequestType:
    try:
        request_type = RequestType(request["type"])
    except KeyError:
        raise ValueError("Invalid request: request type not specified.")
    except ValueError:
        raise ValueError("Invalid request type.")

    return request_type
