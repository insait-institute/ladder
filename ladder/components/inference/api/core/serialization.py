from typing import Any, Dict

import msgpack
import msgpack_numpy

# apply msgpack_numpy monkey-patch to enable automatic numpy serialization
msgpack_numpy.patch()


def serialize(data: Dict[str, Any]) -> bytes:
    return msgpack.packb(data)


def deserialize(message: bytes) -> Dict[str, Any]:
    return msgpack.unpackb(message)
