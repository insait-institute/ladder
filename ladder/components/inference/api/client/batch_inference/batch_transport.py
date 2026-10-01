"""Transport abstraction for ``BatchPolicyServer``."""

from abc import ABC, abstractmethod
from typing import Tuple

import zmq

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


class BatchTransport(ABC):
    """Bytes-and-routing layer. Wire format is owned by the server, not the transport."""

    @abstractmethod
    def poll(self, timeout_ms: int) -> bool:
        """Block up to ``timeout_ms`` for a request. Return True if one is ready."""

    @abstractmethod
    def receive(self) -> Tuple[bytes, bytes]:
        """``(identity, payload_bytes)`` for the next request. Call only after ``poll`` returns True."""

    @abstractmethod
    def send(self, identity: bytes, payload: bytes) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class ZmqBatchTransport(BatchTransport):
    """ZMQ ROUTER socket. ``identity`` is the ROUTER framing prefix."""

    def __init__(self, port: int) -> None:
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 0)
        self._socket.bind(f"tcp://*:{port}")
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        LOGGER.info(f"ZMQ batch transport bound on port {port}")

    def poll(self, timeout_ms: int) -> bool:
        events = dict(self._poller.poll(timeout=timeout_ms))
        return self._socket in events

    def receive(self) -> Tuple[bytes, bytes]:
        parts = self._socket.recv_multipart()
        return parts[0], parts[-1]

    def send(self, identity: bytes, payload: bytes) -> None:
        self._socket.send_multipart([identity, b"", payload])

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()
