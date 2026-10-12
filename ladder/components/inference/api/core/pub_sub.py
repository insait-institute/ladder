from abc import abstractmethod
from typing import Any, Dict, TypeVar

import zmq
from typing_extensions import Protocol

from ladder.components.inference.api.core.serialization import deserialize, serialize
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)


class PublisherConfig(Config):
    pass


PublisherConfigT = TypeVar('PublisherConfigT', bound=PublisherConfig)


class Publisher(Configurable[PublisherConfigT]):
    @abstractmethod
    def publish(self, data: Dict[str, Any]) -> None:
        pass


class SubscriberCallbackT(Protocol):
    def __call__(self, message: Dict) -> Dict:
        ...


class SubscriberConfig(Config):
    pass


SubscriberConfigT = TypeVar('SubscriberConfigT', bound=SubscriberConfig)


class Subscriber(Configurable[SubscriberConfigT]):
    @abstractmethod
    def start(self, on_message_cb: SubscriberCallbackT) -> None:
        pass


class ZmqPublisherConfig(PublisherConfig):
    port: int

    # high water mark value for outbound messages i.e. maximum number
    # of outstanding messages that can be kept in the publisher queue
    #
    # for more info on the SNDHWM of ZMQ sockets please refer to:
    # http://api.zeromq.org/4-2:zmq-setsockopt
    send_hwm: int = 2


class ZmqPublisher(Publisher[ZmqPublisherConfig]):
    def __init__(self, config: ZmqPublisherConfig):
        super().__init__(config)

        self._context = zmq.Context()
        self._init_socket()

    def __del__(self):
        if self._socket:
            self._socket.close()
        self._context.term()

    def _init_socket(self) -> None:
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{self.config.port}")
        self._socket.setsockopt(zmq.SNDHWM, self.config.send_hwm)
        self._socket.setsockopt(zmq.LINGER, 0)

    def publish(self, data: Dict[str, Any], topic: str) -> None:
        message_bytes = serialize(data)
        self._socket.send_multipart([topic.encode(), message_bytes])


class ZmqSubscriberConfig(SubscriberConfig):
    ip: str
    port: int
    topic: str = ""


class ZmqSubscriber(Subscriber[ZmqSubscriberConfig]):
    def __init__(self, config: ZmqSubscriberConfig):
        super().__init__(config)

        self._running = False
        self._context = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self.config.ip}:{self.config.port}")
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.subscribe(self.config.topic.encode())

    def __del__(self):
        if self._socket:
            self._socket.close()
        self._context.term()

    def start(self, on_message_cb: SubscriberCallbackT):
        self._running = True
        while self._running:
            try:
                parts = self._socket.recv_multipart()
                if len(parts) != 2:
                    LOGGER.warn(f"socker.recv_multipart received {len(parts)} parts instead of 2. Skipping.")
                    continue
                topic, message_bytes = parts
                message = deserialize(message_bytes)
                on_message_cb(message)
            except zmq.Again:
                LOGGER.warn("ZMQ receive timeout. Continuing.")
                continue
            except zmq.ZMQError as e:
                LOGGER.error(f"A ZMQ error occured while receiving. Error: {e}")
                raise e

    def stop(self):
        self._running = False
