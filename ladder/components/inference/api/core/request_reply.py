from abc import abstractmethod
from typing import Any, Dict, Optional, TypeVar

import zmq
from typing_extensions import Protocol

from ladder.components.inference.api.core.requests import RequestType, parse_request_type
from ladder.components.inference.api.core.serialization import deserialize, serialize
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable
from ladder.core.databib.template import Template

LOGGER = get_logger(__name__)


class CallbackProtocol(Protocol):
    def __call__(self, message: Dict) -> Dict:
        ...


class RequestReplyServerConfig(Config):
    pass


RequestReplyServerConfigT = TypeVar('RequestReplyServerConfigT', bound=RequestReplyServerConfig)


class RequestReplyServer(Configurable[RequestReplyServerConfigT], Template[RequestReplyServerConfigT]):
    """
    Abstract base class for a request-reply server that listens
    for requests, dispatches them to the provided callback, and
    sends the output back to the client.

    Methods:
        start(on_request_cb: CallbackProtocol): Begins listening for requests
    """

    @abstractmethod
    def start(self, on_request_cb: CallbackProtocol) -> None:
        pass


class RequestReplyClientConfig(Config):
    pass


RequestReplyClientConfigT = TypeVar('RequestReplyClientConfigT', bound=RequestReplyClientConfig)


class RequestReplyClient(Configurable[RequestReplyClientConfigT], Template[RequestReplyClientConfigT]):
    """
    Abstract base class for a request-reply client that sends
    requests to the server and receives the output from the server.

    Methods:
        send(request: Dict) -> Optional[Dict]: Send a request to the server.
    """

    @abstractmethod
    def send(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass


class ZmqRequestReplyServerConfig(RequestReplyServerConfig):
    port: int  # Port number to listen on
    recv_timeout_ms: int = 5000  # Timeout in milliseconds for calls to socket.recv


class ZmqRequestReplyServer(RequestReplyServer[ZmqRequestReplyServerConfig]):
    """
    ZeroMQ-based implementation of RequestReplyServer.

    Methods:
        start(on_request_cb: CallbackProtocol): Begins listening for requests.

    Notes:
        Designed to be used in conjunction with the ZmqRequestReplyClient class.
    """

    def __init__(self, config: ZmqRequestReplyServerConfig):
        super().__init__(config)
        self._context = zmq.Context()
        self._init_socket()

    def _init_socket(self):
        self.socket = self._context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.config.port}")
        self.socket.setsockopt(zmq.SNDHWM, 5)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)

    def start(self, on_request_cb: CallbackProtocol) -> None:
        """
        Starts listening for requests over the socket, blocking on the current thread.
        When a request is received, the on_request_cb function is called with the request
        as an argument, and the output is sent back as a reply to the client.

        Args:
            on_request_cb (CallbackProtocol): Function to be called when a request is received.

        Notes: this function is blocking
        """
        LOGGER.info("Starting ZMQ request reply server")
        while True:
            LOGGER.debug("Waiting for new request")
            try:
                message = self.socket.recv()
            except zmq.Again:
                LOGGER.warn("ZMQ receive timeout. Continuing.")
                continue
            except zmq.ZMQError as e:
                LOGGER.error(f"A ZMQ error occured while receiving. Error: {e}")
                raise e
            LOGGER.debug(f"Received new request")

            LOGGER.debug(f"Processing new request")
            message = deserialize(message)
            res = on_request_cb(message)
            res = serialize(res)
            LOGGER.debug(f"Request processed")

            LOGGER.debug(f"Sending reply")
            try:
                self.socket.send(res)
            except zmq.ZMQError as e:
                LOGGER.error(f"A ZMQ error occured while sending. Error: {e}")
                raise e
            LOGGER.debug(f"Reply sent")

            # If the request is an exit request, we stop the loop and
            # let the process exit gracefully.
            if parse_request_type(message) == RequestType.EXIT:
                LOGGER.info("Received EXIT request. Stopping server.")
                break

    def __del__(self):
        if self.socket:
            self.socket.close()
        self._context.term()


class ZmqRequestReplyClientConfig(RequestReplyClientConfig):
    ip: str  # IP address of the server
    port: int  # Port number of the server
    recv_timeout_ms: int = 15000  # Timeout in milliseconds for calls to socket.recv
    send_num_retries: int = 3  # Number of times to retry sending a message
    # If True, recv timeouts are raised as ``zmq.Again`` to the caller after the broken
    # REQ socket has been reset. Use when a timeout indicates the worker should fail and
    # be respawned by its supervisor (e.g. a remote inference client). Default False
    # preserves the legacy "log and return None" behaviour.
    propagate_timeout: bool = False


class ZmqRequestReplyClient(RequestReplyClient[ZmqRequestReplyClientConfig]):
    """
    ZeroMQ-based implementation of RequestReplyClient.

    It communicates with the server using a ZeroMQ in REQ mode.

    Methods:
        send(request: dict) -> Optional[Dict]: Send a request to the server and
            receive a reply.

    Notes:
        This class is designed to be used in conjunction with the
        ZmqRequestReplyServer class.
    """

    def __init__(self, config: ZmqRequestReplyClientConfig):
        super().__init__(config)
        self._closed = False
        self.context = zmq.Context()
        self.socket = None
        self._init_socket()

    def _init_socket(self):
        """
        Reset the socket connection, this is needed when REQ is in a broken state.
        """
        if self.socket:
            LOGGER.info(f"Closing and deleting existing socket")
            self.socket.close()
            del self.socket
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        # this option is needed, among other things, to prevent context.term()
        # from handing if socket.recv() is interrupted e.g. by a KeyboardInterrupt
        self.socket.setsockopt(zmq.LINGER, 0)

        LOGGER.info(f"Connecting to {self.config.ip}:{self.config.port}")
        self.socket.connect(f"tcp://{self.config.ip}:{self.config.port}")
        LOGGER.info(f"Connected to {self.config.ip}:{self.config.port}")

    def send(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.socket is None or self.socket.closed:
            LOGGER.error(f"The socket is closed, request not sent.")
            return None

        try:
            LOGGER.debug(f"Sending request")
            sent = False
            request_serialized = serialize(request)
            for i in range(self.config.send_num_retries):
                try:
                    self.socket.send(request_serialized)
                    sent = True
                    LOGGER.debug(f"Request sent.")
                    break
                except zmq.ZMQError as e:
                    LOGGER.error(
                        f"Failed to send message to {self.config.ip}:{self.config.port}. Error: {e}."
                    )

                LOGGER.debug(f"Attempt {i} failed. Retrying.")

            if not sent:
                LOGGER.warn("Number of retries exceeded. Aborting.")
                return None

            # If the request is an exit request, we don't wait for a reply.
            if request.get("type") == RequestType.EXIT:
                return None

            LOGGER.debug(f"Waiting for reply")
            try:
                message = self.socket.recv()
            except zmq.Again:
                # REQ socket is now stuck in "waiting for reply" — any subsequent send
                # would fail with ZMQError. Reset before deciding how to surface the timeout.
                self._init_socket()
                if self.config.propagate_timeout:
                    raise
                LOGGER.warn("ZMQ receive timeout. Aborting.")
                return None
            except zmq.ZMQError as e:
                LOGGER.error(f"A ZMQ error occured while receiving. Error: {e}.")
                raise e

            return deserialize(message)

        except KeyboardInterrupt:
            # Ctrl+C might be sent mid-episode. In this case, if the exception is raised inside the try
            # block, the socket will fall into a broken state. Here we reinitialize it to keep the
            # communication working.
            self._init_socket()
            raise

    def close(self) -> None:
        """Idempotently tear down the socket and ZMQ context. Safe to call multiple times."""
        if self._closed:
            return
        if self.socket:
            try:
                self.socket.disconnect(f"tcp://{self.config.ip}:{self.config.port}")
            except zmq.ZMQError:
                # Socket already disconnected (e.g. _init_socket re-created it after a timeout).
                pass
            self.socket.close()
        self.context.term()
        self._closed = True

    def __del__(self):
        self.close()
