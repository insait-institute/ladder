from typing import Any, Dict

from ladder.components.inference.api.core.request_reply import (
    CallbackProtocol,
    RequestReplyServer,
    RequestReplyServerConfig,
)
from ladder.components.inference.api.core.requests import RequestType, parse_request_type
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable


class RequestListenerConfig(Config):
    req_rep_server_config: AutoConfig[RequestReplyServerConfig]


class RequestListener(Configurable[RequestListenerConfig]):
    """
    Listens for requests and dispatches them to the appropriate handler
    based on the request type.

    Methods:
        start(): Starts listening for requests
        register_handler(request_type: RequestType, handler): Registers a handler for a specific request type
    """

    def __init__(self, config: RequestListenerConfig):
        super().__init__(config)
        self._req_rep_server: RequestReplyServer = AutoConfigurable(self.config.req_rep_server_config)
        self._handlers: Dict[RequestType, CallbackProtocol] = dict()

    def start(self) -> None:
        """
        Starts listening for requests

        Notes: this function is blocking
        """
        self._req_rep_server.start(self._on_request)

    def _on_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_type = parse_request_type(request)
        handler = self._handlers.get(request_type, None)
        if not handler:
            raise ValueError(f"No handler was registered for request of type {request_type}")
        return handler(request["data"])

    def register_handler(self, request_type: RequestType, handler: CallbackProtocol) -> None:
        """
        Registers a handler for a specific request type

        Args:
            request_type (RequestType): The request type to register the handler for
        """
        if request_type in self._handlers:
            raise ValueError(f"Handler for request type {request_type} is already registered")
        self._handlers[request_type] = handler
