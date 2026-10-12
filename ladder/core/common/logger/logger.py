import inspect
import logging
import os
import signal
import sys
import threading
from typing import Any, Dict, Set, Union

from .colorize import colorize

DEBUG_PATTERNS = [patt for patt in os.environ.get('DEBUG', '').split(',') if patt]

# Default overrides for modules with spammy logs. Patterns should be comma-separated
DEFAULT_LOG_MODULE_OVERRIDES = 'filelock=ERROR'


def _parse_log_overrides(log_override_pattern: str) -> Dict[str, str]:
    """
    Parses log override settings:
    Args:
        log_override_pattern: str with comma-separated values. Each value should be of the format
            `path.to.module=LOG_LEVEL`, e.g. `ladder.core.common.logger.logger=INFO`
    Returns:
        Dict of module names and corresponding log level
    """
    overrides = {}
    for setting in log_override_pattern.split(','):
        parts = setting.strip().split('=')
        if len(parts) < 2:
            continue
        overrides[parts[0]] = parts[1]
    return overrides


LOG_MODULE_OVERRIDES = _parse_log_overrides(
    ','.join([DEFAULT_LOG_MODULE_OVERRIDES, os.environ.get('LOG_MODULE_OVERRIDES', '')])
)

ROOT_LOGGER_LEVEL: str = os.environ.get('ROOT_LOGGER_LEVEL', 'INFO')

_root_logger_set = False
_root_logger_lock = threading.Lock()

# Loggers whose records must not reach the console. Enforced by a single filter on the console handler
# created in `get_root_logger`, so a logger can be excluded at any point without any caller having to
# find handlers itself - and without risk of catching a FileHandler, which subclasses StreamHandler.
_CONSOLE_EXCLUDED_LOGGERS: Set[str] = set()


def _disable_console_logging(logger_name: str) -> None:
    """
    Stop `logger_name`'s records reaching the console. Every other handler still receives them.

    For processes that share one terminal but log to separate files - e.g. every rank of a distributed
    job streaming to the same stdout - where the duplicate copies add nothing because each process
    already writes its own log file.
    """
    _CONSOLE_EXCLUDED_LOGGERS.add(logger_name)


class LadderLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._once_call_sites: Set[tuple] = set()

    @property
    def name(self):
        return self._logger.name

    def is_enabled_for_level(self, level: int):
        return self._logger.isEnabledFor(level)

    def add_handler(self, handler: logging.Handler) -> None:
        self._logger.addHandler(handler)

    def debug(self, event_type: str, **info) -> None:
        self._log(logging.DEBUG, event_type, info)

    def info(self, event_type: str, color: str | None = None, **info) -> None:
        self._log(logging.INFO, event_type, info, color=color)

    def info_once(self, event_type: str, **info) -> None:
        """Like info, but will only print for the first time and drop the log statement afterwards."""

        assert type(self) is LadderLogger, "info_once will not work when subclassing the logger"
        # It will also not work when delegating the logger etc.

        call_site = inspect.stack()[1]
        if (call_site.filename, call_site.lineno) not in self._once_call_sites:
            self._once_call_sites.add((call_site.filename, call_site.lineno))
            self._log(logging.INFO, event_type, info)

    def warn(self, event_type: str, **info) -> None:
        self._log(logging.WARN, event_type, info, color='yellow')

    def error(self, event_type: str, **info) -> None:
        self._log(logging.ERROR, event_type, info, color='red')

    def critical(self, event_type: str, **info) -> None:
        self._log(logging.CRITICAL, event_type, info, color='red')

    def exception(self, event_type: str, _: Exception, **info) -> None:
        self._log(logging.ERROR, event_type, info, log=self._log_exception)

    def log(self, level: int, event_type: str, **info) -> None:
        self._log(level, event_type, info)

    def _log(
        self, level: int, event_type: str, info: Dict[str, Any], log=None, color: str | None = None
    ) -> None:
        if log is None:
            log = self._logger.log

        if not self._logger.isEnabledFor(level):
            return

        extra = {'event_type': event_type}

        if info:
            extra.update(info)
            if 'message' in extra:
                extra.pop('message')  # message is set by logger.
            message = '%s %s' % (event_type, ", ".join(f"{k}={v}" for k, v in info.items()))
        else:
            message = event_type

        if color:
            message = colorize(message, color)

        log(level, message, extra=extra)

    def _log_exception(self, _, msg, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)

    def close(self) -> None:
        self._logger.handlers.clear()

    def set_level(self, level: Union[int, str]) -> None:
        self._logger.setLevel(level)

    def disable_console(self) -> None:
        """
        Stop this logger's records reaching the console; every other handler still receives them.

        Use when the decision depends on runtime state that isn't known at import time - e.g. the
        distributed rank. Otherwise prefer `get_logger(..., console=False)`.
        """
        _disable_console_logging(self.name)


def is_interactive() -> bool:
    """Determines whether the python application has been invoked from an interactive shell or not"""
    return sys.stdin.isatty()


def get_root_logger() -> logging.Logger:
    global _root_logger_set  # noqa: PLW0603 (global-statement)
    with _root_logger_lock:
        root_logger = logging.getLogger()
        if not _root_logger_set:
            root_logger.handlers.clear()  # There could be some default handlers we don't care about. Remove them.
            root_logger.setLevel(ROOT_LOGGER_LEVEL)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(ROOT_LOGGER_LEVEL)

            formatter = logging.Formatter(
                '%(levelname)-8s %(asctime)-15s \033[1;96m%(name)-12s\033[0m %(message)s'
            )

            stream_handler.setFormatter(formatter)
            stream_handler.addFilter(lambda record: record.name not in _CONSOLE_EXCLUDED_LOGGERS)
            root_logger.addHandler(stream_handler)
            _root_logger_set = True
    return root_logger


def get_logger(module_name: str, level=logging.INFO, console: bool = True) -> LadderLogger:
    """
    Args:
        module_name: Logger name, normally `__name__`
        level: Level for this logger
        console: If False, this logger's records go to every handler except the console - useful when
            many processes share one terminal but each also writes its own log file. Evaluated here, so
            only use it for a decision known at import time; otherwise call
            `LadderLogger.disable_console()` once the decision is made.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    get_root_logger()  # Ensure root logger exists.

    if not console:
        _disable_console_logging(module_name)

    if any(module_name.startswith(patt) for patt in DEBUG_PATTERNS):
        logger.setLevel(logging.DEBUG)

        handlers = [h for h in logging.root.handlers if isinstance(h, logging.StreamHandler)]
        for handler in handlers:
            handler.setLevel(logging.DEBUG)

    # Override manually configured loggers
    _set_log_level_overrides()

    return LadderLogger(logger)


def enter_ipdb_breakpoint_on_signal(signal_type=signal.SIGUSR1):
    def handle_signal(_, frame):
        import ipdb  # noqa: T100

        ipdb.set_trace(frame)  # noqa: T100

    signal.signal(signal_type, handle_signal)
    print(f'Run `kill -{signal_type} {os.getpid()}` to enter ipdb`')


def _set_log_level_overrides() -> None:
    """
    Override certain loggers in `LOG_MODULE_OVERRIDES` to their corresponding levels.
    Used to set logger levels from command line (via env variable) or to silence modules with spam logs.
    """
    for module, level in LOG_MODULE_OVERRIDES.items():
        logging.getLogger(module).setLevel(level)
