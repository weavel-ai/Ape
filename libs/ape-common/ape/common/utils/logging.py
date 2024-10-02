import logging
import os
import sys
import typing as t

import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

console = Console(width=200, theme=Theme({"logging.level": "bold"}))

LOGGER_NAME = "ape"

logger: structlog.stdlib.BoundLogger = structlog.get_logger(LOGGER_NAME)


class LogSettings:
    def __init__(self, output_type: str, method: str, file_name: t.Optional[str]) -> None:
        self.output_type = output_type
        self.method = method
        self.file_name = file_name
        self._configure_structlog()

    def _configure_structlog(self):
        if self.output_type == "str":
            renderer = self._create_rich_renderer()
            # renderer = structlog.dev.ConsoleRenderer(colors=True)
        else:
            renderer = structlog.processors.JSONRenderer()

        structlog.configure(
            processors=[
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                renderer,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def _create_rich_renderer(self):
        def rich_renderer(logger, name, event_dict):
            event = event_dict.pop("event", "")
            # Remove redundant information
            event_dict.pop("logger", None)

            return event

        return rich_renderer

    def set_log_output(
        self,
        method: t.Optional[str] = None,
        file_name: t.Optional[str] = None,
        output_type: t.Optional[str] = None,
    ) -> None:
        if method is not None and method not in ["console", "file"]:
            raise ValueError("method provided can only be 'console', 'file'")

        if method == "file" and file_name is None:
            raise ValueError("file_name must be provided when method = 'file'")

        if method is not None:
            self.method = method
            self.file_name = file_name

        if output_type is not None and output_type not in ["str", "json"]:
            raise ValueError("output_type provided can only be 'str', 'json'")

        if output_type is not None:
            self.output_type = output_type

        # Update Renderer
        self._configure_structlog()

        # Grab the ape logger
        log = logging.getLogger(LOGGER_NAME)
        for handler in log.handlers[:]:
            log.removeHandler(handler)

        # Add new Handler
        if self.method == "file":
            assert self.file_name is not None
            file_handler = logging.FileHandler(self.file_name)
            file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            log.addHandler(file_handler)
        else:
            rich_handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
            rich_handler.setFormatter(logging.Formatter())
            log.addHandler(rich_handler)


level = os.environ.get("APE_LOG_LEVEL", "info").upper()


# Set Defaults
def show_logging(level: str = level) -> None:
    ape_logger = logging.getLogger(LOGGER_NAME)
    ape_logger.setLevel(level)
    ape_logger.handlers = [RichHandler(console=console, rich_tracebacks=True, markup=True)]


show_logging(level)
settings = LogSettings(output_type="str", method="console", file_name=None)
set_log_output = settings.set_log_output
