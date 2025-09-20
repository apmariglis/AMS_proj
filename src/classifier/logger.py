import logging
import os
import sys


class ConsoleLogger:
    """Console logger with optional color and severity prefixes, no icons."""

    _LEVEL_COLORS = {
        logging.DEBUG: "\033[37m",  # gray
        logging.INFO: "\033[36m",  # cyan
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    _RESET = "\033[0m"

    def __init__(
        self,
        name: str = "mzpredict",
        level: int = logging.INFO,
        stream=None,
        use_color: bool | None = None,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers if constructed multiple times (e.g., tests)
        if not self.logger.handlers:
            if stream is None:
                stream = sys.stderr
            if use_color is None:
                use_color = stream.isatty() and os.environ.get("NO_COLOR") is None

            class _Formatter(logging.Formatter):
                def __init__(self, use_color):
                    super().__init__("%(message)s")
                    self.use_color = use_color

                def format(self, record):
                    msg = super().format(record)
                    level = record.levelname
                    if self.use_color:
                        color = ConsoleLogger._LEVEL_COLORS.get(record.levelno, "")
                        return f"{color}{level}: {msg}{ConsoleLogger._RESET}"
                    return f"{level}: {msg}"

            handler = logging.StreamHandler(stream)
            handler.setFormatter(_Formatter(use_color))
            self.logger.addHandler(handler)

    # Convenience methods (so you can call log.info(...))
    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)
