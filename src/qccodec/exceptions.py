from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qcdata import StructuredData


class BaseError(Exception):
    """Base qccodec exceptions"""

    def __reduce__(self) -> tuple[Callable[..., Exception], tuple[object, ...]]:
        """Preserve error attributes and constructor-independent pickling."""
        return _restore_error, (type(self), self.args, self.__dict__)


def _restore_error(
    cls: type[Exception], args: tuple[Any, ...], attributes: dict[str, Any]
) -> Exception:
    """Restore error state without invoking subclass-specific constructors."""
    error = cls.__new__(cls)
    Exception.__init__(error, *args)
    error.__dict__.update(attributes)
    return error


class DecoderError(BaseError):
    """Exception raised when a decoder error occurs"""


class EncoderError(BaseError):
    """Exception raised when a encoder error occurs"""


class ParserError(BaseError):
    """Base exception for parsers"""

    def __init__(self, message: str, data: StructuredData | None = None) -> None:
        """Retain the original failure and available calculation diagnostics."""
        super().__init__(message)
        self.data: StructuredData | None = data


class MatchNotFoundError(ParserError):
    """Exception raised when a parsing match is not found"""

    def __init__(self, regex: str, contents: str) -> None:
        """Retain the original failure and available calculation diagnostics."""
        self.regex: str = regex
        self.contents: str = contents
        super().__init__(
            f"Could not locate match for regex: '{regex}' in contents: '{contents}'"
        )


class RegistryError(BaseError):
    """Exception raised when a registry error occurs"""


class DataCollectorError(BaseError):
    """Exception raised when a data collector error occurs"""


class CodecNotFoundError(EncoderError):
    """No built-in integration is available for this name."""


class CalculationError(ParserError):
    """The program reported failure; data holds any recovered scientific results."""


class DependencyNotFoundError(BaseError):
    """A codec explicitly reports an unavailable execution dependency."""
