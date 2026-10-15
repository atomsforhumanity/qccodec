"""Exceptions raised by codecs, parsers, and the registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qcdata import StructuredData


class BaseError(Exception):
    """Base class for qccodec exceptions."""

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
    """Raised when native output cannot be decoded."""


class EncoderError(BaseError):
    """Raised when an input cannot be encoded for a program."""


class ParserError(BaseError):
    """Base class for parser failures; ``data`` holds any recovered results."""

    def __init__(self, message: str, data: StructuredData | None = None) -> None:
        """Retain the original failure and available calculation diagnostics."""
        super().__init__(message)
        self.data: StructuredData | None = data


class MatchNotFoundError(ParserError):
    """Raised when a parser finds no match for its pattern."""

    def __init__(self, regex: str, contents: str) -> None:
        """Retain the original failure and available calculation diagnostics."""
        self.regex: str = regex
        self.contents: str = contents
        super().__init__(
            f"Could not locate match for regex: '{regex}' in contents: '{contents}'"
        )


class RegistryError(BaseError):
    """Raised when a codec or parser registration is invalid."""


class DataCollectorError(BaseError):
    """Raised when two parsers fill the same result field."""


class CodecNotFoundError(EncoderError):
    """No built-in integration is available for this name."""


class CalculationError(ParserError):
    """The program reported failure; data holds any recovered scientific results."""


class DependencyNotFoundError(BaseError):
    """A codec explicitly reports an unavailable execution dependency."""
