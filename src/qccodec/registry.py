from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from qcdata import CalcType

from qccodec.exceptions import RegistryError


@dataclass
class ParserSpec:
    """Specification for a parser function.

    This class holds information about the parser function, the filetype it operates on,
    whether it is required, the calculation types it works on, and where to place the
    parsed output on the data collector.

    Attributes:
        parser: The parser function.
        filetype: The filetype that the parser is for.
        required: Whether a MatchNotFoundError should be raised in the decode function
            if the parser fails. Typically True, but should be set to False on parsers
            for values that are not always present such as excited state energies.
        calctypes: The calculation types that the parser works on.
        target: The location on the data collector where the parser's output should be placed.
    """

    parser: Callable[..., Any]
    filetype: Enum | str
    required: bool
    calctypes: list[CalcType]
    target: str | tuple[str, ...] | None = None
    explicit_calctypes: bool = True

    def __post_init__(self) -> None:
        """Ensure that the parser function is Callable and that target exists unless filetype is directory."""
        if not callable(self.parser):
            raise RegistryError(f"Parser '{self.parser}' is not Callable.")
        # Ensure all non-directory parsers have a target
        if (
            self.target is None
            and getattr(self.filetype, "value", self.filetype) != "directory"
        ):
            raise RegistryError(
                f"Parser '{self.parser.__name__}' must have a target if it is not a directory parser."
            )
        # Ensure that the target is a string or tuple of strings
        if self.target and not isinstance(self.target, (str, tuple)):
            raise RegistryError(
                f"Target '{self.target}' must be a string or tuple of strings."
            )
