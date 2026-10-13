"""Local invocation descriptions. These objects contain no execution machinery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qcdata import Data, ProgramInput, ProgramOutput, StructuredData


@dataclass(frozen=True)
class SubprocessInput:
    """An argument vector and files to execute within a calculation workspace."""

    command: tuple[str, ...]
    files: Mapping[str, str | bytes] = field(default_factory=dict)
    wavefunction_files: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PythonInput:
    """A native callable, its arguments, and optional execution requirements."""

    function: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    files: Mapping[str, str | bytes] = field(default_factory=dict)
    # Only callback-driven integrations request this injected keyword argument.
    child_callback: str | None = None
    loggers: tuple[str, ...] = ()
    wavefunction_files: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass
class SubprocessOutput:
    """Subprocess state and artifacts; stdout holds merged process output."""

    stdout: str = ""
    directory: Path | None = None
    returncode: int | None = None
    error: Exception | None = None
    traceback: str | None = None

    @property
    def failed(self) -> bool:
        """Report an invocation failure independently of scientific completion."""
        return self.error is not None or self.returncode not in (None, 0)


@dataclass
class PythonOutput:
    """A callable's return value, merged output in stdout, or invocation failure."""

    value: Any = None
    stdout: str = ""
    directory: Path | None = None
    error: Exception | None = None
    traceback: str | None = None

    @property
    def failed(self) -> bool:
        """Report whether invoking the Python function raised an exception."""
        return self.error is not None


NativeInput = SubprocessInput | PythonInput
NativeOutput = SubprocessOutput | PythonOutput


class Codec(Protocol):
    """Two methods are sufficient for a local, externally supplied integration."""

    def encode(self, input_data: ProgramInput) -> NativeInput:
        """Translate a scientific request without executing the calculation."""
        ...

    def decode(
        self, input_data: ProgramInput, native_output: NativeOutput
    ) -> StructuredData:
        """Interpret native output, returning data or raising with partial data."""
        ...


class ComputeChild(Protocol):
    """Runtime callback for integrations that request further calculations."""

    def __call__(
        self,
        input_data: ProgramInput,
        /,
        *,
        previous: ProgramOutput[ProgramInput, Data] | None = None,
        codec: Codec | None = None,
    ) -> ProgramOutput[ProgramInput, Data]:
        """Execute one child synchronously and retain its output on failure."""
        ...
