"""Native invocation and output descriptions. These objects contain no execution machinery."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qcdata import Data, ProgramInput, ProgramOutput


@dataclass(frozen=True)
class SubprocessInput:
    """A command and the files it needs, to run inside a calculation directory.

    Attributes:
        command: The argument vector; the executable is the first element.
        files: Files written to the calculation directory before running.
        wavefunction_files: Relative glob patterns of restart files worth keeping.
        environment: Environment defaults required by the program.
        threads: CPU threads the calculation uses, so a runner computing several
            at once can keep their total within the available cores.
        gpus: GPUs the calculation uses; a runner gives each calculation its own.
    """

    command: tuple[str, ...]
    files: Mapping[str, str | bytes] = field(default_factory=dict)
    wavefunction_files: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    threads: int = 1
    gpus: int = 0


@dataclass(frozen=True)
class PythonInput:
    """A Python callable with its arguments, to run inside a calculation directory.

    Attributes:
        function: The callable to invoke.
        args: Positional arguments.
        kwargs: Keyword arguments.
        files: Files written to the calculation directory before running.
        child_callback: Keyword name into which the runner injects a function that
            computes child calculations synchronously; only for integrations such
            as optimizers that drive other programs.
        children_callback: Keyword name into which the runner injects a function that
            computes several independent child calculations at once and returns
            their outputs in order (``ComputeChildren``), so a runner can batch or
            parallelize them; for drivers that need many structures per step, such
            as a path optimization's images. A runner that does not support it
            leaves the keyword out, so the function should fall back to
            ``child_callback`` one input at a time.
        loggers: Python logger names whose output should be captured.
        wavefunction_files: Relative glob patterns of restart files worth keeping.
        environment: Environment defaults required by the library.
        threads: CPU threads the calculation uses, so a runner computing several
            at once can keep their total within the available cores.
        gpus: GPUs the calculation uses; a runner gives each calculation its own.
    """

    function: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    files: Mapping[str, str | bytes] = field(default_factory=dict)
    child_callback: str | None = None
    children_callback: str | None = None
    loggers: tuple[str, ...] = ()
    wavefunction_files: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    threads: int = 1
    gpus: int = 0


@dataclass
class SubprocessOutput:
    """What a subprocess produced; ``stdout`` holds merged stdout and stderr."""

    stdout: str = ""
    directory: Path | None = None
    returncode: int | None = None
    error: Exception | None = None
    traceback: str | None = None

    @property
    def failed(self) -> bool:
        """Whether launching or running the process failed."""
        return self.error is not None or self.returncode not in (None, 0)


@dataclass
class PythonOutput:
    """What a Python callable produced: its return value, merged logs, or an error."""

    value: Any = None
    stdout: str = ""
    directory: Path | None = None
    error: Exception | None = None
    traceback: str | None = None

    @property
    def failed(self) -> bool:
        """Whether invoking the callable raised an exception."""
        return self.error is not None


NativeInput = SubprocessInput | PythonInput
NativeOutput = SubprocessOutput | PythonOutput


class ComputeChild(Protocol):
    """Runtime callback for integrations that request further calculations."""

    def __call__(
        self,
        input_data: ProgramInput,
        /,
        *,
        previous: ProgramOutput[ProgramInput, Data] | None = None,
    ) -> ProgramOutput[ProgramInput, Data]:
        """Execute one child synchronously and return its output, failed or not."""
        ...


class ComputeChildren(Protocol):
    """Runtime callback computing several independent child calculations at once."""

    def __call__(
        self, inputs: Sequence[ProgramInput], /
    ) -> list[ProgramOutput[ProgramInput, Data]]:
        """Execute the children, batched or in parallel where possible.

        Returns their outputs, failed or not, in input order.
        """
        ...
