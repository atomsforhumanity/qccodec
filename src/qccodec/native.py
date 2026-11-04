"""Native invocation and output descriptions. These objects contain no execution machinery."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, overload

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
        compute_callback: Keyword name into which the runner injects its ``Compute``,
            which computes child calculations: one input, or a list computed
            together (batched or side by side). For drivers, such as optimizers,
            that run other programs.
        child_callback: Deprecated; runners before qccompute 0.23 inject a
            one-input ``ComputeChild`` here. Built-in drivers name it next to
            ``compute_callback`` until those runners are no longer supported.
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
    compute_callback: str | None = None
    child_callback: str | None = None
    loggers: tuple[str, ...] = ()
    wavefunction_files: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    threads: int = 1
    gpus: int = 0

    def __post_init__(self) -> None:
        """Warn when only the deprecated ``child_callback`` is named."""
        if self.child_callback and not self.compute_callback:
            warnings.warn(
                "PythonInput.child_callback is deprecated; name compute_callback, "
                "into which runners inject a Compute.",
                DeprecationWarning,
                stacklevel=3,
            )


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


class Compute(Protocol):
    """Computes child calculations for a driver; a runner injects it.

    The call takes what ``qccompute.compute`` takes: one input, or a list of
    independent inputs that the runner may batch or run side by side. It returns
    outputs, failed or not, and never raises for a failed calculation.

    ``previous`` holds outputs whose wavefunctions start the children (one output for
    one input; one output or ``None`` per input for a list). Runners use it only when
    asked to propagate wavefunctions, such as qccompute's ``propagate_wfn=True``.
    """

    @overload
    def __call__(
        self,
        inputs: ProgramInput,
        /,
        *,
        previous: ProgramOutput[ProgramInput, Data] | None = None,
    ) -> ProgramOutput[ProgramInput, Data]: ...

    @overload
    def __call__(
        self,
        inputs: Sequence[ProgramInput],
        /,
        *,
        previous: Sequence[ProgramOutput[ProgramInput, Data] | None] | None = None,
    ) -> list[ProgramOutput[ProgramInput, Data]]: ...


class ComputeChild(Protocol):
    """Deprecated one-input callback of runners before qccompute 0.23; see Compute."""

    def __call__(
        self,
        input_data: ProgramInput,
        /,
        *,
        previous: ProgramOutput[ProgramInput, Data] | None = None,
    ) -> ProgramOutput[ProgramInput, Data]:
        """Execute one child synchronously and return its output, failed or not."""
        ...


def sequential(compute_one: ComputeChild) -> Compute:
    """Make a ``Compute`` from a function that computes one input at a time.

    For running a driver outside a runner, such as in tests, and for runners that
    compute children one by one: the inputs of a list are computed in order.
    """

    def compute(
        inputs: ProgramInput | Sequence[ProgramInput],
        /,
        *,
        previous: Any = None,
    ) -> Any:
        if isinstance(inputs, ProgramInput):
            return compute_one(inputs, previous=previous)
        starts = list(previous) if previous is not None else [None] * len(inputs)
        if len(starts) != len(inputs):
            raise ValueError(
                f"previous holds {len(starts)} outputs for {len(inputs)} inputs."
            )
        return [
            compute_one(inp, previous=start)
            for inp, start in zip(inputs, starts, strict=True)
        ]

    return cast(Compute, compute)
