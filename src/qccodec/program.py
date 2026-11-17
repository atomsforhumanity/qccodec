"""Each codec as a program: how qccompute runs external programs.

`CodecProgram` wraps a `Codec` in the program protocol of ``qccompute.program``
(structurally: running a program needs no qccompute import). Every built-in codec
is registered as one under the ``qccompute.programs`` entry points, so
``qccompute.compute(ProgramInput(program="orca", ...))`` finds it by name. A call
encodes the input, runs the command (or the Python callable) in a scratch directory
with its output captured and a deadline, decodes what it wrote, and returns a
`ProgramOutput`; a failure raises `qcdata.ProgramError` with the failed output, logs
included.

What qccompute grants a calculation (its `RunContext`) reaches the program like this:

- ``threads``: a calculation that requests no threads (no ``parallel`` for xtb, nothing
  for DFTB+) runs with the grant, through ``OMP_NUM_THREADS`` and the other numerical
  libraries' variables; one that requests them (ORCA ``pal``, Psi4 ``-n``, CREST
  ``threads``) runs with its request, which the runner packs by.
- ``gpus``: a program that uses GPUs sees the granted ones in ``CUDA_VISIBLE_DEVICES``.
- ``timeout``: the program and every process it started are killed when it expires
  (kind ``timeout``).
- ``scratch``: the calculation directory.
- ``compute``: the subprograms of geomeTRIC and Sella. Called directly
  (``ctx=None``), such a program hands itself to ``qccompute.compute``.

An adapter's own configuration is its executable and environment: the variable
``QCCODEC_<PROGRAM>_EXECUTABLE`` (``QCCODEC_ORCA_EXECUTABLE``), or an adapter made with
``executable=`` and registered with ``qccompute.register_program``. Keywords stay the
program's own and pass through unchanged.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
import traceback
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qcdata import (
    CalcType,
    Data,
    ErrorInfo,
    ErrorKind,
    ExecutionInfo,
    FileData,
    FileInput,
    Frame,
    OptimizationData,
    PathData,
    ProgramError,
    ProgramInput,
    ProgramOutput,
    ProgramWarning,
    Provenance,
    get_data_type,
)

from . import __version__
from .codec import Codec
from .exceptions import CalculationError, DependencyNotFoundError, EncoderError
from .native import (
    NativeInput,
    NativeOutput,
    PythonInput,
    PythonOutput,
    SubprocessInput,
)
from .run import (
    collect_files,
    resolve_executable,
    run_python,
    run_subprocess,
    stage_files,
)

__all__ = ["CodecProgram", "THREAD_VARIABLES", "executable_digest"]

#: Thread counts of numerical libraries; a calculation's threads go into these.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_SCHEMA_NOTE = (
    "The program's own keywords, passed through unchanged; its documentation is the "
    "reference."
)


class CodecProgram:
    """A codec as a program qccompute runs: encode, run, decode.

    Args:
        codec: The codec of the program.
        name: The program's name; the codec's by default (``files`` for FileInputs).
        in_process: Whether it may share the runner's process. Codecs that call a
            Python library capture the process's output streams and change its
            directory while they run, so they say False and run in a worker process.
        gpu: Whether the program uses GPUs.
        executable: The executable to run, instead of ``QCCODEC_<NAME>_EXECUTABLE`` or
            the codec's command on ``PATH``.
        environment: Environment variables for the program, over every default.
        command: The program's command, when it is not its name (``dftb+``).
        library: For a codec that calls a Python library: its distribution name,
            whose version is the program's numerics identity.
    """

    def __init__(
        self,
        codec: Codec,
        *,
        name: str | None = None,
        in_process: bool = True,
        gpu: bool = False,
        executable: str | None = None,
        environment: Mapping[str, str] | None = None,
        command: str | None = None,
        library: str | None = None,
    ) -> None:
        """Wrap a codec."""
        self.codec = codec
        self.name: str = name or codec.program
        self.in_process = in_process
        self.gpu = gpu
        self.executable = executable
        self.environment = dict(environment or {})
        self.command = command or self.name
        self.library = library

    def __repr__(self) -> str:
        """Show the program."""
        return f"CodecProgram({self.name!r})"

    # ------------------------------------------------------------------ protocol

    def version(self) -> str:
        """Qccodec's version: the adapter's. The program's own is in provenance."""
        return __version__

    def numerics(self, spec: Any = None) -> str:
        """What produces the numbers: the sha256 of the executable that would run.

        A version string is not enough (the g-xTB build of xtb prints the version of
        upstream xtb 6.7.1), so the executable file itself identifies the program,
        hashed once per process per path. A codec that calls a Python library gives
        the library's version; FileInputs run any command, so qccodec's version
        stands in.

        Args:
            spec: The input or subprogram spec; every spec of an external program
                runs the same executable.

        Returns:
            ``sha256:<hex>``, ``<library> <version>``, or ``not installed``.
        """
        if self.library is not None:
            from .programs._python import package_version

            return f"{self.library} {package_version(self.library) or 'not installed'}"
        if self.name == "files":
            return __version__
        try:
            executable = resolve_executable(
                self.name,
                self.command,
                {**os.environ, **self.environment},
                self.executable,
            )
        except DependencyNotFoundError:
            return "not installed"
        return executable_digest(executable)

    def capabilities(self) -> dict[str, Any]:
        """The codec's calctypes; batches when the codec batches; where it may run."""
        return {
            "calctypes": sorted(c.value for c in self.codec.calctypes),
            "batches": self.codec.max_batch > 1 and self.codec._batch_key is not None,
            "in_process": self.in_process,
            "thread_safe": self.in_process,
            "gpu": self.gpu,
        }

    def settings_schema(self, calctype: CalcType) -> dict[str, Any]:
        """Any keywords: the program's own, which it validates when it runs."""
        return {
            "type": "object",
            "description": _SCHEMA_NOTE,
            "properties": {},
            "additionalProperties": True,
        }

    def record_schema(self, calctype: CalcType) -> dict[str, Any]:
        """No record: external programs report through their results."""
        return {"type": "object", "properties": {}}

    def plan(self, inp: ProgramInput | FileInput) -> dict[str, Any]:
        """Encode the input without running it: its settings and resources.

        Raises:
            ProgramError: The codec refuses the input (kind ``input``) or its
                calculation type (kind ``unsupported``).
        """
        native = self._encode(inp)
        keywords = dict(inp.keywords) if isinstance(inp, ProgramInput) else {}
        return {
            "resolved": {"settings": keywords},
            "resources": {"threads": native.threads or 1, "gpus": native.gpus},
        }

    def compute(
        self, inp: ProgramInput | FileInput, ctx: Any = None
    ) -> ProgramOutput[Any, Data]:
        """Run the program on one input.

        Args:
            inp: The input; a `qcdata.FileInput` runs its command and returns every
                file it wrote.
            ctx: The runner's grant (a ``qccompute.program.RunContext``); None for a
                direct call.

        Returns:
            The successful output, with the program's logs.

        Raises:
            ProgramError: The calculation failed; ``output`` holds the partial
                results and the logs.
        """
        if ctx is None and isinstance(inp, ProgramInput) and inp.subprograms:
            import qccompute  # a program with subprograms runs them on a runner

            return qccompute.compute(inp)
        start = time.perf_counter()
        native = self._encode(inp)
        warnings: list[ProgramWarning] = []
        grant = ctx.threads if ctx is not None else None
        requested = native.threads
        if (
            grant
            and requested
            and requested > grant
            and self.codec.cap_threads is not None
            and isinstance(inp, ProgramInput)
        ):
            # A program uses the smaller of its request and its grant.
            native = self._encode(self.codec.cap_threads(inp, grant))
            warnings.append(
                ProgramWarning(
                    code="resource_capped",
                    message=f"{self.name} asked for {requested} threads and was granted "
                    f"{grant}; it ran with {grant}.",
                )
            )
        if ctx is None:
            directory = Path(tempfile.mkdtemp(prefix="qccodec-"))
            try:
                return self._run(inp, native, ctx, directory, start, warnings)
            finally:
                shutil.rmtree(directory, ignore_errors=True)
        return self._run(inp, native, ctx, ctx.scratch, start, warnings)

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: Any = None
    ) -> list[ProgramOutput[Any, Data]]:
        """Run inputs that share a model in batches; one output per input, in order.

        Inputs the codec batches together (same batch key) go in batches of at most
        ``codec.max_batch``; a batch that fails runs its inputs one at a time, so each
        failure is reported with its own input. Failures, refused inputs included, are
        failed outputs in place.
        """
        outputs: list[ProgramOutput[Any, Data] | None] = [None] * len(inputs)
        groups: dict[Hashable, list[int]] = {}
        alone: list[int] = []
        for i, inp in enumerate(inputs):
            try:
                key = self.codec.batch_key(inp)
            except Exception:  # noqa: BLE001 - computing it alone reports it
                key = None
            if key is None:
                alone.append(i)
            else:
                groups.setdefault(key, []).append(i)
        for members in groups.values():
            size = self.codec.max_batch
            for s in range(0, len(members), size):
                chunk = members[s : s + size]
                done = self._batch([inputs[i] for i in chunk], ctx)
                if done is None:
                    alone.extend(chunk)
                else:
                    for i, out in zip(chunk, done, strict=True):
                        outputs[i] = out
        for i in sorted(alone):
            try:
                outputs[i] = self.compute(inputs[i], ctx)
            except ProgramError as e:
                outputs[i] = e.output or ProgramOutput.failure(inputs[i], e.error)
        return [o for o in outputs if o is not None]

    # ------------------------------------------------------------------ running

    def _encode(self, inp: ProgramInput | FileInput) -> NativeInput:
        """The native invocation; a refusal as a ProgramError of kind input."""
        if isinstance(inp, ProgramInput) and inp.calctype not in self.codec.calctypes:
            raise ProgramError.of(
                "unsupported",
                f"{self.name} computes "
                f"{sorted(c.value for c in self.codec.calctypes)}, "
                f"not {inp.calctype.value}.",
            )
        try:
            return self.codec.encode(inp)
        except EncoderError as e:
            raise ProgramError.of("input", str(e)) from e

    def _environment(
        self, native: NativeInput, ctx: Any
    ) -> tuple[dict[str, str], int | None]:
        """The program's environment overrides and the threads it runs with."""
        threads = native.threads
        if threads is None and ctx is not None:
            threads = ctx.threads
        env: dict[str, str] = {}
        if threads is not None:
            env.update({name: str(threads) for name in THREAD_VARIABLES})
        if native.gpus and ctx is not None and ctx.gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(ctx.gpus[: native.gpus])
        env.update(native.environment)  # the codec knows its program (CREST: 1)
        env.update(self.environment)  # the adapter's configuration wins
        return env, threads

    def _run(
        self,
        inp: ProgramInput | FileInput,
        native: NativeInput,
        ctx: Any,
        directory: Path,
        start: float,
        warnings: list[ProgramWarning],
    ) -> ProgramOutput[Any, Data]:
        """Run an encoded input in its directory and build the output."""
        failures: list[tuple[Exception, str]] = []
        results: Data = _empty(inp)
        logs: str | None = None
        executable: str | None = None
        env, threads = self._environment(native, ctx)
        timeout = ctx.timeout if ctx is not None else None

        def failed(exc: Exception, trace: str | None = None) -> None:
            failures.append((exc, trace or "".join(traceback.format_exception(exc))))

        try:
            stage_files(directory, native.files)
            out: NativeOutput
            if isinstance(native, SubprocessInput):
                command = native.command[0]
                name = inp.program if isinstance(inp, ProgramInput) else command
                executable = resolve_executable(
                    name, command, {**os.environ, **env}, self.executable
                )
                request = SubprocessInput(
                    command=(executable, *native.command[1:]),
                    files=native.files,
                    environment=native.environment,
                    threads=native.threads,
                    gpus=native.gpus,
                )
                out = run_subprocess(
                    request, directory, environment=env, timeout=timeout
                )
            else:
                out = self._run_python(native, directory, env, ctx)
            logs = out.stdout
            exit_status_only = out.failed and out.error is None
            if out.failed:
                error = out.error or RuntimeError(
                    f"{self.name} failed with exit status "
                    f"{getattr(out, 'returncode', None)}."
                )
                failed(error, out.traceback)
            try:  # always recover what the program wrote, failed or not
                results = self.codec.decode(inp, out)
            except Exception as exc:  # noqa: BLE001 - a decoding failure is a failure
                recovered = getattr(exc, "data", None)
                if isinstance(recovered, type(results)):
                    results = recovered
                failed(exc)
                if exit_status_only and isinstance(exc, CalculationError):
                    # The codec explains a nonzero exit: lead with its reason.
                    failures.insert(0, failures.pop())
            if not isinstance(inp, ProgramInput):
                results = FileData.model_validate(
                    {
                        **results.model_dump(),
                        "files": collect_files(directory, exclude=inp.files),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - staging, launching, collecting
            failed(exc)
        execution = ExecutionInfo(
            threads=threads,
            timings={"total": round(time.perf_counter() - start, 6)},
            extras={"executable": executable} if executable else {},
        )
        if not failures:
            return ProgramOutput(
                input_data=inp,
                success=True,
                results=results,
                logs=logs,
                warnings=warnings,
                execution=execution,
            )
        info = _error_info(failures[0][0], results)
        trace = "\nAdditional failure:\n".join(t for _, t in failures)
        try:
            output = ProgramOutput.failure(
                inp,
                info,
                results=results,
                traceback=trace,
                logs=logs,
                warnings=warnings,
                execution=execution,
            )
        except Exception:  # noqa: BLE001 - partial results that break the contract
            output = ProgramOutput.failure(
                inp,
                info,
                traceback=trace,
                logs=logs,
                warnings=warnings,
                execution=execution,
            )
        raise ProgramError(info, output=output) from failures[0][0]

    def _run_python(
        self, native: PythonInput, directory: Path, env: dict[str, str], ctx: Any
    ) -> PythonOutput:
        """Call a Python library, with subprograms through the runner's ctx."""

        def compute_child(
            child: ProgramInput, *, previous: Any = None
        ) -> ProgramOutput[ProgramInput, Data]:
            return ctx.compute(child, raise_exc=False)  # type: ignore[no-any-return]

        def compute_children(
            children: Sequence[ProgramInput],
        ) -> list[ProgramOutput[ProgramInput, Data]]:
            return ctx.compute(list(children), raise_exc=False)  # type: ignore[no-any-return]

        return run_python(
            native,
            directory,
            environment=env,
            compute_child=compute_child if ctx is not None else None,
            compute_children=compute_children if ctx is not None else None,
        )

    def _batch(
        self, inputs: Sequence[ProgramInput], ctx: Any
    ) -> list[ProgramOutput[Any, Data]] | None:
        """One batch, each value decoded as if its input ran alone; None if it failed."""
        start = time.perf_counter()
        try:
            request = self.codec.encode_batch(inputs)
        except Exception:  # noqa: BLE001 - computed one at a time instead
            return None
        env, threads = self._environment(request, ctx)
        directory = Path(tempfile.mkdtemp(prefix="qccodec-batch-"))
        try:
            stage_files(directory, request.files)
            native = run_python(request, directory, environment=env)
            values = native.value
            if (
                native.error is not None
                or not isinstance(values, Sequence)
                or len(values) != len(inputs)
            ):
                return None
            wall = round(time.perf_counter() - start, 6)
            outputs: list[ProgramOutput[Any, Data]] = []
            for index, (inp, value) in enumerate(zip(inputs, values, strict=True)):
                execution = ExecutionInfo(
                    threads=threads,
                    timings={"total": wall},
                    extras={"batch_size": len(inputs), "batch_index": index},
                )
                try:
                    results = self.codec.decode(
                        inp,
                        PythonOutput(
                            value=value, stdout=native.stdout, directory=directory
                        ),
                    )
                    outputs.append(
                        ProgramOutput(
                            input_data=inp,
                            success=True,
                            results=results,
                            logs=native.stdout,
                            execution=execution,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - this input's failure
                    outputs.append(
                        ProgramOutput.failure(
                            inp,
                            _error_info(exc, None),
                            traceback="".join(traceback.format_exception(exc)),
                            logs=native.stdout,
                            execution=execution,
                        )
                    )
            return outputs
        finally:
            shutil.rmtree(directory, ignore_errors=True)


_DIGESTS: dict[str, str] = {}


def executable_digest(executable: str) -> str:
    """The sha256 of an executable file (symbolic links followed), once per path.

    Args:
        executable: The executable's path.

    Returns:
        ``sha256:<hex>``, or ``not installed`` when the file cannot be read.
    """
    if executable not in _DIGESTS:
        try:
            with open(os.path.realpath(executable), "rb") as f:
                digest = "sha256:" + hashlib.file_digest(f, "sha256").hexdigest()
        except OSError:
            return "not installed"
        _DIGESTS[executable] = digest
    return _DIGESTS[executable]


def _empty(inp: ProgramInput | FileInput) -> Data:
    """Empty results of the input's type, before anything ran."""
    data_type = (
        get_data_type(inp.calctype) if isinstance(inp, ProgramInput) else FileData
    )
    return data_type(provenance=Provenance(program=inp.program))


def _error_info(exc: Exception, results: Data | None) -> ErrorInfo:
    """The typed error of a failed run, from its primary failure.

    Codecs do not classify a program's failures, so what the program reported, an
    unexplained exit and a decoding failure are all kind ``internal``; a missing
    executable is ``environment`` and a deadline ``timeout``. A trajectory whose last
    step failed failed because of that subprogram: kind ``program``, with its error.
    """
    if isinstance(exc, TimeoutError):
        return ErrorInfo(kind=ErrorKind.timeout, message=str(exc))
    if isinstance(exc, DependencyNotFoundError):
        return ErrorInfo(kind=ErrorKind.environment, message=str(exc))
    if isinstance(exc, EncoderError):
        return ErrorInfo(kind=ErrorKind.input, message=str(exc))
    frames: list[Frame] = []
    if isinstance(results, OptimizationData):
        frames = list(results.frames)
    elif isinstance(results, PathData):
        frames = list(results.frames)
    cause = frames[-1].error if frames and not frames[-1].success else None
    if cause is not None:
        program = frames[-1].results.provenance.program
        return ErrorInfo(
            kind=ErrorKind.program,
            message=f"subprogram {program} failed: {cause.message}",
            program=program,
            cause=cause,
        )
    named = not isinstance(exc, CalculationError | DependencyNotFoundError)
    message = f"{type(exc).__name__}: {exc}" if named else str(exc)
    return ErrorInfo(kind=ErrorKind.internal, message=message.strip())
