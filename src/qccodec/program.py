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

- ``threads``: the plan asks for the input's request, its ``threads`` keyword (the
  name every program shares) or the program's own setting (ORCA ``pal``, Psi4 ``-n``,
  xtb ``parallel``; giving both is an error, code ``no_effect``), else the adapter's
  default: 1, or None (all the runner has) for a program that scales with threads
  (Psi4, CREST, and geomeTRIC or Sella whose subprogram does). The runner grants at
  most that. A program with its own thread setting runs with the grant written into
  it (the codec's ``cap_threads``), the others with it in ``OMP_NUM_THREADS`` and the
  other numerical libraries' variables; a request larger than the grant is lowered
  with a ``resource_capped`` warning.
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

import dataclasses
import hashlib
import os
import shutil
import subprocess
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
from .codec import STDOUT, VERSION_TARGET, Codec
from .exceptions import (
    BaseError,
    CalculationError,
    DependencyNotFoundError,
    EncoderError,
)
from .native import (
    NativeInput,
    NativeOutput,
    PythonInput,
    PythonOutput,
    SubprocessInput,
)
from .run import (
    call_python,
    collect_files,
    resolve_executable,
    run_python,
    run_subprocess,
    stage_files,
)

__all__ = [
    "CodecProgram",
    "THREAD_VARIABLES",
    "executable_digest",
    "executable_version",
]

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
        threads: The threads the program plans when its input requests none: 1, or
            None for all the runner grants, for a program that scales with threads.
            The codec writes the grant into the input with ``cap_threads``.
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
        threads: int | None = 1,
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
        self.threads = threads

    def __repr__(self) -> str:
        """Show the program."""
        return f"CodecProgram({self.name!r})"

    # ------------------------------------------------------------------ protocol

    def version(self) -> str:
        """The program's version: the library's, or what its executable reports.

        An executable is asked once per process (``<executable> --version``, read
        with the codec's version parser); a Python library's version is its
        distribution's. FileInputs run any command, so qccodec's version stands in.

        Returns:
            The version as the program prints it (``6.7.1``, ``1.11``, ``0.7.0``),
            ``unknown`` when the executable's answer names none, or ``not
            installed``.
        """
        if self.library is not None:
            from .programs._python import package_version

            return package_version(self.library) or "not installed"
        if self.name == "files":
            return __version__
        executable = self._executable()
        if executable is None:
            return "not installed"
        return executable_version(executable, self.codec)

    def numerics(self, spec: Any = None) -> str:
        """What produces the numbers: the program's version and its executable's sha256.

        A version is not enough for an executable (the g-xTB build of xtb prints the
        version of upstream xtb 6.7.1), so the executable file itself identifies the
        program, hashed once per process per path. The file is not enough either when
        it is a launcher (Psi4's ``psi4`` is a Python script that stays the same
        while the library behind it changes), so the version comes first. A codec
        that calls a Python library gives the library's version, as ``version()``.

        Args:
            spec: The input or subprogram spec; every spec of an external program
                runs the same executable.

        Returns:
            ``<version> sha256:<64 hex digits>`` for an executable (``6.7.1
            sha256:cab4...``), the version for a library (``0.7.0``) or for FileInputs
            (qccodec's), or ``not installed``.
        """
        if self.library is not None or self.name == "files":
            return self.version()
        executable = self._executable()
        if executable is None:
            return "not installed"
        digest = executable_digest(executable)
        if digest == "not installed":
            return digest
        return f"{executable_version(executable, self.codec)} {digest}"

    def _executable(self) -> str | None:
        """The executable that would run, or None when there is none."""
        try:
            return resolve_executable(
                self.name,
                self.command,
                {**os.environ, **self.environment},
                self.executable,
            )
        except DependencyNotFoundError:
            return None

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

        The threads are the input's request (its ``threads`` keyword, or the
        program's own setting such as ORCA's ``pal``), else the adapter's
        ``threads`` (None: all the runner has).

        Raises:
            ProgramError: The codec refuses the input (kind ``input``, with a code
                such as ``unknown_key``) or its calculation type (kind
                ``unsupported``).
        """
        native, asked = self._encode_requested(inp)
        keywords = dict(inp.keywords) if isinstance(inp, ProgramInput) else {}
        return {
            "resolved": {"settings": keywords},
            "resources": {
                "threads": asked or native.threads or self.threads,
                "gpus": native.gpus,
            },
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
        native, asked = self._encode_requested(inp)
        warnings: list[ProgramWarning] = []
        requested = asked or native.threads
        grant = ctx.threads if ctx is not None else None
        use = asked  # called directly: the threads asked for, if any
        if grant:
            if requested and requested > grant:
                use = grant  # a program uses the smaller of its request and its grant
                warnings.append(
                    ProgramWarning(
                        code="resource_capped",
                        message=f"{self.name} asked for {requested} threads and was "
                        f"granted {grant}; it ran with {grant}.",
                    )
                )
            elif not requested and self.threads is None:
                use = grant  # a program that scales with threads uses all it got
        if use is not None and use != native.threads:
            native = self._with_threads(inp, native, use)
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
                # A thread request is honoured one input at a time.
                threads = isinstance(inp, ProgramInput) and "threads" in inp.keywords
                key = None if threads else self.codec.batch_key(inp)
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
            raise ProgramError.of(e.kind, str(e), code=e.code) from e

    def _encode_requested(
        self, inp: ProgramInput | FileInput
    ) -> tuple[NativeInput, int | None]:
        """Encode an input apart from its ``threads`` keyword; return what it asks.

        ``threads`` is the request every program shares; the codec sees the input
        without it, and the threads reach the program through its own setting
        (`_with_threads`).

        Raises:
            ProgramError: The codec refuses the input; ``threads`` is not a positive
                integer (kind ``input``); or the input also sets the program's own
                thread count (``pal``, ``-n``, ``parallel``: code ``no_effect``).
        """
        if not isinstance(inp, ProgramInput) or "threads" not in inp.keywords:
            return self._encode(inp), None
        asked = inp.keywords["threads"]
        if isinstance(asked, bool) or not isinstance(asked, int) or asked < 1:
            raise ProgramError.of(
                "input", f"threads is a positive integer, not {asked!r}."
            )
        native = self._encode(self._without_threads(inp))
        if native.threads is not None:
            raise ProgramError.of(
                "input",
                f"threads={asked} and the program's own thread count "
                f"({native.threads}) are both given; keep one.",
                code="no_effect",
            )
        return native, asked

    @staticmethod
    def _without_threads(inp: ProgramInput) -> ProgramInput:
        """The input without its ``threads`` keyword."""
        keywords = {k: v for k, v in inp.keywords.items() if k != "threads"}
        return inp.model_copy(update={"keywords": keywords})

    def _with_threads(
        self, inp: ProgramInput | FileInput, native: NativeInput, threads: int
    ) -> NativeInput:
        """The invocation running with ``threads``.

        Through the program's own setting (``cap_threads``: ORCA ``pal``, Psi4
        ``-n``), else ``OMP_NUM_THREADS`` and the other libraries' variables.
        """
        cap = self.codec.cap_threads
        if cap is not None and isinstance(inp, ProgramInput):
            return self._encode(cap(self._without_threads(inp), threads))
        return dataclasses.replace(native, threads=threads)

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
                if (
                    out.failed
                    and isinstance(exc, CalculationError)
                    and not isinstance(
                        out.error, TimeoutError | DependencyNotFoundError
                    )
                ):
                    # The codec explains the failure (an exit status, a library's
                    # exception): lead with its reason, which carries the kind.
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
        """Call a Python library, with subprograms through the runner's ctx.

        A function that takes its directory (``directory_keyword``) is called as is
        (`qccodec.run.call_python`); any other runs with the process's working
        directory, environment and output redirected (`qccodec.run.run_python`),
        which is why such codecs run in a worker process.
        """

        def compute_child(
            child: ProgramInput, *, previous: Any = None
        ) -> ProgramOutput[ProgramInput, Data]:
            return ctx.compute(child, raise_exc=False)  # type: ignore[no-any-return]

        def compute_children(
            children: Sequence[ProgramInput],
        ) -> list[ProgramOutput[ProgramInput, Data]]:
            return ctx.compute(list(children), raise_exc=False)  # type: ignore[no-any-return]

        child = compute_child if ctx is not None else None
        children = compute_children if ctx is not None else None
        if native.directory_keyword is not None:
            # Written to run in the caller's process: nothing process-wide changes.
            return call_python(
                native, directory, compute_child=child, compute_children=children
            )
        return run_python(
            native,
            directory,
            environment=env,
            compute_child=child,
            compute_children=children,
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
_VERSIONS: dict[str, str] = {}


def executable_version(executable: str, codec: Codec) -> str:
    """Ask an executable for its version (``--version``), once per path per process.

    The answer is read with the codec's version parser when it reads stdout (ORCA,
    xtb, CREST, DFTB+ and TeraChem print their banner), else it is the first line of
    the answer (Psi4 prints ``1.11``).

    Args:
        executable: The executable's path.
        codec: The codec of the program.

    Returns:
        The version, or ``unknown`` when the answer names none.
    """
    if executable not in _VERSIONS:
        try:
            with tempfile.TemporaryDirectory(prefix="qccodec-version-") as directory:
                answer = subprocess.run(
                    [executable, "--version"],
                    cwd=directory,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=60,
                )
            text = answer.stdout + answer.stderr
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        parsers = [
            spec.parser
            for spec in codec.parsers
            if spec.target == VERSION_TARGET and spec.file == STDOUT
        ]
        try:
            found = parsers[0](text) if parsers else text.strip().splitlines()[0]
        except (BaseError, IndexError):
            found = ""
        _VERSIONS[executable] = str(found).strip() or "unknown"
    return _VERSIONS[executable]


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

    A missing executable is ``environment`` and a deadline ``timeout``. A trajectory
    whose last step failed failed because of that subprogram: kind ``program``, with
    its error. A failure the program reported has the kind and code its codec gives
    it (`Codec` ``errors``); an unexplained exit, a failure no rule classifies and
    unreadable output are kind ``internal``.
    """
    if isinstance(exc, TimeoutError):
        return ErrorInfo(kind=ErrorKind.timeout, message=str(exc))
    if isinstance(exc, DependencyNotFoundError):
        return ErrorInfo(kind=ErrorKind.environment, message=str(exc))
    if isinstance(exc, EncoderError):
        return ErrorInfo(kind=ErrorKind(exc.kind), message=str(exc), code=exc.code)
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
    if isinstance(exc, CalculationError):
        return ErrorInfo(
            kind=ErrorKind(exc.kind), message=str(exc).strip(), code=exc.code
        )
    message = f"{type(exc).__name__}: {exc}"
    return ErrorInfo(kind=ErrorKind.internal, message=message.strip())
