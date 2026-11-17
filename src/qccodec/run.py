"""Run native invocations: a subprocess or a Python call in a calculation directory.

What an adapter (`qccodec.program.CodecProgram`) does between encoding and decoding:
stage the input files in a directory, run the command or the callable while capturing
its output, kill it at a deadline, and collect the files it wrote. Nothing here knows
any program's conventions.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from time import monotonic
from typing import TextIO

from .exceptions import DependencyNotFoundError
from .native import (
    ComputeChild,
    ComputeChildren,
    PythonInput,
    PythonOutput,
    SubprocessInput,
    SubprocessOutput,
)

_python_lock = threading.RLock()
# How many Python calls are running in this thread's stack; nested lists run serially.
_python_depth = threading.local()
# Programs running right now, so an interrupted list can kill every process tree.
_running: set[subprocess.Popen[str]] = set()
_running_lock = threading.Lock()


def in_python_call() -> bool:
    """Whether the current thread is inside a Python calculation (a child callback)."""
    return getattr(_python_depth, "value", 0) > 0


def kill_running() -> None:
    """Kill every program this process started that is still running, with its tree."""
    with _running_lock:
        procs = list(_running)
    for proc in procs:
        if proc.poll() is None:
            _kill_tree(proc)


def _flush_native_output() -> None:
    """Flush C stdio while the intended file descriptors are still installed."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError):
        pass


def run_subprocess(
    request: SubprocessInput,
    directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    update_func: Callable[[str, str], None] | None = None,
    update_interval: float | None = None,
    timeout: float | None = None,
) -> SubprocessOutput:
    """Execute a native command and monitor its merged stdout/stderr.

    Run in the calculation directory without changing the parent process's working
    directory or environment. Standard error is redirected into standard output,
    and stdin is closed. Output is read until EOF, including any final text without
    a trailing newline.

    Args:
        request: The prepared command and invocation environment defaults. Input
            files must already have been written to the calculation directory.
        directory: The working directory for the child process.
        environment: Overrides applied after the inherited environment and request
            defaults. These affect only the child process.
        update_func: Optional callback accepting two strings: all captured output
            and the new output since the previous callback. For console output,
            use ``lambda all_logs, new_logs: print(new_logs, end="")``. Any remaining
            output is delivered after the child exits, including on a nonzero exit.
            No callback is made if the child produces no output.
        update_interval: Minimum seconds between live callbacks; defaults to 0.5.
            Updates are line-based, so an unterminated line waits until a newline
            or EOF. This interval is not a heartbeat. The final callback is not
            delayed by the interval.
        timeout: Seconds the program may run. When exceeded, the program and every
            process it started are killed, and ``error`` holds a
            ``TimeoutError``; output produced so far is kept.

    Returns:
        Native execution observations with merged output in ``stdout``, the working directory, and the return code. Launch, reading,
        or callback exceptions are recorded in ``error`` and ``traceback``, with
        output read so far retained. A nonzero exit is returned for the caller to
        interpret, allowing decoding of partial scientific results.

    On POSIX the program runs in its own session, so a timeout or an interrupted
    caller kills the whole process tree (for example the xtb processes CREST
    starts), not just the immediate child. Scientific decoding and scratch-directory
    cleanup belong to the caller.
    """
    output = SubprocessOutput(directory=directory)
    proc = None
    timer: threading.Timer | None = None
    timed_out = threading.Event()
    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            request.command,
            cwd=directory,
            env={**os.environ, **request.environment, **(environment or {})},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=_GROUP_KILL,
        )
        with _running_lock:
            _running.add(proc)
        if timeout is not None:
            running = proc

            def expire() -> None:
                """Kill the program tree once the deadline passes."""
                timed_out.set()
                _kill_tree(running)

            timer = threading.Timer(timeout, expire)
            timer.daemon = True
            timer.start()
        assert proc.stdout is not None
        last_update = monotonic()
        reported = 0
        interval = 0.5 if update_interval is None else update_interval
        for line in proc.stdout:
            lines.append(line)
            if update_func and monotonic() - last_update >= interval:
                update_func("".join(lines), "".join(lines[reported:]))
                reported = len(lines)
                last_update = monotonic()
        output.returncode = proc.wait()
        if update_func and len(lines) > reported:
            update_func("".join(lines), "".join(lines[reported:]))
        if timed_out.is_set():
            output.error = TimeoutError(
                f"{request.command[0]} exceeded the {timeout:g} s timeout and was killed."
            )
    except Exception as exc:
        output.error = exc
        output.traceback = traceback.format_exc()
        if (
            proc is None
            and isinstance(exc, FileNotFoundError)
            and exc.filename == request.command[0]
        ):
            # Popen identifies a missing executable separately from a missing cwd.
            output.error = DependencyNotFoundError(str(exc))
            output.error.__cause__ = exc
    finally:
        if timer is not None:
            timer.cancel()
        if proc is not None:
            if proc.poll() is None:
                _kill_tree(proc)
            output.returncode = proc.wait()
            with _running_lock:
                _running.discard(proc)
            if proc.stdout is not None:
                proc.stdout.close()
        output.stdout = "".join(lines)
    return output


# A new session gives the program its own process group, so the whole tree can be
# killed together. Windows has no process groups in this sense.
_GROUP_KILL = hasattr(os, "killpg")


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill a program and, on POSIX, every process in its session."""
    try:
        if _GROUP_KILL:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


class _CallStream:
    """``sys.stdout`` or ``sys.stderr`` during a Python call.

    Writes go to the capture file while the call runs. Libraries often keep a
    reference to the stream they saw at import time (a logging handler bound to
    ``sys.stdout``); once the call ends, this stream forwards to whatever
    ``sys.stdout``/``sys.stderr`` is current instead of to the closed capture file,
    so such handlers keep working and are captured again by later calls.
    """

    def __init__(self, capture: TextIO, name: str) -> None:
        self._capture: TextIO | None = capture
        self._name = name

    def _target(self) -> TextIO:
        if self._capture is not None:
            return self._capture
        current: TextIO = getattr(sys, self._name)
        # A stale proxy can be installed again by code that saved it; never loop.
        return current if current is not self else getattr(sys, f"__{self._name}__")

    def release(self) -> None:
        """Stop writing to the capture file and follow the live stream from now on."""
        self._capture = None

    def write(self, text: str) -> int:
        """Write to the capture file, or to the live stream after the call."""
        return self._target().write(text)

    def flush(self) -> None:
        """Flush the current target."""
        self._target().flush()

    def __getattr__(self, name: str) -> object:
        """Delegate everything else (``fileno``, ``encoding``, ...) to the target."""
        return getattr(self._target(), name)


@contextmanager
def _python_context(
    directory: Path,
    environment: Mapping[str, str],
    loggers: tuple[str, ...],
    capture: TextIO,
) -> Iterator[None]:
    """Capture Python and native output while temporarily preparing process state.

    Args:
        directory: Working directory for the invocation.
        environment: Environment overrides to restore afterward.
        loggers: Named loggers whose handlers temporarily write into capture.
        capture: Caller-owned file receiving both stdout and stderr. It remains
            open after this context exits so the caller can read the completed log.

    Yields:
        Control with Python streams, native descriptors, and loggers redirected.

    The reentrant lock permits synchronous nested calculations. It serializes
    Python runners within this process, but does not isolate unrelated threads.
    Each restoration is registered separately so cleanup continues if one fails.
    """
    with _python_lock, ExitStack() as restore:
        _python_depth.value = getattr(_python_depth, "value", 0) + 1
        restore.callback(setattr, _python_depth, "value", _python_depth.value - 1)
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        _flush_native_output()
        restore.callback(os.chdir, Path.cwd())
        os.chdir(directory)
        for name, value in environment.items():
            if name in os.environ:
                restore.callback(os.environ.__setitem__, name, os.environ[name])
            else:
                restore.callback(os.environ.pop, name, None)
            os.environ[name] = value
        for fd in (1, 2):
            saved = os.dup(fd)
            restore.callback(os.close, saved)
            restore.callback(os.dup2, saved, fd)
            os.dup2(capture.fileno(), fd)
        for name in loggers:
            logger = logging.getLogger(name)
            restore.callback(setattr, logger, "handlers", logger.handlers[:])
            restore.callback(logger.setLevel, logger.level)
            restore.callback(setattr, logger, "propagate", logger.propagate)
            logger.handlers = [logging.StreamHandler(capture)]
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
        streams = [_CallStream(capture, "stdout"), _CallStream(capture, "stderr")]
        for stream in streams:
            restore.callback(stream.release)
        restore.enter_context(redirect_stdout(streams[0]))  # type: ignore[arg-type]
        restore.enter_context(redirect_stderr(streams[1]))  # type: ignore[arg-type]
        try:
            yield
        finally:
            _flush_native_output()
            capture.flush()


def run_python(
    request: PythonInput,
    directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    compute_child: ComputeChild | None = None,
    compute_children: ComputeChildren | None = None,
    update_func: Callable[[str, str], None] | None = None,
) -> PythonOutput:
    """Invoke a callable and retain its return value, merged logs, or failure.

    Args:
        request: Prepared callable, arguments, and capture requirements.
        directory: Calculation directory, with input files already staged.
        environment: Overrides applied after the invocation's environment defaults.
        compute_child: Synchronous calculation callback injected into the keyword
            named by request.child_callback, when requested by the integration.
        compute_children: Callback for several independent children at once,
            injected into the keyword named by request.children_callback.
        update_func: Optional callback receiving the completed log twice, as both
            all output and new output. Called once for nonempty logs after capture
            is restored; Python execution does not provide live updates.

    Returns:
        Native observations with the callable's value, merged output in stdout,
        and exceptions. Exceptions are recorded with their traceback and logs;
        a reporting or cleanup error does not replace an earlier invocation error.

    Native writes to descriptors 1 and 2 are captured alongside Python output.
    C stdio is flushed before restoration; bindings remain responsible for other
    runtime-specific buffers. Use separate processes for concurrent calculations.
    """
    output = PythonOutput(directory=directory)
    try:
        overrides = {**request.environment, **(environment or {})}
        with tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        ) as capture:
            try:
                with _python_context(directory, overrides, request.loggers, capture):
                    try:
                        kwargs = dict(request.kwargs)
                        if request.child_callback:
                            if (
                                request.child_callback in kwargs
                                or compute_child is None
                            ):
                                raise ValueError(
                                    "Missing or conflicting child calculation callback."
                                )
                            kwargs[request.child_callback] = compute_child
                        if request.children_callback and compute_children is not None:
                            kwargs[request.children_callback] = compute_children
                        output.value = request.function(*request.args, **kwargs)
                    except Exception as exc:
                        output.error = exc
                        output.traceback = traceback.format_exc()
            finally:
                capture.seek(0)
                output.stdout = capture.read()
        if update_func and output.stdout:
            update_func(output.stdout, output.stdout)
    except Exception as exc:
        if output.error is None:
            output.error = exc
            output.traceback = traceback.format_exc()
    return output


def executable_variable(program: str) -> str:
    """Name the environment variable that overrides a program's executable.

    ``orca`` becomes ``QCCODEC_ORCA_EXECUTABLE``; characters other than letters
    and digits become underscores.
    """
    return f"QCCODEC_{re.sub(r'[^A-Z0-9]', '_', program.upper())}_EXECUTABLE"


def resolve_executable(
    program: str,
    command: str,
    environment: Mapping[str, str],
    executable: str | None = None,
) -> str:
    """Return the absolute path of the executable to run for ``program``.

    The first of these wins: ``executable`` (the adapter's configuration), the
    environment variable named by ``executable_variable(program)``, then ``command``
    (the codec's first argument). A value without a path separator is looked up on the ``PATH`` in
    ``environment``; a relative path is taken relative to the current directory,
    not the calculation directory.

    Args:
        program: The program name from the input, which keys the overrides.
        command: The executable the codec asked for.
        environment: The environment the program will run with.
        executable: The adapter's configured executable, if any.

    Raises:
        DependencyNotFoundError: The chosen executable does not exist or is not
            executable.
    """
    variable = executable_variable(program)
    if executable:
        value, source = executable, "the configured executable"
    elif environment.get(variable):
        value, source = environment[variable], variable
    else:
        value, source = command, None
    found = shutil.which(os.path.expanduser(value), path=environment.get("PATH"))
    if found is not None:
        return os.path.abspath(found)
    if source is not None:
        message = f"{source} is {value!r}, which is not an executable file."
    else:
        message = (
            f"Program not found: {value!r} is not on PATH. Install it, or set "
            f"{variable} to its full path."
        )
    raise DependencyNotFoundError(message)


@contextmanager
def workspace(
    directory: str | Path | None = None, remove: bool = True
) -> Iterator[Path]:
    """Keep calculation files alive through decoding and optionally remove them afterward."""
    path = (
        Path(directory).resolve()
        if directory is not None
        else Path(tempfile.mkdtemp()).resolve()
    )
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        if remove:
            shutil.rmtree(path)


def safe_path(directory: Path, name: str) -> Path:
    """Reject file names that would escape the calculation workspace."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"File name must stay within the calculation directory: {name!r}"
        )
    path = directory / relative
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"File path escapes the calculation directory: {name!r}")
    return path


def stage_files(directory: Path, files: Mapping[str, str | bytes]) -> None:
    """Write native inputs while keeping relative paths within the workspace."""
    for name, value in files.items():
        path = safe_path(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value)


def collect_files(
    directory: Path,
    *,
    exclude: Collection[str] = (),
    patterns: tuple[str, ...] | None = None,
    binary: bool = False,
) -> dict[str, str | bytes]:
    """Read selected regular artifacts before workspace cleanup."""
    files: dict[str, str | bytes] = {}
    paths = (
        directory.rglob("*")
        if patterns is None
        else (path for pattern in patterns for path in directory.glob(pattern))
    )
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        name = path.relative_to(directory).as_posix()
        if name in exclude or not path.resolve().is_relative_to(directory.resolve()):
            continue
        contents = path.read_bytes()
        if binary:
            files[name] = contents
            continue
        try:
            files[name] = contents.decode("utf-8")
        except UnicodeDecodeError:
            files[name] = contents
    return files
