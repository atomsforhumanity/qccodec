"""Run native invocations: a subprocess or a Python call in a calculation directory.

What an adapter (`qccodec.program.CodecProgram`) does between encoding and decoding:
stage the input files in a directory, run the command or the callable while capturing
its output, kill it at a deadline, and collect the files it wrote. Nothing here knows
any program's conventions.
"""

from __future__ import annotations

import contextvars
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
    interrupt: Callable[[], Exception | None] | None = None,
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
        interrupt: Asked about once a second (`INTERRUPT_POLL`) while the program
            runs: an exception when it must end now (its calculation was cancelled,
            or asked to stop), else None. The program and every process it started
            are then killed, and ``error`` holds that exception; output produced so
            far is kept.

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
    finished = threading.Event()
    ended: list[Exception] = []  # why the watcher killed the program
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
        if timeout is not None or interrupt is not None:
            threading.Thread(
                target=_watch,
                args=(proc, finished, ended, request.command[0], timeout, interrupt),
                daemon=True,
            ).start()
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
        finished.set()
        if update_func and len(lines) > reported:
            update_func("".join(lines), "".join(lines[reported:]))
        if ended:
            output.error = ended[0]
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
        finished.set()
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

#: Seconds between a running program's checks of ``interrupt`` (`run_subprocess`).
INTERRUPT_POLL = 1.0


def _watch(
    proc: subprocess.Popen[str],
    finished: threading.Event,
    ended: list[Exception],
    command: str,
    timeout: float | None,
    interrupt: Callable[[], Exception | None] | None,
) -> None:
    """Kill a running program's tree at its deadline, or when ``interrupt`` says so.

    Wakes at the deadline and at least every `INTERRUPT_POLL` seconds until the
    program has ended (``finished``); the reason it killed the program goes in
    ``ended``.
    """
    deadline = None if timeout is None else monotonic() + timeout
    while True:
        wait = INTERRUPT_POLL
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - monotonic()))
        if finished.wait(wait):
            return
        why: Exception | None = None
        if deadline is not None and monotonic() >= deadline:
            why = TimeoutError(
                f"{command} exceeded the {timeout:g} s timeout and was killed."
            )
        elif interrupt is not None:
            try:
                why = interrupt()
            except Exception:  # noqa: BLE001 - a broken check never ends a program
                why = None
        if why is not None:
            ended.append(why)
            _kill_tree(proc)
            return


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
    device: str | None = None,
    threads: int | None = None,
    update_func: Callable[[str, str], None] | None = None,
) -> PythonOutput:
    """Invoke a callable and retain its return value, merged logs, or failure.

    Args:
        request: Prepared callable, arguments, and capture requirements.
        directory: Calculation directory, with input files already staged.
        environment: Overrides applied after the invocation's environment defaults.
        compute_child: Synchronous calculation callback injected into the keyword
            named by request.child_callback, when requested by the integration. The
            calculation directory goes into ``request.directory_keyword``, if named.
        compute_children: Callback for several independent children at once,
            injected into the keyword named by request.children_callback.
        device: The device granted, injected into the keyword named by
            ``request.device_keyword``; None leaves it out.
        threads: CPU threads of this call, for the numerical libraries already
            loaded in this process (`thread_limits`); a library the call loads reads
            them from ``environment`` (``OMP_NUM_THREADS``). None leaves them alone.
            A call that runs subprograms (the request has a child callback) holds
            them at one thread whatever this says: its threads are its children's,
            and an idle OpenBLAS thread spins for about 0.1 s after each call, beside
            a child that uses them all.
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
    if request.child_callback or request.children_callback:
        threads = 1  # a parent's own pools: its threads are its children's
    try:
        overrides = {**request.environment, **(environment or {})}
        with tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        ) as capture:
            try:
                with (
                    _python_context(directory, overrides, request.loggers, capture),
                    thread_limits(threads),
                ):
                    try:
                        kwargs = _call_kwargs(
                            request, directory, compute_child, compute_children, device
                        )
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


@contextmanager
def thread_limits(threads: int | None) -> Iterator[None]:
    """Run one call with ``threads`` in the numerical libraries this process has loaded.

    A worker process serves calls with different grants, but a numerical library
    reads ``OMP_NUM_THREADS`` once, when it loads. So for each call, PyTorch's
    intra-op threads are set (``torch.set_num_threads``, when PyTorch is loaded) and
    the BLAS and OpenMP runtimes already loaded are limited
    (``threadpoolctl.threadpool_limits``, undone afterwards). Process-wide: only for
    calls that own the process (`run_python`).

    Args:
        threads: The call's CPU threads; None changes nothing.

    Yields:
        Control with the limits in place.
    """
    if threads is None:
        yield
        return
    from threadpoolctl import threadpool_limits

    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        yield


def _call_kwargs(
    request: PythonInput,
    directory: Path,
    compute_child: ComputeChild | None,
    compute_children: ComputeChildren | None,
    device: str | None = None,
) -> dict[str, object]:
    """The keyword arguments of a call: the request's, and what the runner injects.

    Raises:
        ValueError: The function needs a child callback and none was given, or one
            of the injected keywords is already among its arguments.
    """
    kwargs = dict(request.kwargs)
    injected = (
        request.child_callback,
        request.children_callback,
        request.directory_keyword,
        request.device_keyword,
    )
    if any(name in kwargs for name in injected if name):
        raise ValueError("An injected keyword is already among the arguments.")
    if request.child_callback:
        if compute_child is None:
            raise ValueError("Missing child calculation callback.")
        kwargs[request.child_callback] = compute_child
    if request.children_callback and compute_children is not None:
        kwargs[request.children_callback] = compute_children
    if request.directory_keyword:
        kwargs[request.directory_keyword] = directory.resolve()
    if request.device_keyword and device is not None:
        kwargs[request.device_keyword] = device
    return kwargs


# The log of the in-process call running in this context, if any.
_log_capture: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "qccodec_log_capture", default=None
)
_routed: set[str] = set()
_routed_lock = threading.Lock()


class _RouteToCall(logging.Filter):
    """Send a logger's records to the in-process call of this context, if any.

    Installed once per logger and left in place: outside a call it passes every record
    through untouched; inside one it keeps the record in that call's log and stops it,
    so a program's log reaches its output, not the caller's handlers. Each thread and
    context has its own call, so calls in several threads keep their logs apart.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        captured = _log_capture.get()
        if captured is None:
            return True
        captured.append(record.getMessage())
        return False


def _route(names: tuple[str, ...]) -> None:
    """Install the routing filter on each named logger, once per process."""
    with _routed_lock:
        for name in names:
            if name not in _routed:
                logging.getLogger(name).addFilter(_RouteToCall())
                _routed.add(name)


def call_python(
    request: PythonInput,
    directory: Path,
    *,
    compute_child: ComputeChild | None = None,
    compute_children: ComputeChildren | None = None,
    device: str | None = None,
) -> PythonOutput:
    """Call a function in this process without touching process-wide state.

    For functions that write their files in the directory injected through
    ``request.directory_keyword`` and report through Python loggers: the working
    directory, environment, standard streams and file descriptors stay as they are,
    and no lock is taken, so calls in several threads run side by side. The records
    of ``request.loggers`` emitted in this context become the log, joined as logged
    (the library ends its own lines, as geomeTRIC does); anything the function
    prints goes to the caller's streams. ``request.environment`` is not applied.

    Args:
        request: The callable, its arguments, and the loggers to capture.
        directory: The calculation directory, with input files already staged.
        compute_child: Injected into the keyword named by ``request.child_callback``.
        compute_children: Injected into the keyword named by
            ``request.children_callback``.
        device: Injected into the keyword named by ``request.device_keyword``;
            None leaves it out.

    Returns:
        The callable's value, the captured log in ``stdout``, or the exception with
        its traceback.
    """
    output = PythonOutput(directory=directory)
    captured: list[str] = []
    _route(request.loggers)
    token = _log_capture.set(captured)
    try:
        kwargs = _call_kwargs(
            request, directory, compute_child, compute_children, device
        )
        output.value = request.function(*request.args, **kwargs)
    except Exception as exc:
        output.error = exc
        output.traceback = traceback.format_exc()
    finally:
        _log_capture.reset(token)
        output.stdout = "".join(captured)
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
