"""Test helpers for codecs: fixture cases, a case runner, and a codec contract check.

A *case* is a directory holding one program run and what a codec must make of it::

    tests/data/<program>/<case>/
        input.json      ProgramInput used for the run (program, calctype, structure, ...)
        expected.json   the qcdata *Data object the codec must produce
        stdout          captured program output (optional)
        <files>         output files at their real relative paths, e.g. scr.geom/optim.xyz
        case.json       optional settings: {"options": {...}, "failed": true,
                        "raises": "MatchNotFoundError", "directory": true}
        encoded/        optional: command.txt (one argument per line) and the native
                        input files codec.encode(input) must generate

``cases(root)`` discovers every such directory below ``root``; ``run_case`` checks
encoding, the full decode, and each registered parser on its own file against the
matching slice of ``expected.json``; ``check_codec`` verifies the structural contract
every codec must satisfy, with or without fixtures.

Typical test module::

    import pytest
    from qccodec import get_codec
    from qccodec.testing import cases, run_case

    @pytest.mark.parametrize("case", cases("tests/data"), ids=str)
    def test_case(case):
        run_case(get_codec(case.input_data.program), case)
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from qcdata import CalcType, ProgramInput, StructuredData, get_data_type

from . import exceptions
from .codec import DIRECTORY, STDOUT, VERSION_TARGET, Codec, Context, ParserSpec
from .exceptions import MatchNotFoundError, ParserError
from .native import PythonInput, SubprocessInput

RESERVED = {"input.json", "expected.json", "case.json", "stdout", "encoded"}
# Runtime metadata that legitimately differs between the run that produced a fixture
# and the test that replays it.
IGNORED_KEYS = {"execution", "qcdata_version"}
# qcdata stamps its version into generated files (e.g. xyz comment lines).
_QCDATA_VERSION = re.compile(r"qcdata_version=\S+")


@dataclass(frozen=True)
class Case:
    """One fixture directory, loaded.

    Attributes:
        path: The case directory; output files live here.
        name: ``<program>/<case>`` for test ids.
        input_data: The input that produced the run.
        expected: The data the codec must produce, or the partial data attached to
            the expected exception. ``None`` when no ``expected.json`` exists.
        stdout: Captured output, if the case has a ``stdout`` file.
        directory: ``path`` when output files should be visible to parsers.
        options: Decode options passed to the codec.
        failed: Decode in recovery mode.
        raises: Name of the exception class the decode must raise, if any.
        encoded: The ``encoded/`` directory with expected native inputs, if present.
    """

    path: Path
    name: str
    input_data: ProgramInput
    expected: StructuredData | None
    stdout: str | None
    directory: Path | None
    options: dict[str, Any] = field(default_factory=dict)
    failed: bool = False
    raises: str | None = None
    encoded: Path | None = None

    def __str__(self) -> str:
        """Use the case name as the pytest id."""
        return self.name


def load_case(path: str | Path) -> Case:
    """Load one case directory."""
    path = Path(path)
    input_data = ProgramInput.model_validate_json((path / "input.json").read_text())
    settings: dict[str, Any] = {}
    if (path / "case.json").exists():
        settings = json.loads((path / "case.json").read_text())
    expected: StructuredData | None = None
    if (path / "expected.json").exists():
        data_type = get_data_type(input_data.calctype)
        expected = data_type.model_validate_json((path / "expected.json").read_text())
    stdout = (path / "stdout").read_text() if (path / "stdout").exists() else None
    has_files = any(p.name not in RESERVED for p in path.iterdir())
    use_directory = settings.get("directory", has_files)
    encoded = path / "encoded"
    return Case(
        path=path,
        name=f"{path.parent.name}/{path.name}",
        input_data=input_data,
        expected=expected,
        stdout=stdout,
        directory=path if use_directory else None,
        options=dict(settings.get("options", {})),
        failed=bool(settings.get("failed", False)),
        raises=settings.get("raises"),
        encoded=encoded if encoded.is_dir() else None,
    )


def cases(root: str | Path) -> list[Case]:
    """Discover every case directory below ``root`` (any directory holding input.json)."""
    root = Path(root)
    found = sorted(p.parent for p in root.rglob("input.json"))
    return [load_case(p) for p in found]


# ------------------------------------------------------------------ comparison


def _normalize(value: Any) -> Any:
    """Turn models and arrays into plain containers so they compare structurally."""
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump())
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if k not in IGNORED_KEYS}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _diff(
    actual: Any, expected: Any, where: str, rtol: float, atol: float
) -> str | None:
    """Return a description of the first difference, or None when equal."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(actual) | set(expected)):
            if key not in actual:
                return f"{where}.{key}: missing (expected {expected[key]!r})"
            if key not in expected:
                return f"{where}.{key}: unexpected value {actual[key]!r}"
            problem = _diff(actual[key], expected[key], f"{where}.{key}", rtol, atol)
            if problem:
                return problem
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{where}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            problem = _diff(a, e, f"{where}[{i}]", rtol, atol)
            if problem:
                return problem
        return None
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not isinstance(expected, bool) and not isinstance(actual, bool):
            if math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
                return None
            return f"{where}: {actual!r} != {expected!r}"
    if actual != expected:
        return f"{where}: {actual!r} != {expected!r}"
    return None


def assert_data_equal(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-9,
    atol: float = 0.0,
    where: str = "data",
) -> None:
    """Compare qcdata objects, arrays, or containers structurally.

    Numbers compare within ``rtol``/``atol``; ``execution`` metadata and
    ``qcdata_version`` are ignored because they vary between runs.

    Raises:
        AssertionError: Describing the first difference found.
    """
    problem = _diff(_normalize(actual), _normalize(expected), where, rtol, atol)
    if problem:
        raise AssertionError(problem)


# ------------------------------------------------------------------- run_case


def _expected_slice(expected: dict[str, Any], target: str | tuple[str, ...]) -> Any:
    """Read the value at a parser target from a dumped expected model."""
    node: Any = expected
    for key in target if isinstance(target, tuple) else (target,):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _exception_class(name: str) -> type[BaseException]:
    """Resolve an exception name from qccodec.exceptions or builtins."""
    if hasattr(exceptions, name):
        return getattr(exceptions, name)
    import builtins

    if hasattr(builtins, name):
        return getattr(builtins, name)
    raise ValueError(f"Unknown exception class {name!r} in case.json")


def check_encoding(codec: Codec, case: Case, *, rtol: float = 1e-9) -> None:
    """Compare ``codec.encode(case.input_data)`` against the case's ``encoded/`` files."""
    assert case.encoded is not None, f"{case}: no encoded/ directory"
    native = codec.encode(case.input_data)
    command_file = case.encoded / "command.txt"
    if command_file.exists():
        assert isinstance(native, SubprocessInput), f"{case}: expected a subprocess"
        expected_command = command_file.read_text().splitlines()
        assert list(native.command) == expected_command, (
            f"{case}: command {list(native.command)!r} != {expected_command!r}"
        )
    for path in sorted(p for p in case.encoded.rglob("*") if p.is_file()):
        name = path.relative_to(case.encoded).as_posix()
        if name == "command.txt":
            continue
        assert name in native.files, f"{case}: encode() did not generate {name!r}"
        produced = native.files[name]
        expected_bytes = path.read_bytes()
        if isinstance(produced, bytes):
            assert produced == expected_bytes, f"{case}: {name} differs"
        else:
            expected_text = expected_bytes.decode()
            assert _unversioned(produced) == _unversioned(expected_text), (
                f"{case}: {name} differs:\n--- expected\n{expected_text}"
                f"--- produced\n{produced}"
            )


def _unversioned(text: str) -> str:
    """Drop the qcdata version stamp so fixtures survive qcdata releases."""
    return _QCDATA_VERSION.sub("qcdata_version=*", text)


def check_parsers(
    codec: Codec, case: Case, *, rtol: float = 1e-9, atol: float = 0.0
) -> None:
    """Call every applicable parser directly and compare with its expected slice."""
    assert case.expected is not None
    expected = _normalize(case.expected)
    ctx = Context(
        codec=codec,
        calctype=case.input_data.calctype,
        stdout=case.stdout,
        directory=case.directory,
        input_data=case.input_data,
        failed=case.failed,
        options={**codec.options, **case.options},
    )
    for spec in codec.parsers:
        if ctx.calctype not in spec.calctypes:
            continue
        try:
            contents = ctx.contents(spec)
        except FileNotFoundError:
            continue
        where = f"{case} parser {spec.parser.__name__}"
        try:
            value = spec.parser(contents)
        except MatchNotFoundError:
            assert not spec.required or case.failed, (
                f"{where}: required parser found no match"
            )
            if spec.target is not None:
                found = _expected_slice(expected, spec.target)
                assert found in (None, [], {}), (
                    f"{where}: no match, but expected.json holds {found!r}"
                )
            continue
        if spec.target is None:
            for key, item in value.items():
                assert_data_equal(
                    item,
                    expected.get(key),
                    rtol=rtol,
                    atol=atol,
                    where=f"{where}.{key}",
                )
        else:
            assert_data_equal(
                value,
                _expected_slice(expected, spec.target),
                rtol=rtol,
                atol=atol,
                where=where,
            )


def run_case(
    codec: Codec, case: Case, *, rtol: float = 1e-9, atol: float = 0.0
) -> None:
    """Check a codec against one fixture case.

    Verifies, in order: the encoded native inputs when ``encoded/`` exists; the full
    decode, which must produce ``expected.json`` or raise the exception named in
    ``case.json`` with ``expected.json`` attached as partial data; and each registered
    parser applied directly to its own file.

    Raises:
        AssertionError: With a description of the first mismatch.
    """
    if case.encoded is not None:
        check_encoding(codec, case, rtol=rtol)
    kwargs: dict[str, Any] = dict(
        stdout=case.stdout,
        directory=case.directory,
        input_data=case.input_data,
        failed=case.failed,
        options=case.options or None,
    )
    if case.raises:
        exc_type = _exception_class(case.raises)
        try:
            codec.parse(case.input_data.calctype, **kwargs)
        except exc_type as exc:
            if case.expected is not None:
                assert_data_equal(
                    getattr(exc, "data", None),
                    case.expected,
                    rtol=rtol,
                    atol=atol,
                    where=f"{case} partial",
                )
        else:
            raise AssertionError(f"{case}: expected {case.raises} but decode succeeded")
        return
    assert case.expected is not None, (
        f"{case}: expected.json is required unless raises is set"
    )
    data = codec.parse(case.input_data.calctype, **kwargs)
    assert_data_equal(data, case.expected, rtol=rtol, atol=atol, where=str(case))
    if codec.parsers:
        check_parsers(codec, case, rtol=rtol, atol=atol)


# ---------------------------------------------------------------- check_codec


def _target_exists(model: type[BaseModel], target: str | tuple[str, ...]) -> bool:
    """Whether a parser target names a field (possibly nested) on a result model."""
    keys = target if isinstance(target, tuple) else (target,)
    current: Any = model
    for key in keys:
        if not (isinstance(current, type) and issubclass(current, BaseModel)):
            return True  # a dict or untyped container accepts any key
        if key not in current.model_fields:
            return False
        annotation = current.model_fields[key].annotation
        current = _model_in(annotation)
    return True


def _model_in(annotation: Any) -> Any:
    """Return the BaseModel subclass inside an annotation, or the annotation itself."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in getattr(annotation, "__args__", ()) or ():
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return annotation


def check_codec(
    codec: Codec,
    *,
    inputs: Iterable[ProgramInput] = (),
    calctypes: Collection[CalcType] | None = None,
) -> None:
    """Verify the structural contract every codec must satisfy.

    Checks that parser targets name real result fields for every calculation type
    they apply to, that decoding nothing in recovery mode yields a typed empty result
    and in strict mode raises with typed partial data, that option defaults are
    well formed, and that encoding each of ``inputs`` yields a native invocation whose
    files stay inside the calculation directory.

    Args:
        codec: The codec to check.
        inputs: Inputs the codec must be able to encode.
        calctypes: Calculation types to exercise; defaults to ``codec.calctypes``.

    Raises:
        AssertionError: Listing every problem found.
    """
    problems: list[str] = []
    types = list(calctypes if calctypes is not None else codec.calctypes)
    if not codec.program:
        problems.append("program name is empty")
    if not codec.calctypes:
        problems.append("no calctypes declared")
    for key in codec.options:
        if not isinstance(key, str) or not key:
            problems.append(f"option name {key!r} is not a non-empty string")

    for spec in codec.parsers:
        problems.extend(_check_spec(spec, types))

    if codec.parsers:
        for calctype in types:
            data_type = get_data_type(calctype)
            try:
                empty = codec.parse(calctype, failed=True)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{calctype.value}: parse(failed=True) raised {exc!r}")
            else:
                if not isinstance(empty, data_type):
                    problems.append(
                        f"{calctype.value}: parse(failed=True) returned {type(empty).__name__}"
                    )
            try:
                codec.parse(calctype)
            except ParserError as exc:
                if not isinstance(exc.data, data_type):
                    problems.append(
                        f"{calctype.value}: strict parse raised without typed partial data"
                    )
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{calctype.value}: strict parse raised {exc!r}")

    for inp in inputs:
        try:
            native = codec.encode(inp)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"encode({inp.calctype.value}) raised {exc!r}")
            continue
        for name in native.files:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                problems.append(
                    f"encode({inp.calctype.value}) file escapes directory: {name!r}"
                )
        if isinstance(native, SubprocessInput) and not native.command:
            problems.append(f"encode({inp.calctype.value}) returned an empty command")
        if isinstance(native, PythonInput) and not callable(native.function):
            problems.append(f"encode({inp.calctype.value}) function is not callable")

    gated = [
        spec.parser.__name__ for spec in codec.parsers if spec.versions is not None
    ]
    has_version_parser = any(spec.target == VERSION_TARGET for spec in codec.parsers)
    if gated and not has_version_parser:
        problems.append(
            f"parsers {gated} declare versions but no parser fills "
            f"{VERSION_TARGET!r}, so the version is only known when passed in"
        )

    if codec.propagate_wfn is not None and not callable(codec.propagate_wfn):
        problems.append("propagate_wfn is not callable")
    if problems:
        raise AssertionError(
            f"Codec {codec.program!r} failed checks:\n  " + "\n  ".join(problems)
        )


def _check_spec(spec: ParserSpec, types: list[CalcType]) -> list[str]:
    """Problems with one parser specification."""
    problems: list[str] = []
    name = spec.parser.__name__
    if (
        spec.file not in (STDOUT, DIRECTORY)
        and isinstance(spec.file, str)
        and (Path(spec.file).is_absolute() or ".." in Path(spec.file).parts)
    ):
        problems.append(f"parser {name}: file {spec.file!r} escapes the directory")
    if spec.target is None:
        return problems
    for calctype in types:
        if calctype not in spec.calctypes:
            continue
        if not _target_exists(get_data_type(calctype), spec.target):
            problems.append(
                f"parser {name}: target {spec.target!r} is not a field of "
                f"{get_data_type(calctype).__name__} ({calctype.value})"
            )
    return problems
