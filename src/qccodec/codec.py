"""One ``Codec`` per program: translate inputs in, parse native outputs out."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from qcdata import (
    CalcType,
    Data,
    Inputs,
    ProgramInput,
    ProgramOutput,
    StructuredData,
    get_data_type,
)

from .exceptions import (
    CalculationError,
    DataCollectorError,
    EncoderError,
    MatchNotFoundError,
    ParserError,
    RegistryError,
)
from .native import NativeInput, NativeOutput, PythonInput, SubprocessInput

logger = logging.getLogger(__name__)

STDOUT = "stdout"
DIRECTORY = "directory"
VERSION_TARGET = ("provenance", "program_version")

Target = str | tuple[str, ...]
FileRef = str | Callable[["Context"], str]
F = TypeVar("F", bound=Callable[..., Any])

Encoder = Callable[[Any], NativeInput]
Decoder = Callable[[Any, NativeOutput], Data]
Failure = Callable[["Context"], "str | None"]
PropagateWfn = Callable[[ProgramOutput, ProgramInput], ProgramInput]
VersionKey = Callable[[str], str]


@dataclass(frozen=True)
class ParserSpec:
    """A registered parser and the metadata that decides when it runs.

    Attributes:
        parser: The parser function. It receives file contents, or a ``Context``
            when ``file`` is ``"directory"``.
        file: ``"stdout"``, ``"directory"``, a file name or glob pattern relative
            to the calculation directory, or a function of the ``Context`` that
            returns such a name.
        calctypes: Calculation types for which this parser runs.
        required: Whether a missing file or value fails decoding.
        target: Where the parsed value is stored on the result. A tuple creates
            nested fields. ``None`` is permitted only for directory parsers, which
            then return a mapping merged into the result.
        versions: Program versions this parser applies to, as a PEP 440
            specifier set; ``None`` means every version.
    """

    parser: Callable[..., Any]
    file: FileRef
    calctypes: frozenset[CalcType]
    required: bool
    target: Target | None
    versions: SpecifierSet | None = None


@dataclass
class Context:
    """Everything a parser may need beyond the contents of one file.

    Attributes:
        codec: The codec running the decode; used for nested decoding.
        calctype: The calculation type being decoded.
        stdout: Captured program output, if available.
        directory: The calculation directory, if available.
        input_data: The input used for the calculation, if available.
        failed: Whether partial data is being recovered from a failed run.
        options: The codec's options merged with those passed for this decode.
        version: The program version string, once known.
    """

    codec: Codec
    calctype: CalcType
    stdout: str | None = None
    directory: Path | None = None
    input_data: ProgramInput | None = None
    failed: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    version: str | None = None

    def path(self, pattern: str) -> Path:
        """Resolve a file name or glob pattern to exactly one path in the directory.

        Raises:
            FileNotFoundError: No directory is available or nothing matches.
            ParserError: The pattern matches more than one file.
        """
        if self.directory is None:
            raise FileNotFoundError(pattern)
        matches = sorted(p for p in self.directory.glob(pattern) if p.is_file())
        if not matches:
            raise FileNotFoundError(pattern)
        if len(matches) > 1:
            names = ", ".join(str(p.relative_to(self.directory)) for p in matches)
            raise ParserError(f"Pattern {pattern!r} matched several files: {names}")
        return matches[0]

    def read(self, pattern: str) -> str:
        """Read the text of the single file matching a name or glob pattern."""
        return self.path(pattern).read_text()

    def contents(self, spec: ParserSpec) -> Any:
        """Return what a parser receives: stdout, this context, or a file's text.

        Raises:
            FileNotFoundError: The parser's input is not available.
        """
        if spec.file == STDOUT:
            if self.stdout is None:
                raise FileNotFoundError(STDOUT)
            return self.stdout
        if spec.file == DIRECTORY:
            if self.directory is None:
                raise FileNotFoundError(DIRECTORY)
            return self
        return self.read(spec.file(self) if callable(spec.file) else spec.file)

    def decode(
        self,
        calctype: CalcType | str,
        *,
        stdout: str | None = None,
        directory: str | Path | None = None,
    ) -> StructuredData:
        """Parse another set of artifacts with this codec, e.g. one trajectory step.

        The failure hook is not run again: the whole run was already checked before
        this parser was called, and partial logs such as one optimization step
        would not pass a whole-run check.
        """
        return self.codec.parse(
            calctype,
            stdout=stdout,
            directory=directory,
            input_data=self.input_data,
            failed=self.failed,
            options=self.options,
            version=self.version,
            check_failure=False,
        )


class Codec:
    """Program-specific translation between qcdata objects and native inputs/outputs.

    A codec encodes a ``ProgramInput`` into a ``SubprocessInput`` or ``PythonInput``
    and decodes the native output back into a qcdata ``*Data`` object. File-based
    programs register small parsers with ``@codec.parser(...)``; Python-native
    programs supply a ``decode`` function instead.

    Args:
        program: Name matched against ``ProgramInput.program``.
        calctypes: Calculation types this codec can encode.
        options: Decode options and their defaults, e.g. what to collect. Values
            passed to ``decode``/``parse`` are validated against these keys and
            reach parsers as ``Context.options``.
        encode: Translate an input into a native invocation. See ``encoder``.
        decode: Translate native output into data. See ``decoder``.
        failure: Detect a failure the program reported. See ``failure``.
        propagate_wfn: ``(output, input_data) -> ProgramInput`` placing the
            wavefunction from a previous output onto the next input as a guess.
        version_key: Turns the program's version string into a PEP 440 version
            for matching parser ``versions``. Not needed when the program already
            prints versions like ``6.1.0``.
        parsers: Pre-built parser specifications.
    """

    def __init__(
        self,
        program: str,
        calctypes: Collection[CalcType | str],
        *,
        options: Mapping[str, Any] | None = None,
        encode: Encoder | None = None,
        decode: Decoder | None = None,
        failure: Failure | None = None,
        propagate_wfn: PropagateWfn | None = None,
        version_key: VersionKey | None = None,
        parsers: Iterable[ParserSpec] = (),
    ) -> None:
        """Create a codec; hooks may also be attached later with decorators."""
        self.program: str = program
        self.calctypes: frozenset[CalcType] = frozenset(CalcType(c) for c in calctypes)
        self.options: dict[str, Any] = dict(options or {})
        self.propagate_wfn: PropagateWfn | None = propagate_wfn
        self.version_key: VersionKey | None = version_key
        self.parsers: list[ParserSpec] = list(parsers)
        self._encode: Encoder | None = encode
        self._decode: Decoder | None = None
        self._failure: Failure | None = failure
        if decode is not None:
            self.decoder(decode)

    def __repr__(self) -> str:
        """Summarize the codec by program and registered pieces."""
        return (
            f"Codec({self.program!r}, "
            f"calctypes={sorted(c.value for c in self.calctypes)}, "
            f"parsers={len(self.parsers)})"
        )

    # ------------------------------------------------------------------ hooks

    def encoder(self, func: F) -> F:
        """Register ``func(input_data) -> SubprocessInput | PythonInput``."""
        self._encode = func
        return func

    def decoder(self, func: F) -> F:
        """Register ``func(input_data, native_output) -> Data`` in place of parsers.

        Raises:
            RegistryError: The codec has parsers or options, which only the parser
                pipeline uses.
        """
        if self.parsers:
            raise RegistryError(
                f"Codec {self.program!r} has parsers; a decoder would replace them."
            )
        if self.options:
            raise RegistryError(
                f"Codec {self.program!r} declares options, which only parsers receive."
            )
        self._decode = func
        return func

    def failure(self, func: F) -> F:
        """Register ``func(ctx) -> str | None`` returning a message if the program failed.

        A message makes decoding recover partial data and then raise
        ``CalculationError`` with that data attached.
        """
        self._failure = func
        return func

    def parser(
        self,
        *,
        target: Target | None = None,
        calctypes: Collection[CalcType | str] | None = None,
        file: FileRef = STDOUT,
        required: bool = True,
        versions: str | None = None,
    ) -> Callable[[F], F]:
        """Register a parser function.

        Args:
            target: Result field to fill, or a tuple for a nested field. Directory
                parsers may omit it and return a mapping merged into the result.
            calctypes: Calculation types this parser applies to; all when omitted.
            file: ``"stdout"`` (default), ``"directory"`` (the parser receives a
                ``Context``), a file name/glob relative to the directory, or a
                function of the ``Context`` returning such a name.
            required: If True, a missing file or ``MatchNotFoundError`` fails
                decoding unless partial data is being recovered.
            versions: Program versions this parser applies to, as a PEP 440
                specifier such as ``">=6.0"`` or ``">=5.0,<6.0"``; every version
                when omitted. Parsers for the same target may coexist when each
                declares ``versions``.

        Returns:
            The original function, unchanged, so it can be called and tested directly.
        """
        types = (
            frozenset(CalcType)
            if calctypes is None
            else frozenset(CalcType(c) for c in calctypes)
        )
        try:
            specifier = SpecifierSet(versions) if versions is not None else None
        except InvalidSpecifier as exc:
            raise RegistryError(f"Invalid versions specifier {versions!r}.") from exc

        def decorate(func: F) -> F:
            """Validate and record the parser specification."""
            if self._decode is not None:
                raise RegistryError(
                    f"Codec {self.program!r} has a decoder; parsers would never run."
                )
            if target is None and file != DIRECTORY:
                raise RegistryError(
                    f"Parser '{func.__name__}' must have a target unless file='directory'."
                )
            if target is not None and not (
                isinstance(target, str)
                or (
                    isinstance(target, tuple)
                    and all(isinstance(k, str) for k in target)
                )
            ):
                raise RegistryError(
                    f"Target {target!r} must be a string or tuple of strings."
                )
            if specifier is not None and target == VERSION_TARGET:
                raise RegistryError(
                    "The program version parser runs on every version; "
                    "it cannot declare versions."
                )
            for existing in self.parsers:
                if (
                    target is not None
                    and existing.target == target
                    and types & existing.calctypes
                    and (specifier is None or existing.versions is None)
                ):
                    raise RegistryError(
                        f"Duplicate parser target {target!r} for overlapping "
                        "calculation types. Declare versions on both parsers if "
                        "they handle different program versions."
                    )
            self.parsers.append(
                ParserSpec(func, file, types, required, target, specifier)
            )
            return func

        return decorate

    # -------------------------------------------------------------- interface

    def encode(self, input_data: Inputs) -> NativeInput:
        """Translate an input into a native invocation without executing it.

        Raises:
            EncoderError: The codec has no encoder, the input's program or
                calculation type is not supported, or the encoder returned
                something other than a ``SubprocessInput`` or ``PythonInput``.
        """
        if self._encode is None:
            raise EncoderError(f"Codec {self.program!r} has no encoder.")
        if isinstance(input_data, ProgramInput):
            if input_data.program != self.program:
                raise EncoderError(
                    f"Codec {self.program!r} cannot encode {input_data.program!r}."
                )
            if input_data.calctype not in self.calctypes:
                raise EncoderError(
                    f"{self.program} does not support {input_data.calctype.value}."
                )
        native = self._encode(input_data)
        if not isinstance(native, (SubprocessInput, PythonInput)):
            raise EncoderError(
                f"{self.program} encoder must return SubprocessInput or PythonInput, "
                f"not {type(native).__name__}."
            )
        return native

    def decode(
        self,
        input_data: Inputs,
        native_output: NativeOutput,
        options: Mapping[str, Any] | None = None,
    ) -> Data:
        """Interpret native output, returning data or raising with partial data.

        When the run failed (for example a nonzero exit status), whatever can be
        parsed is recovered, and if the codec's failure hook explains the failure
        ``CalculationError`` carries that explanation and the recovered data.

        Args:
            input_data: The input that produced the output.
            native_output: What the runner observed.
            options: Overrides for this codec's declared ``options``.

        Raises:
            CalculationError: The program failed and the failure hook says why.
            ParserError: A required output is missing or malformed in a run that
                did not fail.
        """
        if self._decode is not None:
            self._merge_options(options)  # rejects unknown keys
            return self._decode(input_data, native_output)
        if not isinstance(input_data, ProgramInput):
            raise ParserError(f"Codec {self.program!r} has no decoder for file inputs.")
        return self.parse(
            input_data.calctype,
            stdout=native_output.stdout,
            directory=native_output.directory,
            input_data=input_data,
            failed=native_output.failed,
            options=options,
            report_failure=True,
        )

    @overload
    def parse(
        self,
        calctype: CalcType | str,
        *,
        stdout: str | None = None,
        directory: str | Path | None = None,
        input_data: ProgramInput | None = None,
        failed: bool = False,
        options: Mapping[str, Any] | None = None,
        version: str | None = None,
        as_dict: Literal[True],
        check_failure: bool = True,
        report_failure: bool | None = None,
    ) -> dict[str, Any]: ...

    @overload
    def parse(
        self,
        calctype: CalcType | str,
        *,
        stdout: str | None = None,
        directory: str | Path | None = None,
        input_data: ProgramInput | None = None,
        failed: bool = False,
        options: Mapping[str, Any] | None = None,
        version: str | None = None,
        as_dict: Literal[False] = False,
        check_failure: bool = True,
        report_failure: bool | None = None,
    ) -> StructuredData: ...

    def parse(
        self,
        calctype: CalcType | str,
        *,
        stdout: str | None = None,
        directory: str | Path | None = None,
        input_data: ProgramInput | None = None,
        failed: bool = False,
        options: Mapping[str, Any] | None = None,
        version: str | None = None,
        as_dict: bool = False,
        check_failure: bool = True,
        report_failure: bool | None = None,
    ) -> StructuredData | dict[str, Any]:
        """Run this codec's parsers over saved artifacts without running the program.

        Args:
            calctype: The type of calculation that produced the artifacts.
            stdout: The program's captured output.
            directory: The calculation directory holding output files.
            input_data: The input used for the calculation, for parsers that need it.
            failed: Recover what is available from a failed calculation. Missing
                files and values are tolerated; malformed values still fail.
            options: Overrides for this codec's declared ``options``.
            version: The program version, when the output does not print it or
                should be overridden. Otherwise the version parser supplies it.
            as_dict: Return the collected values instead of a validated model.
            check_failure: Run the codec's failure hook first. Nested decodes of
                partial logs turn this off.
            report_failure: Raise ``CalculationError`` when the failure hook
                explains a failure. Defaults to ``not failed``, so recovery mode
                returns partial data quietly; ``decode`` passes True so a program
                that exited with an error still reports why.

        Returns:
            The qcdata model for ``calctype``, or a dictionary when ``as_dict``.

        Raises:
            ParserError: A required file is missing or a value is malformed.
            MatchNotFoundError: A required parser found no match.
            CalculationError: The failure hook explained a failure and
                ``report_failure`` is true. Recovered data is attached to every
                exception as ``.data``.
        """
        ctx = Context(
            codec=self,
            calctype=CalcType(calctype),
            stdout=stdout,
            directory=Path(directory) if directory is not None else None,
            input_data=input_data,
            failed=failed,
            options=self._merge_options(options),
        )
        data_type = get_data_type(ctx.calctype)
        message = (
            self._failure(ctx) if self._failure is not None and check_failure else None
        )
        if message:
            ctx.failed = True
        collected: dict[str, Any] = {"provenance": {"program": self.program}}
        not_found: MatchNotFoundError | None = None
        missing: list[str] = []
        # With an unknown version, version-gated alternatives for one target are
        # tried in order; a miss only counts if no alternative fills the target.
        deferred: dict[object, MatchNotFoundError | str] = {}
        filled: set[object] = set()

        def partial() -> StructuredData:
            """Validate whatever has been collected so far."""
            return data_type(**collected)

        specs = [s for s in self.parsers if ctx.calctype in s.calctypes]
        if version is not None:
            ctx.version = version
            _assign(collected, VERSION_TARGET, version)
            specs = [s for s in specs if s.target != VERSION_TARGET]
        else:
            # The version parser runs first so later parsers can be filtered by it.
            specs.sort(key=lambda s: s.target != VERSION_TARGET)
        for spec in specs:
            name = spec.parser.__name__
            key: object = spec.target if spec.target is not None else name
            gated = spec.versions is not None
            matched = self._version_matches(spec, ctx.version)
            if matched is False:
                continue
            if matched is None and key in filled:
                continue  # an earlier alternative already handled this target
            try:
                value: Any = spec.parser(ctx.contents(spec))
            except FileNotFoundError as exc:
                logger.info("Parser %r: missing %s", name, exc.filename or exc)
                if spec.required and not ctx.failed:
                    if matched is None and gated:
                        deferred.setdefault(key, str(exc.filename or exc))
                    else:
                        missing.append(str(exc.filename or exc))
                continue
            except MatchNotFoundError as exc:
                logger.info("Parser %r found no match", name)
                if spec.required and not ctx.failed:
                    if matched is None and gated:
                        deferred.setdefault(key, exc)
                    elif not_found is None:
                        not_found = exc
                continue
            except ParserError as exc:
                exc.data = partial()
                raise
            if spec.target is None:
                for item_key, item in value.items():
                    _assign(collected, item_key, item)
            else:
                _assign(collected, spec.target, value)
            if spec.target == VERSION_TARGET:
                ctx.version = value
            filled.add(key)
            deferred.pop(key, None)
            logger.debug("Parser %r filled %r", name, spec.target)

        for problem in deferred.values():
            if isinstance(problem, MatchNotFoundError):
                not_found = not_found or problem
            else:
                missing.append(problem)

        if not_found is not None:
            not_found.data = partial()
            raise not_found
        if missing:
            raise ParserError(
                "Missing required output artifact(s): "
                + ", ".join(sorted(set(missing))),
                data=partial(),
            )
        if report_failure is None:
            report_failure = not failed
        if message and report_failure:
            raise CalculationError(message, data=partial())
        return collected if as_dict else partial()

    def _version_matches(self, spec: ParserSpec, version: str | None) -> bool | None:
        """Whether a parser applies to ``version``; None when that cannot be known.

        Parsers without ``versions`` always apply. A gated parser is undecided
        when the version is unknown or cannot be read as a PEP 440 version.
        """
        if spec.versions is None:
            return True
        if version is None:
            return None
        try:
            key = self.version_key(version) if self.version_key else version
            parsed = Version(key)
        except (InvalidVersion, ValueError, TypeError):
            logger.warning(
                "%s version %r is not comparable; trying every alternative.",
                self.program,
                version,
            )
            return None
        return spec.versions.contains(parsed, prereleases=True)

    def _merge_options(self, options: Mapping[str, Any] | None) -> dict[str, Any]:
        """Overlay caller options on the declared defaults, rejecting unknown keys."""
        unknown = sorted(set(options or {}) - set(self.options))
        if unknown:
            raise EncoderError(
                f"{self.program} does not accept options {unknown}; "
                f"available: {sorted(self.options)}"
            )
        return {**self.options, **(options or {})}


def _assign(data: dict[str, Any], target: Target, value: Any) -> None:
    """Store ``value`` at ``target``, creating nested dictionaries as needed."""
    keys = target if isinstance(target, tuple) else (target,)
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    if keys[-1] in node:
        raise DataCollectorError(f"Target {keys!r} was already filled.")
    node[keys[-1]] = value
