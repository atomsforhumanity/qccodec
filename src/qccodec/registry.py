"""Find codecs by program name: registered, built-in, or installed by other packages."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

from qcdata import CalcType, FileInput, Inputs, ProgramInput, StructuredData

from .codec import Codec
from .exceptions import CodecNotFoundError
from .native import NativeInput

# Program name -> module under qccodec.programs. A module exposes ``codec`` or, when
# it serves several names, a ``codecs`` dictionary keyed by program name.
_BUILTIN: dict[str, str] = {
    "terachem": "terachem",
    "orca": "orca",
    "crest": "crest",
    "xtb": "xtb",
    "psi4": "psi4",
    "geometric": "geometric",
    "terachem-fe": "tcpb",
    "terachem-pbs": "tcpb",
}
_REGISTERED: dict[str, Codec] = {}

# Third-party packages publish codecs as entry points in this group. The entry point
# name is the program name and its value is a Codec object::
#
#     [project.entry-points."qccodec.codecs"]
#     myprog = "mypackage.codec:codec"
ENTRY_POINT_GROUP = "qccodec.codecs"


def register(codec: Codec) -> Codec:
    """Make a codec resolvable by its program name for this process.

    A registered codec takes precedence over a built-in with the same name, so
    scripts can both add new programs and override shipped integrations. The
    codec is returned so this can be used as a decorator-style one-liner.
    """
    _REGISTERED[codec.program] = codec
    return codec


def unregister(program: str) -> None:
    """Remove a codec registered with ``register``; built-ins are unaffected."""
    _REGISTERED.pop(program, None)


def supported_programs() -> tuple[str, ...]:
    """Built-in and installed plugin program names plus any registered in this process."""
    plugins = [ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)]
    return tuple(dict.fromkeys([*_BUILTIN, *plugins, *_REGISTERED]))


def _plugin(name: str) -> Codec | None:
    """Load an installed codec published under the entry point group, if any."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name != name:
            continue
        loaded = ep.load()
        if not isinstance(loaded, Codec):
            raise CodecNotFoundError(
                f"Entry point {ep.value!r} for program '{name}' is not a qccodec.Codec."
            )
        return loaded
    return None


def get_codec(program: str | Inputs) -> Codec:
    """Resolve the codec for a program name or input object.

    Resolution order: codecs passed to ``register``, built-ins, then codecs installed
    by other packages under the ``qccodec.codecs`` entry point group. A ``FileInput``
    resolves to the pass-through file codec.

    Args:
        program: A program name, or an input whose program should be resolved.

    Raises:
        CodecNotFoundError: No codec is available for the program.
    """
    if isinstance(program, ProgramInput):
        name = program.program
    elif isinstance(program, FileInput):
        from .programs.file import codec as file_codec

        return file_codec
    else:
        name = program

    codec = _REGISTERED.get(name)
    if codec is None and name in _BUILTIN:
        module = import_module(f"qccodec.programs.{_BUILTIN[name]}")
        codec = module.codecs[name] if hasattr(module, "codecs") else module.codec
    if codec is None:
        codec = _plugin(name)
    if codec is None:
        raise CodecNotFoundError(f"No codec found for program '{name}'.")
    return codec


def encode(input_data: Inputs) -> NativeInput:
    """Prepare a complete native invocation for an input without executing it."""
    return get_codec(input_data).encode(input_data)


def decode(
    program: str,
    calctype: CalcType | str,
    *,
    stdout: str | None = None,
    directory: Any = None,
    input_data: ProgramInput | None = None,
    failed: bool = False,
    options: Mapping[str, Any] | None = None,
    version: str | None = None,
) -> StructuredData:
    """Decode saved program output into structured data without running anything.

    Args:
        program: Name of the program that produced the output.
        calctype: The type of calculation that was run.
        stdout: The program's captured output.
        directory: The calculation directory holding output files.
        input_data: The input used for the calculation, for parsers that need it.
        failed: Recover available data from a failed calculation.
        options: Overrides for the codec's declared decode options.
        version: The program version, when the output does not print it.

    Returns:
        The qcdata model matching ``calctype``.
    """
    return get_codec(program).parse(
        calctype,
        stdout=stdout,
        directory=directory,
        input_data=input_data,
        failed=failed,
        options=options,
        version=version,
    )
