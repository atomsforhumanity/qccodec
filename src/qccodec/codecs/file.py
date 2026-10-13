"""Pass-through translation for qcdata.FileInput."""

from __future__ import annotations

from qcdata import (
    FileData,
    FileInput,
    Provenance,
)

from ..native import NativeOutput, SubprocessInput


def encode(inp: FileInput) -> SubprocessInput:
    """Describe an executable and supplied native files without scientific translation."""
    return SubprocessInput(command=(inp.program, *inp.cmdline_args), files=inp.files)


def decode(inp: FileInput, out: NativeOutput) -> FileData:
    """Provide file-only result provenance for generic command execution."""
    return FileData(provenance=Provenance(program=inp.program))
