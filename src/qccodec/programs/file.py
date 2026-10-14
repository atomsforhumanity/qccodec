"""Pass-through codec for ``qcdata.FileInput``: run a command, keep its files."""

from __future__ import annotations

from qcdata import FileData, FileInput, Provenance

from ..codec import Codec
from ..native import NativeOutput, SubprocessInput


def encode(inp: FileInput) -> SubprocessInput:
    """Describe the executable and the user's files; nothing is translated."""
    return SubprocessInput(command=(inp.program, *inp.cmdline_args), files=inp.files)


def decode(inp: FileInput, out: NativeOutput) -> FileData:
    """Return file-only results; qccompute attaches the collected files."""
    return FileData(provenance=Provenance(program=inp.program))


codec = Codec("file", calctypes=(), encode=encode, decode=decode)
