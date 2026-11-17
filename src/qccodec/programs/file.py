"""Pass-through codec for ``qcdata.FileInput``: run a command, keep its files.

qccompute runs every FileInput under the ``files`` program: it runs
``<program> <cmdline_args>`` in a scratch directory holding the input's files and
returns every file the command wrote in ``results.files``.
"""

from __future__ import annotations

from qcdata import FileData, FileInput, Provenance

from ..codec import Codec
from ..native import NativeOutput, SubprocessInput
from ..program import CodecProgram


def encode(inp: FileInput) -> SubprocessInput:
    """Describe the executable and the user's files; nothing is translated."""
    return SubprocessInput(command=(inp.program, *inp.cmdline_args), files=inp.files)


def decode(inp: FileInput, out: NativeOutput) -> FileData:
    """Return file-only results; the adapter attaches the collected files."""
    return FileData(provenance=Provenance(program=inp.program))


codec = Codec("file", calctypes=(), encode=encode, decode=decode)

#: The program qccompute runs every FileInput under (entry point ``files``).
program = CodecProgram(codec, name="files")
