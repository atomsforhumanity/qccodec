"""Helpers shared by file-based program codecs."""

from __future__ import annotations

from qcdata import ProgramInput

from ..exceptions import EncoderError
from ..native import SubprocessInput


def invocation(
    inp: ProgramInput,
    generated: dict[str, str | bytes],
    filename: str,
    *,
    wavefunction_files: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
) -> SubprocessInput:
    """Build ``<program> <filename> <cmdline_args>`` with generated and supplied files.

    Raises:
        EncoderError: A supplied file would overwrite a generated input file.
    """
    conflicts = generated.keys() & inp.files.keys()
    if conflicts:
        raise EncoderError(
            f"Supplied files conflict with generated inputs: {sorted(conflicts)}"
        )
    return SubprocessInput(
        command=(inp.program, filename, *inp.cmdline_args),
        files={**inp.files, **generated},
        wavefunction_files=wavefunction_files,
        environment=environment or {},
    )
