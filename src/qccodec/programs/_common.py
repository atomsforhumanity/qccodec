"""Helpers shared by file-based program codecs."""

from __future__ import annotations

import re

from qcdata import ProgramInput
from qcref import constants

from ..exceptions import EncoderError
from ..native import SubprocessInput
from ..utils import re_finditer


def invocation(
    inp: ProgramInput,
    generated: dict[str, str | bytes],
    filename: str,
    *,
    wavefunction_files: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
    arguments: tuple[str, ...] = (),
) -> SubprocessInput:
    """Build ``<program> <filename> <arguments> <cmdline_args>`` and the input files.

    Raises:
        EncoderError: A supplied file would overwrite a generated input file.
    """
    conflicts = generated.keys() & inp.files.keys()
    if conflicts:
        raise EncoderError(
            f"Supplied files conflict with generated inputs: {sorted(conflicts)}"
        )
    return SubprocessInput(
        command=(inp.program, filename, *arguments, *inp.cmdline_args),
        files={**inp.files, **generated},
        wavefunction_files=wavefunction_files,
        environment=environment or {},
    )


def parse_g98_freqs(contents: str) -> list[float]:
    """Parse vibrational frequencies (wavenumbers) from a Gaussian-98-style ``g98.out``."""
    regex = r"Frequencies\s+--\s+(?P<floats>(?:-?\d+\.\d+\s*)+)"
    matches = re_finditer(regex, contents)
    return [float(f) for m in matches for f in m.group("floats").split()]


def parse_g98_normal_modes(contents: str) -> list[list[list[float]]]:
    """Parse normal mode displacements from ``g98.out`` as (n_modes, n_atoms, 3) in Bohr."""
    block_re = re.compile(r"(Frequencies\s+--.*?)(?=Frequencies\s+--|$)", re.DOTALL)
    floats_re = re.compile(r"[-+]?\d*\.\d+")
    modes: list[list[list[float]]] = []
    for block in block_re.findall(contents):
        n_freqs = len(block.splitlines()[0].split()[2:])
        displacements = floats_re.findall(block.split("Atom AN")[1])
        block_modes: list[list[list[float]]] = [[] for _ in range(n_freqs)]
        n_atoms = len(displacements) // (3 * n_freqs)
        for i in range(n_atoms):
            for j in range(n_freqs):
                index = i * (3 * n_freqs) + j * 3
                block_modes[j].append(
                    [
                        float(v) * constants.ANGSTROM_TO_BOHR
                        for v in displacements[index : index + 3]
                    ]
                )
        modes.extend(block_modes)
    return modes
