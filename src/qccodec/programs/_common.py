"""Helpers shared by file-based program codecs."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from qcdata import ProgramInput
from qcref import constants, periodic_table

from ..codec import Context
from ..exceptions import EncoderError, ParserError
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
    threads: int = 1,
    gpus: int = 0,
) -> SubprocessInput:
    """Build ``<program> <filename> <arguments> <cmdline_args>`` and the input files.

    ``threads`` and ``gpus`` declare what the calculation uses, so runners computing
    several at once do not oversubscribe the machine.

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
        threads=threads,
        gpus=gpus,
    )


def parse_g98_freqs(contents: str) -> list[float]:
    """Parse vibrational frequencies (wavenumbers) from a Gaussian-98-style ``g98.out``."""
    regex = r"Frequencies\s+--\s+(?P<floats>(?:-?\d+\.\d+\s*)+)"
    matches = re_finditer(regex, contents)
    return [float(f) for m in matches for f in m.group("floats").split()]


def _g98_mass_weighted_modes(contents: str) -> list[list[list[float]]]:
    """Read the normal coordinates printed in ``g98.out``, (n_modes, n_atoms, 3).

    xtb (and CREST, which runs it) writes the eigenvectors of the mass-weighted
    Hessian here, where Gaussian puts Cartesian displacements.
    """
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
                    [float(v) for v in displacements[index : index + 3]]
                )
        modes.extend(block_modes)
    return modes


def _g98_coordinates(contents: str) -> np.ndarray:
    """Read the ``Standard orientation`` coordinates of ``g98.out``, in bohr.

    Raises:
        ParserError: The file has no coordinate block.
    """
    lines = contents.splitlines()
    try:
        start = next(
            i for i, line in enumerate(lines) if "Standard orientation" in line
        )
    except StopIteration:
        raise ParserError("g98.out has no Standard orientation block.") from None
    rules = [i for i in range(start, len(lines)) if lines[i].strip().startswith("---")]
    if len(rules) < 3:
        raise ParserError("g98.out has an incomplete Standard orientation block.")
    rows = [line.split()[3:6] for line in lines[rules[1] + 1 : rules[2]]]
    return np.array(rows, dtype=float) * constants.ANGSTROM_TO_BOHR


def unit_modes(modes: Any) -> list[list[list[float]]]:
    """Scale each normal mode to unit length, as qcdata defines them."""
    array = np.asarray(modes, dtype=float)
    norms = np.linalg.norm(array.reshape(len(array), -1), axis=1)
    if np.any(norms == 0):
        raise ParserError("A normal mode has no displacement.")
    return (array / norms[:, None, None]).tolist()


def rotation_onto(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper rotation that best maps centered ``source`` onto ``target``.

    Args:
        source: (n_atoms, 3) coordinates in a program's frame, in bohr.
        target: (n_atoms, 3) coordinates of the input structure, in bohr.

    Raises:
        ParserError: The two sets of coordinates are not the same structure.
    """
    p = source - source.mean(axis=0)
    q = target - target.mean(axis=0)
    if np.sqrt(np.mean(np.sum((p - q) ** 2, axis=1))) < 1e-5:
        # Only translated (xtb, ORCA), within the printed precision: the exact
        # identity, so results do not carry SVD round-off that varies by platform.
        return np.eye(3)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    rmsd = np.sqrt(np.mean(np.sum((p @ rotation.T - q) ** 2, axis=1)))
    if rmsd > 1e-3:
        raise ParserError(
            f"The program's coordinates differ from the input structure (RMSD "
            f"{rmsd:.2g} bohr after alignment), so its normal modes cannot be "
            "mapped onto the input."
        )
    return rotation


def parse_g98_normal_modes(ctx: Context) -> list[list[list[float]]]:
    """Normal modes from xtb's ``g98.out`` in qcdata's convention.

    The file holds mass-weighted eigenvectors, rounded to two decimals, in the frame
    of its own ``Standard orientation`` coordinates. They are converted to Cartesian
    displacements with the input's atomic masses, rotated onto the input frame (CREST
    reorients the molecule; xtb does not), and scaled to unit length.
    """
    if ctx.input_data is None:
        raise ParserError("Normal mode parsing requires input_data.")
    contents = ctx.read("g98.out")
    structure = ctx.input_data.require_structure()
    modes = np.asarray(_g98_mass_weighted_modes(contents))
    if modes.size == 0:
        raise ParserError("g98.out holds no normal modes.")
    masses = np.array([float(periodic_table[s].mass) for s in structure.symbols])
    cartesian = modes / np.sqrt(masses)[None, :, None]
    rotation = rotation_onto(_g98_coordinates(contents), structure.geometry)
    return unit_modes(cartesian @ rotation.T)
