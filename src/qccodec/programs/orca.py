"""ORCA: write ``orca.inp``, run ``orca orca.inp``, and parse stdout and output files.

Inputs map to ORCA as follows:

- ``model.method`` and ``model.basis`` start the simple-input line, ``! method basis``.
  Extra simple-input words, such as ``CPCM(water)``, ``TightSCF``, or ``D3BJ``, can go in
  ``model.method`` after the method name. Leave ``basis`` empty for composite methods.
- The calculation type adds ``! energy``/``engrad``/``freq``/``opt``/``optts``.
  ``{"numgrad": True}`` and ``{"numfreq": True}`` request numerical derivatives.
- A keyword set to ``True`` adds a simple-input word, so ``{"tightscf": True}`` writes
  ``! tightscf``; ``False`` or ``None`` leaves it out. Any other non-dictionary value
  is an error, so nothing is dropped silently.
- ``{"maxcore": 2000}`` and ``{"pal": 4}`` become ``%maxcore`` and ``%pal nprocs``.
- A keyword whose value is a dictionary becomes an input block:
  ``{"scf": {"maxiter": 200}}`` writes ``%scf maxiter 200 end``. Keys are kept as
  written; ORCA ignores case.
- The structure is written to ``geometry.xyz`` with its charge and multiplicity.
  Setting ``coords``, or ``method``/``basis`` inside a block, is an error.
- ``cmdline_args`` are appended to the ``orca`` command.

Calculation types: ``energy``, ``gradient``, ``hessian`` (also the gradient when ORCA
writes it, and the harmonic frequencies and normal modes from ``<basename>.hess``:
unit-length Cartesian displacements without translations and rotations, imaginary
frequencies negative), ``optimization``, and ``transition_state`` (``OptTS``; give ORCA a
starting Hessian, e.g. ``{"geom": {"calc_hess": True}}``). ORCA keeps the input's
coordinates and atom order. No input field carries atom indices.

Run ORCA by its full path: its parallel runs need it, and on many Linux desktops
``/usr/bin/orca`` is the GNOME screen reader. qccompute always runs the absolute path;
put ORCA first on ``PATH`` or set ``QCCOMPUTE_ORCA_EXECUTABLE``.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from qcdata import (
    CalcType,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
    Structure,
)

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..utils import re_search
from ._common import invocation, rotation_onto, unit_modes

codec = Codec(
    "orca",
    calctypes={
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
        CalcType.transition_state,
    },
)

XYZ_FILENAME = "geometry.xyz"
PADDING = 20  # padding between keyword and value


# ------------------------------------------------------------------ encoding


def _validate_keywords(keywords: dict[str, Any]) -> None:
    """Reject keywords that belong on structured qcdata fields. Expects lowercase keys."""
    non_blocks = {"coords": ".structure"}
    non_block_keywords = {
        "method": {"method": ".model.method", "runtyp": ".calctype"},
        "basis": {"basis": ".model.basis"},
    }
    for block, where in non_blocks.items():
        if block in keywords:
            raise EncoderError(
                f"Block '{keywords[block]}' should not be set as a keyword. "
                f"Set its data on '{where}'."
            )
    for block, disallowed in non_block_keywords.items():
        for variable, where in disallowed.items():
            if keywords.get(block, {}).get(variable) is not None:
                raise EncoderError(
                    f"Keyword '{variable}' in block '{block}' should not be set "
                    f"directly. It should be set at '{where}'",
                )


def _fmt(key: str, value: Any) -> Any:
    """Format a value for an ORCA input block."""
    if key.casefold() in {"auxc", "auxj", "auxjk"}:
        return f'"{value}"'  # ORCA needs quotes for certain keywords
    if isinstance(value, bool):
        return str(value).lower()
    return value


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write ``orca.inp`` and the geometry file for an ORCA calculation.

    Notes:
        - ORCA keywords are case-insensitive; names are written in lowercase.
        - ``True`` adds a simple-input word (``{"tightscf": True}`` writes
          ``! tightscf``); ``False`` or ``None`` leaves it out.
        - A dictionary becomes an input block. ``numgrad`` and ``numfreq`` may be
          either: a dictionary also sets the numerical derivative.

    Raises:
        EncoderError: The model is missing, a keyword duplicates structured data, or
            a keyword is neither a flag, a block, ``maxcore``, nor ``pal``.
    """
    if inp.model is None:
        raise EncoderError("orca input requires a scientific model.")

    kw = {k.casefold(): v for k, v in inp.keywords.items()}
    _validate_keywords(kw)
    flags: list[str] = []
    for key, value in kw.items():
        if key in {"maxcore", "pal"} or isinstance(value, Mapping):
            continue
        if value is True:
            flags.append(key)
        elif value is not False and value is not None:
            raise EncoderError(
                f"ORCA keyword {key!r} must be True (a simple-input word), a "
                f"dictionary (an input block), or False/None; got {value!r}."
            )
    numgrad = "numgrad" in flags or isinstance(kw.get("numgrad"), Mapping)
    numfreq = "numfreq" in flags or isinstance(kw.get("numfreq"), Mapping)
    flags = [f for f in flags if f not in {"numgrad", "numfreq"}]

    lines: list[str] = []
    if "maxcore" in kw:
        lines.append(f"%maxcore {kw['maxcore']}")
    if "pal" in kw:
        lines.append(f"%pal nprocs {kw['pal']} end")
    if lines:
        lines.append("")

    model_line = f"! {inp.model.method}"
    if inp.model.basis:  # composite methods carry no basis
        model_line += f" {inp.model.basis}"
    lines.append(model_line)
    if numgrad:
        lines.append("! numgrad")
    lines.extend(f"! {flag}" for flag in flags)

    runtyp = {
        CalcType.energy: "energy",
        CalcType.gradient: "engrad",
        CalcType.hessian: "numfreq" if numfreq else "freq",
        CalcType.optimization: "opt",
        CalcType.transition_state: "optts",
    }[inp.calctype]
    lines.append(f"! {runtyp}\n")

    blocks = {k: v for k, v in kw.items() if isinstance(v, Mapping)}
    for block, values in blocks.items():
        lines.append(f"%{block}")
        for key, value in values.items():
            lines.append(f"    {key:<{PADDING}} {_fmt(key, value)}")
        lines.append("end")
    if blocks:
        lines.append("")

    lines.append(
        f"* xyzfile {inp.structure.charge} {inp.structure.multiplicity} {XYZ_FILENAME}"
    )
    return invocation(
        inp,
        {"orca.inp": "\n".join(lines) + "\n", XYZ_FILENAME: inp.structure.to_xyz()},
        "orca.inp",
        threads=int(kw.get("pal", 1)),
    )


# ------------------------------------------------------------------ decoding


@codec.parser(
    target="energy", calctypes=[CalcType.energy, CalcType.gradient, CalcType.hessian]
)
def parse_energy(stdout: str) -> float:
    """Parse the final single point energy from ORCA stdout."""
    return float(
        re_search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", stdout).group(1)
    )


@codec.parser(target="gradient", calctypes=[CalcType.gradient])
def parse_gradient(stdout: str) -> list[list[float]]:
    """Parse the Cartesian gradient block from ORCA stdout as an N x 3 list."""
    header = re_search(r"CARTESIAN GRADIENT.*?(?=\d)", stdout, flags=re.DOTALL)
    block_lines = itertools.takewhile(
        lambda line: re.search(r"\d\s*$", line), stdout[header.end() :].splitlines()
    )
    line_regex = r".*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    return [
        list(map(float, re_search(line_regex, line).groups())) for line in block_lines
    ]


@codec.parser(target="gradient", calctypes=[CalcType.hessian], required=False)
def parse_gradient_hessian(stdout: str) -> list[list[float]]:
    """Parse the gradient from a Hessian job; analytic Hessian jobs omit it."""
    return parse_gradient(stdout)


def _hess_file(ctx: Context) -> str:
    """ORCA writes Hessians and vibrations to ``<basename>.hess``."""
    return f"{parse_basename(ctx.stdout or '')}.hess"


def _hess_section(contents: str, name: str) -> str:
    """Return one ``$name`` section of a ``.hess`` file.

    Raises:
        MatchNotFoundError: The file has no such section.
    """
    for section in contents.split("$"):
        if section.split("\n", 1)[0].strip() == name:
            # Comment lines ("# The atoms: label mass x y z") precede some sections.
            lines = [line for line in section.splitlines() if not line.startswith("#")]
            return "\n".join(lines).rstrip() + "\n"
    raise MatchNotFoundError(f"${name}", contents)


def _hess_matrix(section: str) -> list[list[float]]:
    """Read a matrix section written in column blocks, such as ``$hessian``.

    Raises:
        ParserError: The blocks do not form a matrix of the declared size.
    """
    rows_declared = int(section.splitlines()[1].split()[0])
    # Blocks are separated by column-index lines such as '  0  1  2  3 ...'; the
    # final block may list a single column.
    parts = re.split(r"^\s*\d+(?:\s+\d+)*\s*$", section, flags=re.MULTILINE)
    if not len(parts) > 1:
        raise ParserError(f"Failed to parse blocks in section: {section}")
    matrix: list[list[float]] = [[] for _ in range(rows_declared)]
    for block in (part.strip() for part in parts[1:]):
        lines = block.splitlines()
        if len(lines) != rows_declared:
            raise ParserError(
                f"Block line count {len(lines)} does not match dimension "
                f"{rows_declared}: {block}"
            )
        for i, line in enumerate(lines):
            matrix[i].extend(map(float, line.split()[1:]))
    return matrix


@codec.parser(file=_hess_file, target="hessian", calctypes=[CalcType.hessian])
def parse_hessian(contents: str) -> list[list[float]]:
    """Parse the Hessian from ORCA's ``<basename>.hess`` file."""
    try:
        section = _hess_section(contents, "hessian")
    except MatchNotFoundError:
        raise ParserError("Failed to find hessian block in Hessian file.") from None
    return _hess_matrix(section)


def _vibrations(contents: str) -> tuple[list[float], np.ndarray]:
    """Read the vibrational frequencies and modes, without translations and rotations.

    ``$normal_modes`` holds one mode per column, unit-length Cartesian displacements;
    ORCA zeroes the translation and rotation columns and lists their frequencies as 0.
    """
    lines = _hess_section(contents, "vibrational_frequencies").splitlines()
    count = int(lines[1])
    freqs = [float(line.split()[1]) for line in lines[2 : 2 + count]]
    modes = np.array(_hess_matrix(_hess_section(contents, "normal_modes"))).T
    if len(modes) != len(freqs):
        raise ParserError(f"{len(modes)} normal modes for {len(freqs)} frequencies.")
    vibrational = np.linalg.norm(modes, axis=1) > 0
    return [f for f, keep in zip(freqs, vibrational, strict=True) if keep], modes[
        vibrational
    ]


@codec.parser(
    file=_hess_file,
    target="freqs_wavenumber",
    calctypes=[CalcType.hessian],
    required=False,
)
def parse_frequencies(contents: str) -> list[float]:
    """Parse vibrational frequencies (cm^-1, negative for imaginary modes)."""
    return _vibrations(contents)[0]


@codec.parser(
    file="directory",
    target="normal_modes_cartesian",
    calctypes=[CalcType.hessian],
    required=False,
)
def parse_normal_modes(ctx: Context) -> list[list[list[float]]]:
    """Parse normal modes as unit-length Cartesian displacements in the input frame.

    The ``$atoms`` coordinates of the ``.hess`` file are aligned onto the input
    structure; ORCA translates but does not rotate the molecule, so the alignment is a
    check that leaves the modes unchanged. Without ``input_data`` the modes are
    returned as ORCA wrote them.
    """
    contents = ctx.read(_hess_file(ctx))
    _, modes = _vibrations(contents)
    if ctx.input_data is None:
        # Nothing to align against; ORCA does not rotate the molecule.
        return unit_modes(modes.reshape(len(modes), -1, 3))
    atoms = _hess_section(contents, "atoms").splitlines()
    natoms = int(atoms[1])
    coordinates = np.array(
        [line.split()[2:5] for line in atoms[2 : 2 + natoms]], dtype=float
    )
    rotation = rotation_onto(coordinates, ctx.input_data.structure.geometry)
    return unit_modes(modes.reshape(len(modes), natoms, 3) @ rotation.T)


@codec.parser(
    file="directory",
    target="trajectory",
    calctypes=[CalcType.optimization, CalcType.transition_state],
)
def parse_trajectory(ctx: Context) -> list[ProgramOutput]:
    """Build an optimization trajectory from ``<basename>_trj.xyz`` and stdout."""
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    if not ctx.stdout and ctx.failed:
        return []
    stdout = ctx.stdout or ""
    basename = parse_basename(stdout)
    try:
        file = ctx.path(f"{basename}_trj.xyz")
    except FileNotFoundError as exc:
        if ctx.failed:
            return []
        raise ParserError(
            f"Trajectory file does not exist: {basename}_trj.xyz"
        ) from exc

    # The trajectory file carries no charge/multiplicity; take them from the input.
    structures = Structure.open_multi(
        file,
        charge=ctx.input_data.structure.charge,
        multiplicity=ctx.input_data.structure.multiplicity,
    )

    initialization = re_search(
        r"^(.*?\*\*\*\*END\s+OF\s+INPUT\*\*\*\*\s*\n\s*=*)",
        stdout,
        flags=re.MULTILINE | re.VERBOSE | re.DOTALL,
    ).group(1)
    regex = (
        r"GEOMETRY\s*OPTIMIZATION\s*CYCLE\s*\d+.*?"  # cycle line
        r"\*+\n"  # line of stars
        r"(.*?)"  # body (captured)
        r"-+\s*\n\s*ORCA\s+GEOMETRY\s+RELAXATION\s+STEP"  # footer
    )
    per_step = re.findall(regex, stdout, flags=re.DOTALL)
    if not per_step:
        raise MatchNotFoundError(regex, stdout)

    trajectory: list[ProgramOutput] = []
    for structure, step_stdout in zip(structures, per_step, strict=False):
        parsed = ctx.decode(CalcType.gradient, stdout=initialization + step_stdout)
        assert isinstance(parsed, SinglePointData)  # for mypy
        values = parsed.model_dump()
        values["energy"] = structure.extras[Structure._xyz_comment_key][-1]
        results = SinglePointData(**values)
        step_input = ProgramInput.model_validate(
            {
                **ctx.input_data.model_dump(),
                "calctype": CalcType.gradient
                if results.gradient is not None
                else CalcType.energy,
                "structure": structure,
            }
        )
        trajectory.append(
            ProgramOutput(
                input_data=step_input,
                success=True,
                results=results,
                execution=ExecutionInfo(scratch_dir=ctx.directory),
            )
        )
    return trajectory


@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    """Parse the version string from ORCA stdout."""
    return re_search(r"Program Version (\d+\.\d+\.\d+)", stdout).group(1)


@codec.parser(target="calcinfo_natoms", required=False)
def parse_natoms(stdout: str) -> int:
    """Parse the number of atoms from ORCA stdout."""
    return int(re_search(r"Number of atoms\s*...\s*(\d+)", stdout).group(1))


def parse_basename(stdout: str) -> str:
    """Parse the output file basename from ORCA stdout."""
    return Path(re_search(r"NAME\s+=\s+(.*)", stdout).group(1)).stem
