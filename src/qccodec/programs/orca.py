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
  ORCA runs serially unless ``pal`` (or ``threads``) is given, whatever the
  calculation's thread count: its parallel runs need OpenMPI's ``mpirun`` (without
  it they fail, kind ``environment``), so they are asked for, never assumed. A count
  smaller than ``pal`` lowers it, with a ``resource_capped`` warning.
- A keyword whose value is a dictionary becomes an input block:
  ``{"scf": {"maxiter": 200}}`` writes ``%scf maxiter 200 end``. Keys are kept as
  written; ORCA ignores case.
- The structure is written to ``geometry.xyz`` with its charge and multiplicity.
  Setting ``coords``, or ``method``/``basis`` inside a block, is an error.
- ``cmdline_args`` are appended to the ``orca`` command.

Calculation types: ``energy``, ``gradient``, ``hessian`` (also the gradient where ORCA
gives it: ``keywords={"engrad": True}`` asks for it beside ``! freq``, which also
tightens ORCA's SCF convergence; and the harmonic frequencies and normal modes from
``<basename>.hess``:
unit-length Cartesian displacements without translations and rotations, imaginary
frequencies negative), ``optimization``, and ``transition_state`` (``OptTS``; give ORCA a
starting Hessian, e.g. ``{"geom": {"calc_hess": True}}``). ORCA keeps the input's
coordinates and atom order. No input field carries atom indices.

Keywords are checked as far as the codec can: values must be ``True``, a dictionary,
``False``/``None``, or ``maxcore``/``pal`` numbers. ORCA's simple-input words, blocks
and block settings are too many to list, but ORCA refuses an unknown one itself before
calculating anything, and the codec reports that as kind ``input``, code
``unknown_key`` (a misspelled method or basis set is ``input`` without a code).

Failures and their kinds: an SCF that does not converge is ``convergence`` (code
``scf_not_converged``), with the energy of its last iteration; coupled-perturbed SCF
equations (for analytic Hessians) that end with perturbations above their tolerance
are ``convergence`` (``response_not_converged``), with the results ORCA wrote, since
ORCA itself carries on with them; an optimization that runs out of cycles is
``convergence`` (``max_iter``), with its trajectory attached (ORCA itself reports it as
a normal termination); a multiplicity the number of electrons cannot have is
``input``; a module out of memory is ``resource`` (``host_memory``), with
``required_gib`` and ``available_gib`` in the error's ``details`` where ORCA prints
what a step needed against ``%maxcore`` (the SCF's estimate, DLPNO's minimum), in
the terms of ORCA's memory keyword: the ``%maxcore`` with which the step fits and the
one it ran with, each times the processes (ORCA's per-process MB of 2^20 bytes), in
GiB. Any other error termination is ``internal``, with ORCA's error line.

ORCA stops when the SCF does not converge unless ``%scf ConvIgnored true`` lets it
carry on with the unconverged SCF; its results then come with the warning
``loosely_converged``, whose message gives the last iteration's residuals.

Run ORCA by its full path: its parallel runs need it, and on many Linux desktops
``/usr/bin/orca`` is the GNOME screen reader. qccompute always runs the absolute path;
put ORCA first on ``PATH`` or set ``QCCODEC_ORCA_EXECUTABLE``.
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
    Frame,
    ProgramInput,
    ProgramWarning,
    SinglePointData,
    Structure,
)

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..program import CodecProgram
from ..utils import re_search
from ._common import invocation, rotation_onto, unit_modes, vibrations

codec = Codec(
    "orca",
    calctypes={
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
        CalcType.transition_state,
    },
    # Matched against the failure hook's messages below.
    errors=[
        (r"does not recognize the method or basis", "input", None),
        (r"does not recognize the simple-input", "input", "unknown_key"),
        (r"has no input block|has no setting", "input", "unknown_key"),
        (r"the SCF did not converge", "convergence", "scf_not_converged"),
        (r"CP-SCF equations did not converge", "convergence", "response_not_converged"),
        (r"optimization reached its maximum", "convergence", "max_iter"),
        (r"multiplicity .* impossible", "input", None),
        (r"need OpenMPI's mpirun", "environment", None),
        (r"ORCA ran out of memory", "resource", "host_memory"),
    ],
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
        f"* xyzfile {inp.require_structure().charge} {inp.require_structure().multiplicity} {XYZ_FILENAME}"
    )
    return invocation(
        inp,
        {
            "orca.inp": "\n".join(lines) + "\n",
            XYZ_FILENAME: inp.require_structure().to_xyz(),
        },
        "orca.inp",
        threads=int(kw["pal"]) if "pal" in kw else None,
    )


# ------------------------------------------------------------------ decoding


@codec.failure
def failure(ctx: Context) -> str | tuple[str, dict[str, Any]] | None:
    """Explain why ORCA stopped, including what it reports with exit status 0.

    ORCA refuses unknown simple-input words, input blocks and block settings itself,
    before calculating anything. It exits 0 when the SCF fails to converge and when
    an optimization runs out of cycles (``ORCA TERMINATED NORMALLY`` after the
    latter), so both are read from the output. A module out of memory comes with
    the memory it needed where ORCA prints it (`out_of_memory`).
    """
    stdout = ctx.stdout or ""
    if re.search(r"mpirun: (command )?not found", stdout):
        return (
            "ORCA's parallel runs (pal) need OpenMPI's mpirun on PATH; it was not "
            "found."
        )
    words = re.search(
        r"UNRECOGNIZED OR DUPLICATED KEYWORD\(S\) IN SIMPLE INPUT LINE\s*\n(.*?)\n\s*!{4,}",
        stdout,
        flags=re.DOTALL,
    )
    if words:
        unknown = words.group(1).split()
        model = ctx.input_data.model if ctx.input_data is not None else None
        named = (
            {w.upper() for w in f"{model.method} {model.basis or ''}".split()}
            if model is not None
            else set()
        )
        if named & set(unknown):
            return (
                "ORCA does not recognize the method or basis set word(s) "
                f"{', '.join(unknown)} (or one is given twice)."
            )
        return (
            "ORCA does not recognize the simple-input keyword(s) "
            f"{', '.join(unknown)} (or one is given twice)."
        )
    block = re.search(r"ERROR: Unknown identifier % ?(\S+)", stdout)
    if block:
        return f"ORCA has no input block %{block.group(1).lower()}."
    setting = re.search(
        r"Unknown identifier in (\w+) block.*?\n\s*Last token : (\S+)", stdout
    )
    if setting:
        return (
            f"ORCA has no setting {setting.group(2).lower()} in the "
            f"%{setting.group(1).lower()} block."
        )
    multiplicity = re.search(r"Error : (multiplicity .*? impossible)", stdout)
    if multiplicity:
        return f"ORCA: {multiplicity.group(1)}."
    if re.search(r"The optimization did not converge but reached the maximum", stdout):
        return "ORCA: the optimization reached its maximum number of cycles."
    # Only a Hessian is made of the response; a transition-state search's starting
    # Hessian (calc_hess) is a guess the search does not depend on.
    response = (
        unconverged_response(stdout) if ctx.calctype == CalcType.hessian else None
    )
    if response is not None:
        return response
    stopped = re.search(r"ORCA finished by error termination in (\w+)", stdout)
    if stopped:
        memory = out_of_memory(stdout, stopped.group(1))
        if memory is not None:
            return memory
        cycles = re.search(r"SCF NOT CONVERGED AFTER\s+(\d+) CYCLES", stdout)
        if cycles:
            return f"ORCA: the SCF did not converge in {cycles.group(1)} cycles."
        errors = re.findall(r"^(?:\[file [^\]]*\]: )?\s*(Error.*\S)", stdout, re.M)
        reason = f": {errors[0]}" if errors else ""
        return f"ORCA stopped with an error in {stopped.group(1)}{reason}"
    return None


#: An iteration of ORCA's CP-SCF solver: ``ITERATION   4:   ||err||_max =
#: 1.8527e-04 (   0.0 sec   9/  9 done)``, the last two the perturbations converged.
_CPSCF_ITERATION = re.compile(
    r"ITERATION\s+(\d+):\s+\|\|err\|\|_max =\s+(\S+)\s+\(.*?(\d+)/\s*(\d+) done\)"
)


def unconverged_response(stdout: str) -> str | None:
    """Read CP-SCF equations that ended with perturbations above their tolerance.

    ORCA's solver stops at its maximum number of iterations and goes on with the
    response it has, without a word, so the last iteration of each solve is checked:
    it counts the perturbations that met the tolerance.

    Returns:
        The failure's message, or None when every solve converged.
    """
    for block in re.split(r"CP-SCF DRIVER", stdout)[1:]:
        iterations = _CPSCF_ITERATION.findall(block.split("CP-SCF equations solved")[0])
        if not iterations:
            continue
        step, error, done, total = iterations[-1]
        if int(done) < int(total):
            tolerance = re.search(r"Convergence Tolerance\s+\.+\s+(\S+)", block)
            limit = f" {tolerance.group(1)}" if tolerance else ""
            return (
                f"ORCA's CP-SCF equations did not converge: {int(total) - int(done)} "
                f"of {total} perturbations were above the tolerance{limit} after "
                f"{int(step) + 1} iterations (largest error {error})."
            )
    return None


#: How ORCA modules report the memory a step needed against what it had, in ORCA's MB
#: (2^20 bytes) per process: the SCF's estimate, DLPNO's minimum, and the batching
#: of the correlated methods.
_MEMORY_NEEDS = (
    r"Memory available for [^:\n]*:\s*(?P<have>[\d.]+) MB\s*\n"
    r"\s*Memory needed \(estimated\)\s*:\s*(?P<need>[\d.]+) MB",
    r"MINIMUM REQUIRED:\s*(?P<need>[\d.]+) MB\s*\n\s*MAXCORE:\s*(?P<have>[\d.]+) MB",
    r"Current MaxCore: (?P<have>\d+) MB -> Minimum MaxCore: (?P<need>\d+) MB",
)
_OUT_OF_MEMORY = (
    r"not enough memory|out of memory|insufficient memory|present MaxCore"
    r"|exceeds? MaxCore|increase MaxCore"
)


def out_of_memory(stdout: str, module: str) -> tuple[str, dict[str, Any]] | None:
    """Read a module's out-of-memory error, with what it needed where ORCA says.

    ORCA gives ``%maxcore`` to each process, so the calculation's need and budget
    are the per-process figures times the processes it ran (``Program running with
    N parallel MPI-processes``; one when it ran serially).

    Args:
        stdout: ORCA's output.
        module: The module that stopped (``LEANSCF``, ``MDCI``).

    Returns:
        The failure's message and details (``required_gib`` and ``available_gib``
        when the module printed them), or None when the module did not run out of
        memory.
    """
    for pattern in _MEMORY_NEEDS:
        found = re.search(pattern, stdout)
        if found:
            need, have = float(found["need"]), float(found["have"])
            ranks = re.findall(r"Program running with\s+(\d+) parallel MPI", stdout)
            n = int(ranks[-1]) if ranks else 1
            message = (
                f"ORCA ran out of memory in {module}: it needs {need:g} MB per process "
                f"and %maxcore gives it {have:g} MB"
                + (f" ({n} processes)" if n > 1 else "")
                + "; raise maxcore."
            )
            details = {
                "required_gib": round(need * n / 1024, 9),
                "available_gib": round(have * n / 1024, 9),
            }
            return message, details
    error = re.search(rf"^.*(?:{_OUT_OF_MEMORY}).*$", stdout, re.M | re.I)
    if error is None:
        return None
    line = error.group(0)
    start = re.search(r"\berror\b", line, re.I)  # after any "[file ...]:" prefix
    reason = line[start.start() if start else 0 :].strip()
    return f"ORCA ran out of memory in {module}: {reason}", {}


@codec.parser(
    target="energy", calctypes=[CalcType.energy, CalcType.gradient, CalcType.hessian]
)
def parse_energy(stdout: str) -> float:
    """Parse the final single point energy from ORCA stdout.

    An SCF that did not converge has none; its last iteration's energy is returned.
    """
    final = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", stdout)
    if final is None and "SCF NOT CONVERGED AFTER" in stdout:
        last = last_scf_iteration(stdout)
        if last is not None:
            return last[0]
    return float(
        re_search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", stdout).group(1)
    )


def last_scf_iteration(stdout: str) -> tuple[float, dict[str, str]] | None:
    """The energy (Eh) and residuals of the last SCF iteration ORCA printed.

    ORCA's DIIS and SOSCF tables print a row per iteration under a header naming the
    columns after the energy (``Delta-E  RMSDP  MaxDP  DIISErr ...``). None when the
    last SCF ran in a solver without such a table (TRAH).
    """
    columns: list[str] = []  # empty outside a table
    last: tuple[float, dict[str, str]] | None = None
    end = last_end = 0
    for line in stdout.splitlines(keepends=True):
        end += len(line)
        header = re.match(r"Iteration\s+Energy \(Eh\)\s+(.*)", line)
        row = re.match(r"\s+\d+\s+(-?\d+\.\d{10,})\s+(-?\d\.\d+e[-+]\d+\s.*)", line)
        if header:
            columns = header.group(1).split()
        elif columns and row:
            values = dict(zip(columns, row.group(2).split(), strict=False))
            last = (float(row.group(1)), values)
            last_end = end
        elif "***" not in line and not line.lstrip().startswith("-"):
            columns = []  # the table ended
    if last is None or "TRAH" in stdout[last_end:]:
        return None
    return last


@codec.warning
def warnings(ctx: Context) -> list[ProgramWarning]:
    """Warn ``loosely_converged`` when ORCA carried on with an unconverged SCF.

    ORCA does that only with ``%scf ConvIgnored true``; the message gives the last
    iteration's residuals.
    """
    stdout = ctx.stdout or ""
    accepted = len(re.findall(r"the SCF did not converge, but SCFConvIgnored", stdout))
    if not accepted:
        return []
    last = last_scf_iteration(stdout)
    residuals = ""
    if last is not None:
        named = [
            f"{k} {v}" for k, v in last[1].items() if k not in ("Damp", "Time(sec)")
        ]
        residuals = f"; the last iteration printed had {', '.join(named)}"
    count = f"{accepted} SCFs" if accepted > 1 else "the SCF"
    return [
        ProgramWarning(
            code="loosely_converged",
            message=(
                f"ORCA did not converge {count} and carried on with the unconverged "
                f"result because %scf ConvIgnored is true{residuals}."
            ),
        )
    ]


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
    """Parse the gradient of a Hessian job that asked for one (``engrad``); else none."""
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


def parse_frequencies(contents: str) -> list[float]:
    """Parse vibrational frequencies (cm^-1, negative for imaginary modes)."""
    return _vibrations(contents)[0]


@codec.parser(
    file="directory",
    target="vibrational_analysis",
    calctypes=[CalcType.hessian],
    required=False,
)
def parse_vibrations(ctx: Context) -> dict[str, Any]:
    """The vibrational analysis of the ``.hess`` file, modes in the input frame."""
    if ctx.input_data is None:  # the masses come from the input's elements
        raise FileNotFoundError("input_data")
    return vibrations(
        parse_frequencies(ctx.read(_hess_file(ctx))),
        parse_normal_modes(ctx),
        ctx.input_data.require_structure(),
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
    rotation = rotation_onto(coordinates, ctx.input_data.require_structure().geometry)
    return unit_modes(modes.reshape(len(modes), natoms, 3) @ rotation.T)


@codec.parser(
    file="directory",
    target="frames",
    calctypes=[CalcType.optimization, CalcType.transition_state],
)
def parse_trajectory(ctx: Context) -> list[Frame]:
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
        charge=ctx.input_data.require_structure().charge,
        multiplicity=ctx.input_data.require_structure().multiplicity,
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

    trajectory: list[Frame] = []
    for structure, step_stdout in zip(structures, per_step, strict=False):
        parsed = ctx.decode(CalcType.gradient, stdout=initialization + step_stdout)
        assert isinstance(parsed, SinglePointData)  # for mypy
        values = parsed.model_dump()
        values["energy"] = structure.extras[Structure._xyz_comment_key][-1]
        results = SinglePointData(**values)
        trajectory.append(
            Frame(
                structure=structure,
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


@codec.parser(target=("provenance", "program_commit"), required=False)
def parse_commit(stdout: str) -> str:
    """Parse the build's commit under the version line: ``(GIT: $679e74b$)``.

    ORCA prints it from 6.1; 6.0 names no commit.
    """
    return re_search(r"\(GIT: \$(\w+)\$\)", stdout).group(1)


@codec.parser(target="calcinfo_natoms", required=False)
def parse_natoms(stdout: str) -> int:
    """Parse the number of atoms from ORCA stdout."""
    return int(re_search(r"Number of atoms\s*...\s*(\d+)", stdout).group(1))


def parse_basename(stdout: str) -> str:
    """Parse the output file basename from ORCA stdout."""
    return Path(re_search(r"NAME\s+=\s+(.*)", stdout).group(1)).stem


def cap_threads(inp: ProgramInput, threads: int) -> ProgramInput:
    """The input with ``pal`` lowered to ``threads``."""
    return inp.model_copy(update={"keywords": {**inp.keywords, "pal": threads}})


codec.cap_threads = cap_threads

#: The program qccompute runs (the ``qccompute.programs`` entry point 'orca').
#: Serial unless asked: parallel ORCA needs OpenMPI's mpirun, which may be absent.
program = CodecProgram(codec, threads=1)
