"""TeraChem: write ``tc.in``, run ``terachem tc.in``, and parse stdout and scratch files.

Inputs map to TeraChem as follows:

- ``model.method`` and ``model.basis`` become ``method`` and ``basis``; the structure's
  charge and multiplicity become ``charge`` and ``spinmult``. Setting any of these, or
  ``run``, in ``keywords`` is an error.
- Every other keyword becomes a ``tc.in`` line, ``key value``, so TeraChem's own
  keyword names apply (``scf``, ``convthre``, ``dftd``, ``pcm``, ``epsilon``, ...). Values
  are written in lowercase, which suits TeraChem's booleans and options but means
  case-sensitive values such as file names must be lowercase. TeraChem ignores keywords
  it does not know, printing only "Unrecognized or unnecessary keywords" in its output,
  so check the logs when a setting seems to have no effect.
- ``cmdline_args`` are appended to the ``terachem`` command.

Calculation types: ``energy``, ``gradient``, ``hessian`` (``run frequencies``; the
Hessian, gradient, frequencies, and normal modes, the last two from
``scr.geometry/Frequencies.dat``), and ``optimization``, which requires
``new_minimizer yes`` in ``keywords``. TeraChem refuses Hessians away from a stationary
point unless ``mincheck false`` is set. TeraChem keeps the input's coordinates and atom
order.

Optimizations take one keyword that is not a ``tc.in`` line: ``constraints``, the same
dictionary the geomeTRIC codec takes, e.g. ``{"freeze": [{"type": "xyz", "indices":
[0, 1]}], "set": [{"type": "distance", "indices": [0, 2], "value": 2.8}]}``. Types are
``xyz``, ``distance``, ``angle``, and ``dihedral``; ``set`` values are in bohr for
distances and degrees for angles. The codec writes them as TeraChem's
``$constraint_freeze`` and ``$constraint_set`` groups, converting the 0-based
``indices`` to TeraChem's 1-based atom numbers and distances to angstrom. Scans are not
supported. ``indices`` is the only input field that carries atom indices.

The orbital files ``c0`` (or ``ca0``/``cb0``) can seed the next calculation: qccompute's
``propagate_wfn=True`` does this between geomeTRIC steps. A run is a failure unless
stdout ends with ``Job finished:``.
"""

from __future__ import annotations

import re
from typing import Any

from qcdata import (
    CalcType,
    ExecutionInfo,
    Frame,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
    Structure,
)
from qcref import constants

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..utils import re_finditer, re_search
from ._common import invocation, unit_modes

codec = Codec(
    "terachem",
    calctypes={
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
    },
)


def version_key(version: str) -> str:
    """Compare TeraChem builds by release date.

    ``v1.9-2022.03-dev [hash]`` becomes ``2022.3``, so parsers declare ranges such
    as ``versions=">=2023.8"``. Builds without a date fall back to their number
    (``v1.5K`` becomes ``1.5``), which sorts before every dated build.
    """
    dated = re.search(r"(\d{4})\.(\d{2})", version)
    if dated:
        return f"{dated.group(1)}.{int(dated.group(2))}"
    numbered = re.search(r"(\d+(?:\.\d+)*)", version)
    if numbered is None:
        raise ValueError(f"No version number in {version!r}")
    return numbered.group(1)


codec.version_key = version_key

XYZ_FILENAME = "geometry.xyz"
PADDING = 20  # padding between keyword and value in tc.in


# ------------------------------------------------------------------ encoding


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write ``tc.in`` and the geometry file for a TeraChem calculation.

    Raises:
        EncoderError: The model is missing, an unsupported optimizer is requested,
            or a keyword duplicates structured input data.
    """
    if inp.model is None:
        raise EncoderError("terachem input requires a scientific model.")

    if inp.calctype == CalcType.hessian:
        run = "frequencies"
    elif inp.calctype == CalcType.optimization:
        run = "minimize"
        if not inp.keywords.get("new_minimizer", "no") == "yes":
            raise EncoderError(
                "Only the new_minimizer is supported for optimizations. Add "
                "'new_minimizer': 'yes' to the keywords."
            )
    elif inp.calctype == CalcType.transition_state:
        run = "ts"
    else:
        run = inp.calctype.value

    lines = [
        f"{'run':<{PADDING}} {run}",
        f"{'coordinates':<{PADDING}} {XYZ_FILENAME}",
        f"{'charge':<{PADDING}} {inp.require_structure().charge}",
        f"{'spinmult':<{PADDING}} {inp.require_structure().multiplicity}",
        f"{'method':<{PADDING}} {inp.model.method}",
        f"{'basis':<{PADDING}} {inp.model.basis}",
    ]

    # Values that belong on structured qcdata fields, not in keywords.
    non_keywords = {
        "charge": ".structure.charge",
        "spinmult": ".structure.multiplicity",
        "run": ".calctype",
        "basis": ".model.basis",
        "method": ".model.method",
    }
    keywords = dict(inp.keywords)
    constraints = keywords.pop("constraints", None)
    for key, value in keywords.items():
        if key in non_keywords:
            raise EncoderError(
                f"Keyword '{key}' should not be set as a keyword. It "
                f"should be set at '{non_keywords[key]}'",
            )
        lines.append(f"{key:<{PADDING}} {str(value).lower()}")  # lowercase booleans
    if constraints:
        if inp.calctype != CalcType.optimization:
            raise EncoderError("constraints apply to optimizations only.")
        lines.extend(["end", "", *constraint_groups(constraints)])

    return invocation(
        inp,
        {
            "tc.in": "\n".join(lines) + "\n",
            XYZ_FILENAME: inp.require_structure().to_xyz(),
        },
        "tc.in",
        wavefunction_files=("scr*/c0", "scr*/ca0", "scr*/cb0"),
        gpus=_gpu_count(keywords.get("gpus")),
    )


def _gpu_count(value: object) -> int:
    """Return how many GPUs ``gpus`` asks for (``2`` or ``"2 0 1"``); 1 by default."""
    if value is None:
        return 1
    return int(str(value).split()[0])


# geomeTRIC constraint type -> (TeraChem name, number of atoms or None for any).
CONSTRAINT_TYPES: dict[str, tuple[str, int | None]] = {
    "xyz": ("xyz", None),
    "distance": ("bond", 2),
    "angle": ("angle", 3),
    "dihedral": ("dihedral", 4),
}


def constraint_groups(constraints: dict[str, Any]) -> list[str]:
    """Write geomeTRIC-style constraints as ``$constraint_freeze``/``$constraint_set``.

    Args:
        constraints: ``{"freeze": [...], "set": [...]}``, each entry
            ``{"type": "xyz" | "distance" | "angle" | "dihedral", "indices": [...]}``
            with 0-based atom indices; ``set`` entries add ``"value"`` (bohr for
            distances, degrees for angles and dihedrals).

    Returns:
        ``tc.in`` lines with 1-based atom indices and distances in angstrom.

    Raises:
        EncoderError: An unknown group or type, a wrong number of atoms, or a missing
            value.
    """
    lines: list[str] = []
    for group, entries in constraints.items():
        if group not in ("freeze", "set"):
            raise EncoderError(
                f"Unknown constraint group {group!r}; use 'freeze' or 'set' (scans "
                "are not supported)."
            )
        lines.append(f"$constraint_{group}")
        for entry in entries:
            kind = str(entry.get("type", "")).lower()
            if kind not in CONSTRAINT_TYPES:
                raise EncoderError(
                    f"Unknown constraint type {entry.get('type')!r}; use one of "
                    f"{sorted(CONSTRAINT_TYPES)}."
                )
            name, count = CONSTRAINT_TYPES[kind]
            indices = [int(i) for i in entry.get("indices", [])]
            if not indices or (count is not None and len(indices) != count):
                raise EncoderError(
                    f"A {kind} constraint takes {count or 'one or more'} atom indices, "
                    f"not {indices}."
                )
            if kind == "xyz":
                atoms = ",".join(str(i + 1) for i in indices)
            else:
                atoms = "_".join(str(i + 1) for i in indices)
            if group == "set":
                if "value" not in entry or kind == "xyz":
                    raise EncoderError(
                        f"A set constraint needs a distance, angle, or dihedral and a "
                        f"value: {entry}."
                    )
                value = float(entry["value"])
                if kind == "distance":
                    value *= constants.BOHR_TO_ANGSTROM
                lines.append(f"{name} {value:.10g} {atoms}")
            else:
                lines.append(f"{name} {atoms}")
        lines.append("$end")
    return lines


def propagate_wfn(output: ProgramOutput, inp: ProgramInput) -> ProgramInput:
    """Place a previous calculation's orbitals on the next input as its SCF guess.

    Uses ``c0`` (restricted) or ``ca0``/``cb0`` (unrestricted) retained in
    ``output.results.files`` and sets the ``guess`` keyword accordingly.

    Raises:
        EncoderError: The output holds no orbital files.
    """
    matches = {
        name.rsplit("/", 1)[-1]: value
        for name, value in output.results.files.items()
        if name.rsplit("/", 1)[-1] in {"c0", "ca0", "cb0"}
    }
    names = ["c0"] if "c0" in matches else ["ca0", "cb0"]
    if not all(name in matches for name in names):
        raise EncoderError("Could not find c0 or ca0/cb0 wavefunction files in output.")
    values = inp.model_dump()
    values["files"].update({name: matches[name] for name in names})
    values["keywords"]["guess"] = " ".join(names)
    return ProgramInput.model_validate(values)


codec.propagate_wfn = propagate_wfn


# ------------------------------------------------------------------ decoding


@codec.failure
def failure(ctx: Context) -> str | None:
    """TeraChem prints ``Job finished:`` only when a calculation completed."""
    if ctx.stdout is not None and not calculation_succeeded(ctx.stdout):
        return "TeraChem did not finish: no 'Job finished:' line in its output."
    return None


@codec.parser(
    target="energy", calctypes=[CalcType.energy, CalcType.gradient, CalcType.hessian]
)
def parse_energy(stdout: str) -> float:
    """Parse the final energy from TeraChem stdout.

    Works on frequency files containing many energy values because ``re.search``
    returns the first result.
    """
    regex = r"FINAL ENERGY: (-?\d+(?:\.\d+)?)"
    return float(re_search(regex, stdout).group(1))


@codec.parser(target="gradient", calctypes=[CalcType.gradient, CalcType.hessian])
def parse_gradient(stdout: str) -> list[list[float]]:
    """Parse the gradient from TeraChem stdout as an N x 3 list.

    Works for excited state gradients too because TeraChem prints the targeted
    gradient as the "regular" gradient.
    """
    regex = (
        r"(?<=dE\/dX\s{12}dE\/dY\s{12}dE\/dZ\n)"  # Just after the header line
        r"[\d\.\-\s]"  # the gradient block itself
        r"+(?=\n(?:--|-=|\Z))"  # stop at -- (grad), -= (opt), or end of file
    )
    values = [float(val) for val in re_search(regex, stdout).group(0).split()]
    return [values[i : i + 3] for i in range(0, len(values), 3)]


@codec.parser(target="hessian", calctypes=[CalcType.hessian])
def parse_hessian(stdout: str) -> list[list[float]]:
    """Parse the Hessian matrix from TeraChem stdout in one pass.

    Raises:
        MatchNotFoundError: No Hessian data is found.
        ParserError: The extracted numbers do not form a square matrix.
    """
    regex = r"\s+(?P<atom_number>\d+)\s(?P<vals>(?:\s-?\d.\d{15}e[+-]\d{2})+)"
    hessian: list[list[float]] = []
    for match in re_finditer(regex, stdout):
        row = int(match.group("atom_number")) - 1
        if len(hessian) <= row:
            hessian.extend([[] for _ in range(row - len(hessian) + 1)])
        hessian[row].extend(float(val) for val in match.group("vals").split())
    for i, row_values in enumerate(hessian):
        if len(row_values) != len(hessian):
            raise ParserError(
                f"Hessian matrix is not square: row {i} has {len(row_values)} "
                f"elements, expected {len(hessian)}."
            )
    return hessian


def _frequency_blocks(contents: str) -> tuple[list[float], list[list[list[float]]]]:
    """Read every mode from ``Frequencies.dat``.

    TeraChem lists the vibrational modes in blocks of columns: a row of mode numbers,
    a row of frequencies (cm^-1, negative for imaginary modes), a dashed rule, then
    three rows per atom (x, y, z, the first prefixed with the atom number). The modes
    are Cartesian displacements (not mass-weighted).

    Returns:
        The frequencies and the modes, (n_modes, n_atoms, 3).

    Raises:
        ParserError: The file holds no modes or a block is malformed.
    """
    natoms = int(re_search(r"Number Atoms:\s+(\d+)", contents).group(1))
    lines = contents.splitlines()
    freqs: list[float] = []
    modes: list[list[list[float]]] = []
    i = 0
    while i < len(lines):
        fields = lines[i].split()
        if not fields or not all(f.isdigit() for f in fields) or i + 2 >= len(lines):
            i += 1
            continue
        ncol = len(fields)
        block_freqs = [float(f) for f in lines[i + 1].split()]
        rows = [line.split() for line in lines[i + 3 : i + 3 + 3 * natoms]]
        values = [row[-ncol:] for row in rows]
        if len(block_freqs) != ncol or any(len(row) != ncol for row in values):
            raise ParserError(f"Malformed Frequencies.dat block at line {i + 1}.")
        for column in range(ncol):
            displacement = [float(row[column]) for row in values]
            modes.append([displacement[a : a + 3] for a in range(0, 3 * natoms, 3)])
        freqs.extend(block_freqs)
        i += 3 + 3 * natoms
    if not freqs:
        raise ParserError("Frequencies.dat holds no vibrational modes.")
    return freqs, modes


@codec.parser(
    file="scr*/Frequencies.dat",
    target="freqs_wavenumber",
    calctypes=[CalcType.hessian],
    required=False,
)
def parse_frequencies(contents: str) -> list[float]:
    """Parse vibrational frequencies (cm^-1) from ``scr.*/Frequencies.dat``."""
    return _frequency_blocks(contents)[0]


@codec.parser(
    file="scr*/Frequencies.dat",
    target="normal_modes_cartesian",
    calctypes=[CalcType.hessian],
    required=False,
)
def parse_normal_modes(contents: str) -> list[list[list[float]]]:
    """Parse normal modes (n_modes, n_atoms, 3) from ``scr.*/Frequencies.dat``.

    TeraChem prints each mass-weighted eigenvector divided by the square root of the
    atomic masses; the codec scales each mode to unit length, as qcdata defines them.
    """
    return unit_modes(_frequency_blocks(contents)[1])


@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    """Parse the version plus git commit, matching ``terachem --version``.

    Example: ``'v1.9-2022.03-dev [4daa16dd21e78d64be5415f7663c3d7c2785203c]'``
    """
    return f"{parse_terachem_version(stdout)} [{parse_version_control_details(stdout)}]"


@codec.parser(target="calcinfo_natoms")
def parse_natoms(stdout: str) -> int:
    """Parse the number of atoms from TeraChem stdout."""
    return int(re_search(r"Total atoms:\s*(\d+)", stdout).group(1))


@codec.parser(target="calcinfo_nmo")
def parse_nmo(stdout: str) -> int:
    """Parse the number of molecular orbitals from TeraChem stdout."""
    return int(re_search(r"Total orbitals:\s*(\d+)", stdout).group(1))


@codec.parser(
    target=("extras", "excited_states"),
    calctypes=[CalcType.energy, CalcType.gradient],
    required=False,
)
def parse_excited_states(stdout: str) -> list[dict]:
    """Parse excited state information from a TDDFT TeraChem stdout.

    Excitation energies are converted from eV to Hartree.
    """
    regex = (
        r"^\s*(?:\d+)\s+(?P<energy>-?\d+\.\d+)\s+"
        r"(?P<exc_energy>-?\d+\.\d+)\s+"
        r"(?P<osc_strength>-?\d+\.\d+)\s+"
        r"(?P<s_squared>-?\d+\.\d+)\s+"
        r"(?P<max_ci_coeff>-?\d+\.\d+)\s+"
        r"(?P<excitation>\d+\s+->\s+\d+\s+:\s+\w+\s+->\s+\w+)$"
    )
    excited_states = []
    for match in re_finditer(regex, stdout, re.MULTILINE):
        state = match.groupdict()
        for key, value in state.items():
            if key != "excitation":
                state[key] = float(value)
        state["exc_energy"] *= constants.EV_TO_HARTREE  # type: ignore[operator]
        excited_states.append(state)
    return excited_states


@codec.parser(file="directory", target="frames", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[Frame]:
    """Build an optimization trajectory from the scratch trajectory file and stdout.

    Each step's stdout is decoded as a gradient calculation with this codec. Builds
    through at least 2023.09 write ``scr*/optim.xyz`` with the energy first in each
    comment line; newer builds (2026.10 confirmed) write ``scr*/optim_trajectory.xyz``
    with comments such as ``frame 0 SCF Energy = -7.49e+01 Hartree, ...``.
    """
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    if not ctx.stdout and ctx.failed:
        return []
    stdout = ctx.stdout or ""

    try:
        trajectory_file = ctx.path("scr*/optim_trajectory.xyz")
    except FileNotFoundError:
        trajectory_file = ctx.path("scr*/optim.xyz")
    structures = Structure.open_multi(
        trajectory_file,
        charge=ctx.input_data.require_structure().charge,
        multiplicity=ctx.input_data.require_structure().multiplicity,
    )

    # Everything before the "0 additional frames" banner is shared initialization.
    regex = r"""^(.*?)                # group-1 = everything before the banner
        (?=                       # look-ahead, do NOT consume banner
            ^-{55}\s*\r?\n        # 55 dashes
            0\ additional\ frames\ found\ in\ \S+\.xyz\s*\r?\n
            -{55}                 # 55 dashes
        )
    """
    match = re.search(regex, stdout, flags=re.MULTILINE | re.VERBOSE | re.DOTALL)
    if not match:
        raise MatchNotFoundError(regex, stdout)
    initialization = match.group(1)

    regex = (
        r"-=#=-\s+\(We'll Be Right Back\)\s+-=#=-\n(.*?)"
        r"-=#=-\s+Now Returning to Optimizer\s+-=#=-"
    )
    per_step = re.findall(regex, stdout, flags=re.DOTALL)
    if not per_step:
        raise MatchNotFoundError(regex, stdout)

    trajectory: list[Frame] = []
    for structure, step_stdout in zip(structures, per_step, strict=False):
        step_logs = initialization + step_stdout
        parsed = ctx.decode(CalcType.gradient, stdout=step_logs)
        assert isinstance(parsed, SinglePointData)  # for mypy
        values = parsed.model_dump()
        values["energy"] = _comment_energy(structure)
        results = SinglePointData(**values)
        trajectory.append(
            Frame(
                structure=structure,
                success=True,
                results=results,
                execution=ExecutionInfo(scratch_dir=ctx.directory),
                logs=step_logs,
            )
        )
    return trajectory


# ------------------------------------------------------- unregistered helpers


def _comment_energy(structure: Structure) -> float:
    """Read the energy from a TeraChem trajectory frame's xyz comment line."""
    comment = " ".join(structure.extras[Structure._xyz_comment_key])
    return float(re_search(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", comment).group(0))


def parse_version_control_details(stdout: str) -> str:
    """Parse the git commit or Hg version from TeraChem stdout."""
    return re_search(r"(Git|Hg) Version: (\S*)", stdout).group(2)


def parse_terachem_version(stdout: str) -> str:
    """Parse the TeraChem version from stdout."""
    return re_search(r"TeraChem (v\S*)", stdout).group(1)


def calculation_succeeded(stdout: str) -> bool:
    """Determine from stdout whether a calculation completed."""
    return re.search(r"Job finished:", stdout) is not None


def parse_calctype(stdout: str) -> CalcType:
    """Parse the calculation type from TeraChem stdout."""
    calctypes = {
        r"SINGLE POINT ENERGY CALCULATIONS": CalcType.energy,
        r"SINGLE POINT GRADIENT CALCULATIONS": CalcType.gradient,
        r"FREQUENCY ANALYSIS": CalcType.hessian,
    }
    for regex, calctype in calctypes.items():
        if re.search(regex, stdout):
            return calctype
    raise MatchNotFoundError(regex, stdout)
