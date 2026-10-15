"""TeraChem: input file encoding, stdout/scratch parsing, and wavefunction propagation."""

from __future__ import annotations

import re

from qcdata import (
    CalcType,
    ExecutionInfo,
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
from ._common import invocation

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
        f"{'charge':<{PADDING}} {inp.structure.charge}",
        f"{'spinmult':<{PADDING}} {inp.structure.multiplicity}",
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
    for key, value in inp.keywords.items():
        if key in non_keywords:
            raise EncoderError(
                f"Keyword '{key}' should not be set as a keyword. It "
                f"should be set at '{non_keywords[key]}'",
            )
        lines.append(f"{key:<{PADDING}} {str(value).lower()}")  # lowercase booleans

    return invocation(
        inp,
        {"tc.in": "\n".join(lines) + "\n", XYZ_FILENAME: inp.structure.to_xyz()},
        "tc.in",
        wavefunction_files=("scr*/c0", "scr*/ca0", "scr*/cb0"),
    )


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


@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]:
    """Build an optimization trajectory from ``scr*/optim.xyz`` and stdout.

    Each step's stdout is decoded as a gradient calculation with this codec.
    """
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    if not ctx.stdout and ctx.failed:
        return []
    stdout = ctx.stdout or ""

    structures = Structure.open_multi(
        ctx.path("scr*/optim.xyz"),
        charge=ctx.input_data.structure.charge,
        multiplicity=ctx.input_data.structure.multiplicity,
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

    trajectory: list[ProgramOutput] = []
    for structure, step_stdout in zip(structures, per_step, strict=False):
        step_logs = initialization + step_stdout
        parsed = ctx.decode(CalcType.gradient, stdout=step_logs)
        assert isinstance(parsed, SinglePointData)  # for mypy
        values = parsed.model_dump()
        values["energy"] = structure.extras[Structure._xyz_comment_key][0]
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
                logs=step_logs,
            )
        )
    return trajectory


# ------------------------------------------------------- unregistered helpers


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
