"""CREST: TOML input encoding and parsing of its stdout and output files."""

from __future__ import annotations

import copy
import math
import os
import re
from typing import Any

import tomli_w
from qcconst import constants
from qcdata import (
    CalcType,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    Provenance,
    SinglePointData,
    Structure,
)

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..utils import re_finditer, re_search
from ._common import invocation

codec = Codec(
    "crest",
    calctypes={
        CalcType.conformer_search,
        CalcType.optimization,
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
    },
    # Rotamer sets can be very large, so they are not collected unless requested.
    options={"collect_rotamers": False},
)


# ------------------------------------------------------------------ encoding


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write ``input.toml`` and the structure file for a CREST calculation."""
    validate_input(inp)
    struct_filename = "structure.xyz"
    return invocation(
        inp,
        {
            "input.toml": tomli_w.dumps(_to_toml_dict(inp, struct_filename)),
            struct_filename: inp.structure.to_xyz(),
        },
        "input.toml",
        environment={"OMP_NUM_THREADS": "1"},
    )


def validate_input(inp: ProgramInput) -> None:
    """Reject inputs CREST cannot run or that duplicate structured data.

    Raises:
        EncoderError: The input is invalid.
    """
    if inp.model is None:
        raise EncoderError("crest input requires a scientific model.")
    for keyword in ("charge", "uhf"):  # set on the Structure instead
        if keyword in inp.keywords:
            raise EncoderError(
                f"{keyword} should not be set in keywords for CREST. It is already set "
                "on the Structure or ProgramInput elsewhere.",
            )
    if "runtype" in inp.keywords:
        _validate_runtype_calctype(inp.keywords["runtype"], inp.calctype)


_RUNTYPES: dict[CalcType, set[str]] = {
    CalcType.conformer_search: {"imtd-gc", "imtd-smtd", "entropy", "nci", "nci-mtd"},
    CalcType.optimization: {"optimize", "ancopt"},
    CalcType.energy: {"singlepoint"},
    CalcType.gradient: {"singlepoint"},
    CalcType.hessian: {"numhess"},
}
_DEFAULT_RUNTYPE: dict[CalcType, str] = {
    CalcType.conformer_search: "imtd-gc",
    CalcType.optimization: "optimize",
    CalcType.energy: "singlepoint",
    CalcType.gradient: "singlepoint",
    CalcType.hessian: "numhess",
}


def _validate_runtype_calctype(runtype: str, calctype: CalcType) -> None:
    """Validate that the runtype is supported for the calctype."""
    valid = _RUNTYPES.get(calctype, set())
    if runtype not in valid:
        raise EncoderError(
            f"Unsupported runtype {runtype} for calctype {calctype}. Valid runtypes "
            f"are: {valid}.",
        )


def _to_toml_dict(inp: ProgramInput, struct_filename: str) -> dict[str, Any]:
    """Convert a ProgramInput into CREST's TOML structure (kept separate for testing)."""
    if inp.model is None:
        raise EncoderError("crest input requires a scientific model.")
    toml_dict = copy.deepcopy(inp.keywords)
    # Logical cores were 10% faster than physical cores, so psutil is not used.
    toml_dict.setdefault("threads", min(os.cpu_count() or 16, 16))
    toml_dict["input"] = struct_filename
    if "runtype" not in toml_dict:
        if inp.calctype not in _DEFAULT_RUNTYPE:
            raise EncoderError(
                f"Unsupported calctype {inp.calctype} for CREST encoder."
            )
        toml_dict["runtype"] = _DEFAULT_RUNTYPE[inp.calctype]

    calculation = toml_dict.pop("calculation", {})
    levels = calculation.pop("level", []) or [{}]
    for level in levels:
        level["method"] = inp.model.method
        level["charge"] = inp.structure.charge
        level["uhf"] = inp.structure.multiplicity - 1
    calculation["level"] = levels
    toml_dict["calculation"] = calculation
    return toml_dict


# ------------------------------------------------------------------ decoding


@codec.failure
def failure(ctx: Context) -> str | None:
    """CREST prints FAILED on scientific failure even with a zero exit code."""
    if ctx.stdout and "FAILED" in ctx.stdout:
        return "CREST reported FAILED. See calculation logs."
    return None


@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    """Parse the version string from CREST stdout."""
    return re_search(r"Version (\d+\.\d+\.\d+),", stdout).group(1)


def _structures_with_energies(
    ctx: Context, filename: str
) -> tuple[list[Structure], list[float]]:
    """Read a CREST multi-structure xyz file whose comment line holds the energy."""
    assert ctx.input_data is not None
    structures = Structure.open_multi(
        ctx.path(filename),
        charge=ctx.input_data.structure.charge,
        multiplicity=ctx.input_data.structure.multiplicity,
    )
    energies = [float(s.extras[Structure._xyz_comment_key][0]) for s in structures]
    # Identifiers carry over when the topology is unchanged.
    if ctx.input_data.keywords.get("topo", True):
        ids = ctx.input_data.structure.identifiers.model_dump()
        for structure in structures:
            structure.add_identifiers(**ids)
    return structures, energies


@codec.parser(file="directory", calctypes=[CalcType.conformer_search])
def parse_conformers(ctx: Context) -> dict[str, Any]:
    """Parse conformers and their energies from ``crest_conformers.xyz``."""
    if ctx.input_data is None:
        raise ParserError("Conformer parsing requires input_data.")
    conformers, energies = _structures_with_energies(ctx, "crest_conformers.xyz")
    return {"conformers": conformers, "conformer_energies": energies}


@codec.parser(file="directory", calctypes=[CalcType.conformer_search])
def parse_rotamers(ctx: Context) -> dict[str, Any]:
    """Parse rotamers and their energies from ``crest_rotamers.xyz``.

    Runs only with ``options={"collect_rotamers": True}``.
    """
    if not ctx.options["collect_rotamers"]:
        return {}
    if ctx.input_data is None:
        raise ParserError("Rotamer parsing requires input_data.")
    rotamers, energies = _structures_with_energies(ctx, "crest_rotamers.xyz")
    return {"rotamers": rotamers, "rotamer_energies": energies}


@codec.parser(
    file="crest.engrad", target="energy", calctypes=[CalcType.energy, CalcType.gradient]
)
def parse_energy(contents: str) -> float:
    """Parse the energy from ``crest.engrad``."""
    return float(re_search(r"# Energy \( Eh \)\n#*\n\s*([-\d.]+)", contents).group(1))


@codec.parser(
    file="crest.engrad",
    target="gradient",
    calctypes=[CalcType.energy, CalcType.gradient],
)
def parse_gradient(contents: str) -> list[list[float]]:
    """Parse the gradient from ``crest.engrad`` as an N x 3 list."""
    regex = r"# Gradient \( Eh/a0 \)\n#\s*\n((?:\s*[-\d.]+\n)+)"
    vals = [float(x) for x in re_search(regex, contents).group(1).split()]
    return [vals[i : i + 3] for i in range(0, len(vals), 3)]


@codec.parser(target="energy", calctypes=[CalcType.hessian])
def parse_energy_numhess(stdout: str) -> float:
    """Parse the initial single point energy of a numerical Hessian run from stdout."""
    return float(re_search(r"Energy\s=\s*([-+]?\d+\.\d+)\s*Eh", stdout).group(1))


@codec.parser(file="numhess1", target="hessian", calctypes=[CalcType.hessian])
def parse_numhess1(contents: str) -> list[list[float]]:
    """Parse the numerical Hessian from CREST's ``numhess1`` file."""
    numbers = [float(m.group()) for m in re_finditer(r"[-]?\d*\.\d+|\d+", contents)]
    n = int(math.sqrt(len(numbers)))
    if n * n != len(numbers):
        raise ParserError(
            f"Expected a square matrix, but found {len(numbers)} elements."
        )
    return [numbers[i * n : (i + 1) * n] for i in range(n)]


@codec.parser(file="g98.out", target="freqs_wavenumber", calctypes=[CalcType.hessian])
def parse_g98_freqs(contents: str) -> list[float]:
    """Parse vibrational frequencies (wavenumbers) from CREST's ``g98.out``."""
    regex = r"Frequencies\s+--\s+(?P<floats>(?:-?\d+\.\d+\s*)+)"
    matches = re_finditer(regex, contents)
    return [float(f) for m in matches for f in m.group("floats").split()]


@codec.parser(
    file="g98.out", target="normal_modes_cartesian", calctypes=[CalcType.hessian]
)
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


@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]:
    """Build an optimization trajectory from ``crestopt.log`` and ``crest.engrad``.

    Intermediate steps expose energies only; the final step gains the gradient when
    ``crest.engrad`` is present.
    """
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    structures = Structure.from_xyz_multi(
        ctx.read("crestopt.log"),
        charge=ctx.input_data.structure.charge,
        multiplicity=ctx.input_data.structure.multiplicity,
    )
    energies = [float(s.extras[Structure._xyz_comment_key][1]) for s in structures]
    try:
        version = parse_version(ctx.stdout) if ctx.stdout else None
    except MatchNotFoundError:
        version = None
    provenance = Provenance(program="crest", program_version=version)
    trajectory: list[ProgramOutput] = [
        ProgramOutput(
            input_data=ProgramInput.model_validate(
                {
                    **ctx.input_data.model_dump(),
                    "calctype": CalcType.energy,
                    "structure": structure,
                }
            ),
            success=True,
            results=SinglePointData(provenance=provenance, energy=energy),
            execution=ExecutionInfo(scratch_dir=ctx.directory),
        )
        for structure, energy in zip(structures, energies)
    ]
    if not trajectory:
        if ctx.failed:
            return []
        raise ParserError("CREST optimization trajectory is empty.")

    final = trajectory[-1].model_dump()
    try:
        gradient = parse_gradient(ctx.read("crest.engrad"))
    except FileNotFoundError:
        pass
    except MatchNotFoundError:
        if not ctx.failed:
            raise
    else:
        final["input_data"]["calctype"] = CalcType.gradient
        final["results"]["gradient"] = gradient
    trajectory[-1] = ProgramOutput.model_validate(final)
    return trajectory
