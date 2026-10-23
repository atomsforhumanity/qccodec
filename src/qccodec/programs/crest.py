"""CREST: write ``input.toml``, run ``crest input.toml``, and parse its output files.

Inputs map to CREST 3's TOML input as follows:

- ``keywords`` are the TOML document itself, so CREST's own input-file keys apply:
  top-level settings such as ``threads`` or ``runtype``, and sections such as
  ``{"calculation": {"level": [{"alpb": "water"}]}}`` or ``{"cregen": {...}}``.
  CREST stops on keys it does not know.
- ``model.method`` (``gfn2``, ``gfn1``, ``gfn0``, ``gfnff``, ...) and the structure's
  charge and multiplicity are written into every ``[[calculation.level]]``; setting
  ``charge`` or ``uhf`` in ``keywords`` is an error.
- ``runtype`` defaults from the calculation type (``singlepoint``, ``numhess``,
  ``optimize``, ``imtd-gc``); a given ``runtype`` must match it, e.g. ``nci`` or
  ``entropy`` for conformer searches.
- ``threads`` defaults to the number of CPUs, at most 16. CREST itself always runs
  with ``OMP_NUM_THREADS=1`` and parallelizes over its own threads.

Calculation types: ``energy``, ``gradient``, ``hessian`` (numerical: also frequencies
and normal modes), ``optimization``, and ``conformer_search``. CREST writes the normal
modes to ``g98.out`` as mass-weighted vectors (two decimals) in a reoriented frame; the
codec converts them to unit-length Cartesian displacements and rotates them back onto
the input structure, as qcdata defines them. The Hessian is in the input's frame. Rotamers are parsed only
with the decode option ``{"collect_rotamers": True}``; there can be thousands. CREST
optimizes with the center of mass at the origin; the codec shifts optimization
trajectories back into the input's frame. No input field carries atom indices.

CREST computes GFN-xTB and GFN-FF with its built-in tblite and gfnff libraries, so it
needs no separate xtb. Use the official static build: conda-forge's CREST 3.0.2 crashes
in its optimizer.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import numpy as np
import tomli_w
from qcdata import (
    CalcType,
    ExecutionInfo,
    Frame,
    ProgramInput,
    Provenance,
    SinglePointData,
    Structure,
)

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..utils import re_finditer, re_search
from ._common import invocation, parse_g98_freqs, parse_g98_normal_modes

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
    toml_dict = _to_toml_dict(inp, struct_filename)
    return invocation(
        inp,
        {
            "input.toml": tomli_w.dumps(toml_dict),
            struct_filename: inp.require_structure().to_xyz(),
        },
        "input.toml",
        environment={"OMP_NUM_THREADS": "1"},
        threads=int(toml_dict["threads"]),
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
        level["charge"] = inp.require_structure().charge
        level["uhf"] = inp.require_structure().multiplicity - 1
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


def _ensemble(ctx: Context, filename: str) -> list[Frame]:
    """Read a CREST multi-structure xyz file whose comment lines hold the energies."""
    assert ctx.input_data is not None
    structures = Structure.open_multi(
        ctx.path(filename),
        charge=ctx.input_data.require_structure().charge,
        multiplicity=ctx.input_data.require_structure().multiplicity,
    )
    energies = [float(s.extras[Structure._xyz_comment_key][0]) for s in structures]
    # Identifiers carry over when the topology is unchanged.
    if ctx.input_data.keywords.get("topo", True):
        ids = ctx.input_data.require_structure().identifiers.model_dump()
        for structure in structures:
            structure.add_identifiers(**ids)
    try:
        version = parse_version(ctx.stdout) if ctx.stdout else None
    except MatchNotFoundError:
        version = None
    provenance = Provenance(program="crest", program_version=version)
    return [
        Frame(
            structure=structure,
            success=True,
            results=SinglePointData(provenance=provenance, energy=energy),
        )
        for structure, energy in zip(structures, energies, strict=True)
    ]


@codec.parser(file="directory", calctypes=[CalcType.conformer_search])
def parse_conformers(ctx: Context) -> dict[str, Any]:
    """Parse the conformers, each a frame with its energy, from ``crest_conformers.xyz``."""
    if ctx.input_data is None:
        raise ParserError("Conformer parsing requires input_data.")
    return {"conformers": _ensemble(ctx, "crest_conformers.xyz")}


@codec.parser(file="directory", calctypes=[CalcType.conformer_search])
def parse_rotamers(ctx: Context) -> dict[str, Any]:
    """Parse the rotamers, each a frame with its energy, from ``crest_rotamers.xyz``.

    Runs only with ``options={"collect_rotamers": True}``.
    """
    if not ctx.options["collect_rotamers"]:
        return {}
    if ctx.input_data is None:
        raise ParserError("Rotamer parsing requires input_data.")
    return {"rotamers": _ensemble(ctx, "crest_rotamers.xyz")}


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


codec.parser(file="g98.out", target="freqs_wavenumber", calctypes=[CalcType.hessian])(
    parse_g98_freqs
)
codec.parser(
    file="directory", target="normal_modes_cartesian", calctypes=[CalcType.hessian]
)(parse_g98_normal_modes)


def _to_input_frame(structures: list[Structure], start: Structure) -> list[Structure]:
    """Undo CREST's shift of the molecule to its center of mass.

    CREST optimizes with the center of mass at the origin, so its trajectory is the
    input translated (not rotated). The first frame is the input structure, which gives
    the translation; frames are returned unchanged if that is not a pure translation.
    """
    if not structures or len(structures[0].symbols) != len(start.symbols):
        return structures
    offset = np.asarray(start.geometry) - np.asarray(structures[0].geometry)
    shift = offset.mean(axis=0)
    if not np.allclose(offset, shift, atol=1e-4):
        return structures
    return [
        s.model_copy(update={"geometry": np.asarray(s.geometry) + shift})
        for s in structures
    ]


@codec.parser(file="directory", target="frames", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[Frame]:
    """Build an optimization trajectory from ``crestopt.log`` and ``crest.engrad``.

    Intermediate steps expose energies only; the final step gains the gradient when
    ``crest.engrad`` is present. Geometries are moved back into the input's frame.
    """
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    structures = Structure.from_xyz_multi(
        ctx.read("crestopt.log"),
        charge=ctx.input_data.require_structure().charge,
        multiplicity=ctx.input_data.require_structure().multiplicity,
    )
    energies = [float(s.extras[Structure._xyz_comment_key][1]) for s in structures]
    structures = _to_input_frame(structures, ctx.input_data.require_structure())
    try:
        version = parse_version(ctx.stdout) if ctx.stdout else None
    except MatchNotFoundError:
        version = None
    provenance = Provenance(program="crest", program_version=version)
    trajectory: list[Frame] = [
        Frame(
            structure=structure,
            success=True,
            results=SinglePointData(provenance=provenance, energy=energy),
            execution=ExecutionInfo(scratch_dir=ctx.directory),
        )
        for structure, energy in zip(structures, energies, strict=True)
    ]
    if not trajectory:
        if ctx.failed:
            return []
        raise ParserError("CREST optimization trajectory is empty.")

    try:
        gradient = parse_gradient(ctx.read("crest.engrad"))
    except FileNotFoundError:
        pass
    except MatchNotFoundError:
        if not ctx.failed:
            raise
    else:
        final = trajectory[-1]
        results = final.results.model_copy(update={"gradient": gradient})
        trajectory[-1] = Frame.model_validate({**dict(final), "results": results})
    return trajectory
