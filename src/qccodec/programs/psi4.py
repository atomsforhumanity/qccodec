"""Psi4: run through a small driver script and decode the JSON it writes.

The codec writes ``job.json`` and a driver script (``input.py``, from
``_psi4_driver.py``) and runs ``psi4 input.py -k -o stdout``: Psi4 executes the
script with its own interpreter, so the caller's Python environment never imports
Psi4. The script calls the Psi4 driver for the calculation type and writes
``results.json``; stdout carries Psi4's normal output.

Inputs map to Psi4 as follows:

- ``model.method`` and ``model.basis`` form the method string, e.g. ``"hf/sto-3g"``
  for ``psi4.energy("hf/sto-3g")``. Leave ``basis`` empty for composite methods or to
  set ``basis`` in ``keywords``.
- ``keywords`` go to ``psi4.set_options`` unchanged, so every Psi4 option and its
  module-scoped form (``"scf__maxiter"``) works as in Psi4's documentation.
- Driver keyword arguments are the exception: ``external_potentials`` (point charges
  for QM/MM electrostatic embedding, ``[[charge, [x, y, z]], ...]`` in bohr) and
  ``dertype`` are passed to ``psi4.energy()``/``gradient()``/``frequency()``/
  ``optimize()`` instead.
- ``cmdline_args`` are appended to the ``psi4`` command, e.g. ``["-n", "4"]`` for
  four threads or ``["--memory", "8GB"]``.

The structure runs in a fixed frame without symmetry (``no_com``, ``no_reorient``,
``symmetry c1``), so gradients, Hessians, and point-charge gradients are in the
input's coordinates. No input field carries atom indices.

Calculation types: ``energy``, ``gradient``, ``hessian`` (``psi4.frequency``: also
frequencies, normal modes, and the gradient), and ``optimization`` (``psi4.optimize``
with optking; every step carries its energy and gradient). Every single point also
records Mulliken charges in ``atomic_charges`` when the method supports them, the
gradient on external point charges in ``point_charge_gradient``, and all scalar Psi4
variables in ``extras["psi4_variables"]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qcdata import (
    CalcType,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    Provenance,
    SinglePointData,
    Structure,
    Wavefunction,
)

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ._common import invocation

codec = Codec(
    "psi4",
    calctypes={
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
    },
)

DRIVER_FILE = "input.py"
JOB_FILE = "job.json"
RESULTS_FILE = "results.json"
# Keywords that are arguments of the Psi4 driver call rather than Psi4 options.
DRIVER_KWARGS = frozenset({"external_potentials", "dertype"})
_DRIVER_SOURCE = (Path(__file__).with_name("_psi4_driver.py")).read_text()
SINGLE_POINTS = [CalcType.energy, CalcType.gradient, CalcType.hessian]


# ------------------------------------------------------------------ encoding


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write ``job.json`` and the driver script, and build the ``psi4`` command.

    Raises:
        EncoderError: The model is missing.
    """
    if inp.model is None:
        raise EncoderError(
            "psi4 requires a model, e.g. {'method': 'hf', 'basis': 'sto-3g'}."
        )
    method = inp.model.method
    if inp.model.basis:
        method = f"{method}/{inp.model.basis}"
    options = {k: v for k, v in inp.keywords.items() if k not in DRIVER_KWARGS}
    kwargs = {k: v for k, v in inp.keywords.items() if k in DRIVER_KWARGS}
    job = {
        "calctype": inp.calctype.value,
        "method": method,
        "options": options,
        "kwargs": kwargs,
        "molecule": {
            "symbols": list(inp.structure.symbols),
            "geometry": inp.structure.geometry.tolist(),
            "charge": inp.structure.charge,
            "multiplicity": inp.structure.multiplicity,
        },
    }
    return invocation(
        inp,
        {JOB_FILE: json.dumps(job, indent=2) + "\n", DRIVER_FILE: _DRIVER_SOURCE},
        DRIVER_FILE,
        arguments=("-k", "-o", "stdout"),
        # Keep Psi4's scratch files in the calculation directory.
        environment={"PSI_SCRATCH": "."},
    )


# ------------------------------------------------------------------ decoding


def _results(ctx: Context) -> dict[str, Any]:
    """Read ``results.json``."""
    data: dict[str, Any] = json.loads(ctx.read(RESULTS_FILE))
    return data


@codec.failure
def failure(ctx: Context) -> str | None:
    """Report the exception the driver recorded, or a Psi4 crash before it wrote one."""
    try:
        results = _results(ctx)
    except (FileNotFoundError, ValueError):
        # No results: Psi4 died before the driver could record anything.
        if ctx.stdout and "Psi4 encountered an error" in ctx.stdout:
            return "psi4 encountered an error before writing results. See logs."
        return None
    error = results.get("error")
    return f"psi4: {error}" if error else None


@codec.parser(
    target=("provenance", "program_version"), file="directory", required=False
)
def parse_version(ctx: Context) -> str:
    """Read the Psi4 version the driver recorded."""
    version = _results(ctx).get("psi4_version")
    if version is None:
        raise MatchNotFoundError("psi4_version", RESULTS_FILE)
    return str(version)


@codec.parser(file="directory", calctypes=SINGLE_POINTS)
def parse_single_point(ctx: Context) -> dict[str, Any]:
    """Map the driver's results onto SinglePointData fields."""
    results = _results(ctx)
    if "energy" not in results:
        raise MatchNotFoundError("energy", RESULTS_FILE)
    return _single_point_fields(results)


def _single_point_fields(results: dict[str, Any]) -> dict[str, Any]:
    """Translate driver result keys into qcdata field names, skipping absent values."""
    fields: dict[str, Any] = {
        "energy": results.get("energy"),
        "gradient": results.get("gradient"),
        "hessian": results.get("hessian"),
        "nuclear_repulsion_energy": results.get("nuclear_repulsion_energy"),
        "calcinfo_nbasis": results.get("nbasis"),
        "calcinfo_nmo": results.get("nmo"),
        "calcinfo_nalpha": results.get("nalpha"),
        "calcinfo_nbeta": results.get("nbeta"),
        "scf_dipole_moment": results.get("dipole"),
        "atomic_charges": results.get("mulliken_charges"),
        "point_charge_gradient": results.get("point_charge_gradient"),
        "normal_modes_cartesian": results.get("normal_modes"),
    }
    if results.get("frequencies") is not None:
        fields["freqs_wavenumber"] = results["frequencies"]
    wavefunction = {
        "scf_eigenvalues_a": results.get("eigenvalues_a"),
        "scf_eigenvalues_b": results.get("eigenvalues_b"),
        "scf_occupations_a": results.get("occupations_a"),
        "scf_occupations_b": results.get("occupations_b"),
    }
    if any(v is not None for v in wavefunction.values()):
        fields["wavefunction"] = Wavefunction.model_validate(wavefunction)
    if results.get("variables"):
        fields["extras"] = {"psi4_variables": results["variables"]}
    return {k: v for k, v in fields.items() if v is not None}


@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]:
    """Build the optimization trajectory from optking's history.

    Every step is a gradient calculation at that step's geometry. The final step also
    carries what the final wavefunction provides (charges, orbitals, variables).
    """
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    results = _results(ctx)
    steps = results.get("trajectory")
    if not steps:
        if ctx.failed:
            return []
        raise MatchNotFoundError("trajectory", RESULTS_FILE)
    provenance = Provenance(program="psi4", program_version=results.get("psi4_version"))
    structure = ctx.input_data.structure
    trajectory: list[ProgramOutput] = []
    for i, step in enumerate(steps):
        fields: dict[str, Any] = {
            "energy": step["energy"],
            "gradient": [
                step["gradient"][j : j + 3] for j in range(0, len(step["gradient"]), 3)
            ],
        }
        if i == len(steps) - 1:
            final = _single_point_fields(results)
            fields = {**final, **fields}
        step_structure = Structure(
            symbols=structure.symbols,
            geometry=step["geometry"],
            charge=structure.charge,
            multiplicity=structure.multiplicity,
        )
        trajectory.append(
            ProgramOutput(
                input_data=ProgramInput.model_validate(
                    {
                        **ctx.input_data.model_dump(),
                        "calctype": CalcType.gradient,
                        "structure": step_structure,
                    }
                ),
                success=True,
                results=SinglePointData(provenance=provenance, **fields),
                execution=ExecutionInfo(scratch_dir=ctx.directory),
            )
        )
    return trajectory
