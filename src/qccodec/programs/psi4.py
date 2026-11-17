"""Psi4: run through a small driver script and decode the JSON it writes.

The codec writes ``job.json`` and a driver script (``input.py``, from
``_psi4_driver.py``) and runs ``psi4 input.py -k -o stdout``: Psi4 executes the
script with its own interpreter, so the caller's Python environment never imports
Psi4. The script calls the Psi4 driver for the calculation type and writes
``results.json``; stdout carries Psi4's normal output.

Inputs map to Psi4 as follows:

- ``model.method`` and ``model.basis`` form the method string, e.g. ``"wb97x-d/def2-svp"``
  for ``psi4.energy("wb97x-d/def2-svp")``. Leave ``basis`` empty for composite methods or to
  set ``basis`` in ``keywords``.
- ``keywords`` go to ``psi4.set_options`` unchanged, so every Psi4 option and its
  module-scoped form (``"scf__maxiter"``) works as in Psi4's documentation. Two
  kinds of keyword are not Psi4 options and are routed instead:

  - Driver arguments go to ``psi4.energy()``/``gradient()``/``frequency()``/
    ``optimize()``: ``external_potentials`` (point charges for QM/MM electrostatic
    embedding, ``[[charge, [x, y, z]], ...]`` in bohr), ``dertype`` (``0`` or
    ``"energy"`` for derivatives by finite differences of energies, ``1`` or
    ``"gradient"`` for Hessians from finite differences of gradients), and
    ``project_trans``/``project_rot`` (whether frequency analysis removes
    translations and rotations; Psi4 decides by default).
  - ``symmetry`` is the molecule block's point group, e.g. ``"c1"`` to turn symmetry
    off. By default Psi4 detects whatever symmetry the input frame already has.

- ``cmdline_args`` are appended to the ``psi4`` command, e.g. ``["-n", "4"]`` for
  four threads or ``["--memory", "8GB"]``. Without ``-n``, Psi4 plans all the threads
  the runner has and runs with what it is granted (``-n <grant>``); called directly,
  with Psi4's default of one.

Psi4's options are too many to list in the codec, but Psi4 checks every one itself:
the driver sets them one at a time before calculating and reports each one Psi4 does
not know (kind ``input``, code ``unknown_key``, with the closest option Psi4 has) and
each value it does not take (kind ``input``).

The molecule keeps the input's frame (``no_com``, ``no_reorient``): Psi4 never moves
or rotates it, so gradients, Hessians, normal modes, and trajectories are in the
input's coordinates, and symmetry detection only finds symmetry the frame already
has. No input field carries atom indices.

Calculation types: ``energy``, ``gradient``, ``hessian`` (``psi4.frequency``: also
frequencies, normal modes, and the gradient), ``optimization`` and
``transition_state`` (``psi4.optimize`` with optking; every step carries its energy
and gradient). A transition-state search sets ``opt_type ts`` and, unless given,
``full_hess_every 0`` (an exact Hessian at the first step). Every single point also
records Mulliken charges in ``atomic_charges`` when the method supports them, the
gradient on external point charges in ``point_charge_gradient``, all scalar Psi4
variables in ``extras["psi4_variables"]``, and the orbital levels in
``wavefunction.orbitals``: ``restricted`` when Psi4's wavefunction has one set of
spatial orbitals for both spins (RHF, RKS, ROHF), ``unrestricted`` otherwise (UHF,
UKS, CUHF), even when a UHF lands on the closed-shell solution.

Failures and their kinds: an SCF that does not converge is ``convergence`` (code
``scf_not_converged``), optking out of iterations ``convergence`` (``max_iter``),
another iterative procedure ``convergence``; an unknown option, method or basis set,
an invalid value and a multiplicity the electrons cannot have are ``input``; a method
Psi4 has but not for this calculation is ``unsupported``; an addon this Psi4 lacks
(``ddx`` for ddCOSMO) is ``environment``.

Psi4 1.11 handles point charges correctly only for energies and gradients. Its
Hessians with ``external_potentials`` either fail or come back wrong (they disagree
with finite differences of its own gradients by about as much as the charges change
them), and its optimizations reset the molecule to the starting geometry at every
step, so they never move. The codec rejects those combinations; for QM/MM geometry
optimizations, run geomeTRIC with Psi4 gradients as its child.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qcdata import (
    CalcType,
    ExecutionInfo,
    Frame,
    Orbitals,
    OrbitalSpin,
    ProgramInput,
    Provenance,
    SinglePointData,
    Structure,
    Wavefunction,
)

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..program import CodecProgram
from ._common import invocation, unit_modes, vibrations

codec = Codec(
    "psi4",
    calctypes={
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
        CalcType.transition_state,
    },
    # The exception the driver recorded: "psi4: <name>: <message>".
    errors=[
        (r"OptionError: Unknown Psi4 option", "input", "unknown_key"),
        (r"OptionError", "input", None),
        (r"SCFConvergenceError", "convergence", "scf_not_converged"),
        (r"OptimizationConvergenceError", "convergence", "max_iter"),
        (r"ConvergenceError", "convergence", None),
        (r"BasisSetNotFound|BasisSetNotDefined", "input", None),
        (r"not available for any derivative level", "input", None),  # unknown method
        (r"ManagedMethodError|MissingMethodError", "unsupported", None),
        (r"ValidationError|UpgradeHelper", "input", None),
        (r"ModuleNotFoundError|ImportError", "environment", None),  # an addon missing
    ],
)

DRIVER_FILE = "input.py"
JOB_FILE = "job.json"
RESULTS_FILE = "results.json"
# Keywords that are arguments of the Psi4 driver call rather than Psi4 options.
DRIVER_KWARGS = frozenset(
    {"external_potentials", "dertype", "project_trans", "project_rot"}
)
# Keywords that are directives of the molecule block.
MOLECULE_KEYWORDS = frozenset({"symmetry"})
OPTIMIZATIONS = [CalcType.optimization, CalcType.transition_state]
_DRIVER_SOURCE = (Path(__file__).with_name("_psi4_driver.py")).read_text()
SINGLE_POINTS = [CalcType.energy, CalcType.gradient, CalcType.hessian]


# ------------------------------------------------------------------ encoding


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write ``job.json`` and the driver script, and build the ``psi4`` command.

    Raises:
        EncoderError: The model is missing, ``opt_type`` contradicts the
            calculation type, or point charges are combined with a calculation
            Psi4 cannot run with them.
    """
    if inp.model is None:
        raise EncoderError(
            "psi4 requires a model, e.g. {'method': 'wb97x-d', 'basis': 'def2-svp'}."
        )
    method = inp.model.method
    if inp.model.basis:
        method = f"{method}/{inp.model.basis}"
    options: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}
    molecule_keywords: dict[str, Any] = {}
    for key, value in inp.keywords.items():
        name = key.lower()
        if name in DRIVER_KWARGS:
            kwargs[name] = value
        elif name in MOLECULE_KEYWORDS:
            molecule_keywords[name] = value
        else:
            options[key] = value
    _optimizer_options(inp.calctype, options)
    if "external_potentials" in kwargs and inp.calctype not in (
        CalcType.energy,
        CalcType.gradient,
    ):
        raise EncoderError(
            "Psi4 1.11 supports external_potentials only for energies and gradients: "
            "its Hessians fail or are wrong, and its optimizations never leave the "
            "starting geometry. For QM/MM optimizations, run geomeTRIC with Psi4 "
            "gradients as its child.",
            kind="unsupported",
        )
    job = {
        "calctype": inp.calctype.value,
        "method": method,
        "options": options,
        "kwargs": kwargs,
        "molecule": {
            "symbols": list(inp.require_structure().symbols),
            "geometry": inp.require_structure().geometry.tolist(),
            "charge": inp.require_structure().charge,
            "multiplicity": inp.require_structure().multiplicity,
            "symmetry": molecule_keywords.get("symmetry"),
        },
    }
    return invocation(
        inp,
        {JOB_FILE: json.dumps(job, indent=2) + "\n", DRIVER_FILE: _DRIVER_SOURCE},
        DRIVER_FILE,
        arguments=("-k", "-o", "stdout"),
        # Keep Psi4's scratch files in the calculation directory.
        environment={"PSI_SCRATCH": "."},
        threads=_threads(inp.cmdline_args),
    )


def _threads(cmdline_args: list[str]) -> int | None:
    """Return the thread count given with ``-n``/``--nthread``, None without one."""
    args = list(cmdline_args)
    for i, arg in enumerate(args):
        if arg in ("-n", "--nthread") and i + 1 < len(args):
            return int(args[i + 1])
        if arg.startswith("--nthread="):
            return int(arg.split("=", 1)[1])
    return None


def _optimizer_options(calctype: CalcType, options: dict[str, Any]) -> None:
    """Make optking's search type match the calculation type.

    Raises:
        EncoderError: ``opt_type`` asks for the other kind of search.
    """
    lower = {k.lower(): k for k in options}
    opt_type = str(options[lower["opt_type"]]).lower() if "opt_type" in lower else None
    if calctype == CalcType.transition_state:
        if opt_type not in (None, "ts"):
            raise EncoderError("A transition_state calculation needs opt_type 'ts'.")
        options.setdefault(lower.get("opt_type", "opt_type"), "ts")
        # optking's guess Hessian rarely finds a saddle point; start from an exact one.
        options.setdefault(lower.get("full_hess_every", "full_hess_every"), 0)
    elif calctype == CalcType.optimization and opt_type == "ts":
        raise EncoderError("Use calctype transition_state for opt_type 'ts'.")


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
    fields = _single_point_fields(results)
    if results.get("frequencies") and results.get("normal_modes"):
        if ctx.input_data is None:
            raise ParserError("Vibrational analysis parsing requires input_data.")
        fields["vibrational_analysis"] = vibrations(
            results["frequencies"],
            unit_modes(results["normal_modes"]),
            ctx.input_data.require_structure(),
        )
    return fields


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
        "dipole_moment": results.get("dipole"),
        "atomic_charges": results.get("mulliken_charges"),
        "point_charge_gradient": results.get("point_charge_gradient"),
    }
    orbitals = _orbitals(results)
    if orbitals is not None:
        fields["wavefunction"] = Wavefunction(orbitals=orbitals)
    if results.get("variables"):
        fields["extras"] = {"psi4_variables": results["variables"]}
    return {k: v for k, v in fields.items() if v is not None}


def _orbitals(results: dict[str, Any]) -> Orbitals | None:
    """The orbital levels of the final wavefunction, from the driver's results.

    The driver writes Psi4's ``epsilon`` and ``occupation`` vectors per spin and
    whether the wavefunction's alpha and beta electrons share spatial orbitals
    (``same_a_b_orbs``: RHF, RKS and ROHF do; UHF, UKS and CUHF do not), so a UHF
    that lands on a closed-shell solution is still unrestricted. None when the driver
    wrote no levels.
    """
    energies_a = results.get("eigenvalues_a")
    occupations_a = results.get("occupations_a")
    same_orbitals = results.get("same_orbitals")
    if energies_a is None or occupations_a is None or same_orbitals is None:
        return None
    energies_b = results.get("eigenvalues_b")
    occupations_b = results.get("occupations_b", occupations_a)
    restricted = bool(same_orbitals)
    return Orbitals(
        spin=OrbitalSpin.restricted if restricted else OrbitalSpin.unrestricted,
        energies_a=energies_a,
        energies_b=None if restricted else energies_b,
        occupations_a=occupations_a,
        occupations_b=occupations_b,
    )


@codec.parser(file="directory", target="frames", calctypes=OPTIMIZATIONS)
def parse_trajectory(ctx: Context) -> list[Frame]:
    """Build the optimization trajectory from optking's history.

    Every frame holds the energy and gradient at that step's geometry. The final step also
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
    structure = ctx.input_data.require_structure()
    trajectory: list[Frame] = []
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
            Frame(
                structure=step_structure,
                success=True,
                results=SinglePointData(provenance=provenance, **fields),
                execution=ExecutionInfo(scratch_dir=ctx.directory),
            )
        )
    return trajectory


def cap_threads(inp: ProgramInput, threads: int) -> ProgramInput:
    """The input with ``-n``/``--nthread`` lowered to ``threads``."""
    args = list(inp.cmdline_args)
    kept: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in ("-n", "--nthread"):
            i += 2
            continue
        if not args[i].startswith("--nthread="):
            kept.append(args[i])
        i += 1
    return inp.model_copy(update={"cmdline_args": [*kept, "-n", str(threads)]})


codec.cap_threads = cap_threads

#: The program qccompute runs (the ``qccompute.programs`` entry point 'psi4').
#: Psi4 scales with threads: without ``-n`` it plans all the runner has and runs with
#: the grant.
program = CodecProgram(codec, threads=None)
