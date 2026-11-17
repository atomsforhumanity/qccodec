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

Hessians are the full Cartesian second derivative: analytic, or from finite
differences of gradients (or energies) in which Psi4 projects out only translations,
as the codec leaves its ``fd_project`` unset. Setting ``fd_project`` true, which
projects rotations out too, is refused for Hessians (kind ``unsupported``). Psi4's
harmonic analysis projects out rotations only where it judges the geometry
stationary (gradient rms below 1e-2 Eh/bohr; ``project_rot`` overrides that);
elsewhere it keeps the rotations among the modes, and the codec records
``projection`` as ``translations`` and flags them in ``external``: the modes with
the largest share in the mass-weighted rotational space.

The molecule keeps the input's frame (``no_com``, ``no_reorient``): Psi4 never moves
or rotates it, so gradients, Hessians, normal modes, and trajectories are in the
input's coordinates, and symmetry detection only finds symmetry the frame already
has. No input field carries atom indices.

Calculation types: ``energy``, ``gradient``, ``hessian`` (``psi4.frequency``: also
frequencies, normal modes, and the gradient), ``optimization`` and
``transition_state`` (``psi4.optimize`` with optking; every step carries its energy
and gradient). A transition-state search sets ``opt_type ts`` and, unless given,
``full_hess_every 0`` (an exact Hessian at the first step). Every single point also
records the dipole moment in ``dipole_moment`` for HF and DFT (after a correlated
method such as MP2, Psi4 holds only the reference's dipole, so the codec leaves it
out), Mulliken charges in ``atomic_charges`` when the method supports them, the
gradient on external point charges in ``point_charge_gradient``, all scalar Psi4
variables in ``extras["psi4_variables"]``, and the orbital levels in
``wavefunction.orbitals``: ``restricted`` when Psi4's wavefunction has one set of
spatial orbitals for both spins (RHF, RKS, ROHF), ``unrestricted`` otherwise (UHF,
UKS, CUHF), even when a UHF lands on the closed-shell solution.

Failures and their kinds: an SCF that does not converge is ``convergence`` (code
``scf_not_converged``), with its last iterate's energy and orbital levels; coupled-
perturbed (CPHF/CPKS) equations that end above their tolerance are ``convergence``
(``response_not_converged``), with the results Psi4 computed from them, since Psi4 1.11
goes on with them without a word when its solver runs out of iterations
(``solver_maxiter``); optking out of iterations ``convergence`` (``max_iter``),
another iterative procedure ``convergence``; an unknown option, method or basis set,
an invalid value and a multiplicity the electrons cannot have are ``input``; a method
Psi4 has but not for this calculation is ``unsupported``; an addon this Psi4 lacks
(``ddx`` for ddCOSMO) is ``environment``; a calculation out of memory is ``resource``
(``host_memory``), including a ``--memory`` below Psi4's minimum of 250 MiB. Its
``details`` give ``required_gib`` and ``available_gib`` where Psi4 prints them, in the
terms of Psi4's memory setting (``--memory``): the memory with which the calculation
fits, and the memory it ran with. For the minimum that is the minimum; for density
fitting's in-core tensors (``DFHelper Memory: AOs need 1.142 GiB; user supplied 0.189
GiB``), which get a fixed share of Psi4's memory (``scf_mem_safety_factor``, 0.75 by
default), it is that memory scaled by what the tensors need over what they got.

With ``fail_on_maxiter`` false Psi4 carries on after an SCF that did not converge;
the result then comes with the warning ``loosely_converged``, giving the last
iteration's energy change and orbital gradient against Psi4's thresholds.

Psi4 1.11 handles point charges correctly only for energies and gradients. Its
Hessians with ``external_potentials`` either fail or come back wrong (they disagree
with finite differences of its own gradients by about as much as the charges change
them), and its optimizations reset the molecule to the starting geometry at every
step, so they never move. The codec rejects those combinations; for QM/MM geometry
optimizations, run geomeTRIC with Psi4 gradients as its child.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from qcdata import (
    CalcType,
    ExecutionInfo,
    Frame,
    Orbitals,
    OrbitalSpin,
    ProgramInput,
    ProgramWarning,
    Provenance,
    SinglePointData,
    Structure,
    Wavefunction,
)
from qcref import periodic_table

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..program import CodecProgram
from ._common import invocation, unit_modes, vibrations

#: Psi4's failures for want of memory, in its own words.
_OUT_OF_MEMORY = (
    r"(?i)not enough memory|insufficient memory|out of memory|more memory"
    r"|too little memory|memory error|MemoryError|requires at least .*memory"
    r"|set_memory\(\): Requested"
)
#: set_memory's refusal of less than Psi4's minimum.
_MINIMUM = (
    r"set_memory\(\): Requested (?P<asked>[\d.]+(?:e[+-]?\d+)?) MiB [^;]*; "
    r"minimum (?P<least>[\d.]+) MiB \([^)]*\)"
)

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
        # First: set_memory's minimum is a ValidationError.
        (_OUT_OF_MEMORY, "resource", "host_memory"),
        (r"OptionError: Unknown Psi4 option", "input", "unknown_key"),
        (r"OptionError", "input", None),
        (r"SCFConvergenceError", "convergence", "scf_not_converged"),
        (
            r"CPHF did not converge|CPKS did not converge|coupled-perturbed .* did "
            r"not converge",
            "convergence",
            "response_not_converged",
        ),
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
    if inp.calctype == CalcType.hessian and _projects_hessian(options):
        raise EncoderError(
            "fd_project makes Psi4 project translations and rotations out of a "
            "finite-difference Hessian; results.hessian is the full second "
            "derivative, so leave fd_project unset (or false).",
            kind="unsupported",
        )
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


def _projects_hessian(options: dict[str, Any]) -> bool:
    """Whether ``fd_project`` (or ``findif__fd_project``) is set true."""
    return any(
        key.lower().split("__")[-1] == "fd_project"
        and str(value).lower() in {"true", "1", "on", "yes"}
        for key, value in options.items()
    )


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
def failure(ctx: Context) -> str | tuple[str, dict[str, Any]] | None:
    """Report the exception the driver recorded, or a Psi4 crash before it wrote one.

    A failure out of memory comes with what was needed, where Psi4 prints it
    (`memory_details`).
    """
    stdout = ctx.stdout or ""
    try:
        results = _results(ctx)
    except (FileNotFoundError, ValueError):
        # No results: Psi4 died before the driver could record anything.
        minimum = re.search(_MINIMUM, stdout)
        if minimum:  # --memory below Psi4's minimum: it stops at startup
            asked, least = float(minimum["asked"]), float(minimum["least"])
            return (
                f"psi4: {minimum.group(0)}",
                {
                    "required_gib": round(least / 1024, 9),
                    "available_gib": round(asked / 1024, 9),
                },
            )
        if "Psi4 encountered an error" in stdout:
            return "psi4 encountered an error before writing results. See logs."
        return None
    error = results.get("error")
    if not error:
        # Only a Hessian is made of the response; an optimization's starting
        # Hessian (full_hess_every) is a guess the search does not depend on.
        return (
            unconverged_response(stdout) if ctx.calctype == CalcType.hessian else None
        )
    message = f"psi4: {error}"
    if re.search(_OUT_OF_MEMORY, message):
        return message, memory_details(stdout)
    return message


_UNITS = {"": 1 / 1024**3, "Ki": 1 / 1024**2, "Mi": 1 / 1024, "Gi": 1.0, "Ti": 1024.0}


def memory_details(stdout: str) -> dict[str, Any]:
    """What a calculation out of memory needed, from Psi4's output, in GiB.

    Density fitting prints its in-core tensors' need against the share of Psi4's
    memory they were given (``DFHelper Memory: AOs need 1.142 GiB; user supplied
    0.189 GiB``, ``DFTensor Memory: Qso construction needs ...``). The calculation's
    budget is Psi4's memory (its header's ``Memory:``), and it needs at least that
    budget plus what the tensors lacked.

    Returns:
        ``required_gib`` and ``available_gib``, or nothing when Psi4 printed no need.
    """
    needs = re.findall(r"needs? ([\d.]+) GiB; user supplied ([\d.]+) GiB", stdout)
    budget = re.search(r"^\s*Memory:\s+([\d.]+) ([KMGT]i|)B\s*$", stdout, re.M)
    if not needs or budget is None:
        return {}
    need, supplied = (float(v) for v in needs[-1])
    total = float(budget.group(1)) * _UNITS[budget.group(2)]
    # The tensors get a fixed share of Psi4's memory, so the memory with which they
    # fit scales with need / supplied: bounded for the three decimals Psi4 prints.
    required = total * (need + 5e-4) / max(supplied - 5e-4, 1e-9)
    return {
        "required_gib": round(required, 9),
        "available_gib": round(total, 9),
    }


#: Psi4's coupled-perturbed solvers print a row per iteration, the last column but
#: one the equations not yet converged: ``9  3.175e-07  1.349e-07  0  0``.
_RESPONSE_ROW = re.compile(
    r"^\s+(\d+)\s+(\S+e[-+]\d+)\s+(\S+e[-+]\d+)\s+(\d+)\s+\d+\s*$"
)


def unconverged_response(stdout: str) -> str | None:
    """Read coupled-perturbed equations that ended above their tolerance.

    Psi4 1.11's CPHF solver for analytic Hessians stops at ``solver_maxiter`` and goes
    on with what it has. Each solve's last row counts the equations that remain;
    a solve that printed no row stopped before finishing one iteration.

    Returns:
        The failure's message, or None when every solve converged.
    """
    for block in re.split(r"==> Coupled-Perturbed \w+ Solver <==", stdout)[1:]:
        rows = []
        for line in block.split("##")[0].splitlines():
            row = _RESPONSE_ROW.match(line)
            if row:
                rows.append(row)
        total = re.search(r"Number of equations\s*=\s*(\d+)", block)
        tolerance = re.search(r"Convergence\s*=\s*(\S+)", block)
        remain = int(rows[-1].group(4)) if rows else int(total.group(1)) if total else 1
        if remain:
            reached = f", max RMS {rows[-1].group(3)}" if rows else ""
            return (
                f"Psi4's coupled-perturbed equations did not converge: {remain} "
                f"equations remained above the tolerance "
                f"{tolerance.group(1) if tolerance else ''} after {len(rows)} "
                f"iterations{reached}."
            )
    return None


@codec.warning
def warnings(ctx: Context) -> list[ProgramWarning]:
    """Warn ``loosely_converged`` when Psi4 carried on with an unconverged SCF.

    Psi4 does that only with ``fail_on_maxiter`` false; the message gives the last
    iteration's energy change and orbital gradient against Psi4's thresholds.
    """
    stdout = ctx.stdout or ""
    accepted = re.findall(
        r"Could not converge SCF iterations in (\d+) iterations\.\s*\n.*proceeding "
        r"anyway",
        stdout,
    )
    if not accepted:
        return []
    iteration = re.findall(r"@\S+ iter\s+\d+:\s+\S+\s+(\S+)\s+(\S+)", stdout)
    e_tol = re.findall(r"Energy threshold\s*=\s*(\S+)", stdout)
    d_tol = re.findall(r"Density threshold\s*=\s*(\S+)", stdout)
    reached = ""
    if iteration and e_tol and d_tol:
        delta, gradient = iteration[-1]
        reached = (
            f": its last iteration changed the energy by {delta} Eh (threshold "
            f"{e_tol[-1]}) with an orbital gradient of {gradient} (threshold "
            f"{d_tol[-1]})"
        )
    count = f"{len(accepted)} SCFs" if len(accepted) > 1 else "the SCF"
    return [
        ProgramWarning(
            code="loosely_converged",
            message=(
                f"Psi4 did not converge {count} in {accepted[-1]} iterations and "
                f"carried on because fail_on_maxiter is false{reached}."
            ),
        )
    ]


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
        structure = ctx.input_data.require_structure()
        modes = unit_modes(results["normal_modes"])
        fields["vibrational_analysis"] = vibrations(
            results["frequencies"],
            modes,
            structure,
            *_projection(np.asarray(modes), structure),
        )
    return fields


def _projection(modes: np.ndarray, structure: Structure) -> tuple[str, list[bool]]:
    """What Psi4's harmonic analysis projected out, and which modes are external.

    Psi4 projects rotations out only at what it calls a stationary point (gradient
    rms below 1e-2 Eh/bohr, unless ``project_rot`` says otherwise) and translations
    unless something breaks translational invariance, and it returns every mode it
    did not project. So the number of modes tells the projection. Modes it left in
    that are rotations (or translations) are flagged: those with the largest share
    in the mass-weighted rigid-body space, as many as that space has dimensions.

    Args:
        modes: Unit-length Cartesian displacements, n_modes x n_atoms x 3.
        structure: The input structure.

    Raises:
        ParserError: The number of modes fits no projection.
    """
    masses = np.array([float(periodic_table[s].mass) for s in structure.symbols])
    weights = np.repeat(np.sqrt(masses), 3)
    geometry = np.asarray(structure.geometry, dtype=float)
    center = masses @ geometry / masses.sum()
    translations = [np.tile(e, len(masses)) for e in np.eye(3)]
    rotations = [np.cross(e, geometry - center).ravel() for e in np.eye(3)]
    rotation_space = _basis(np.array(rotations).T * weights[:, None])
    rigid = 3 + rotation_space.shape[1]  # 3 for an atom, 5 for a linear molecule
    projected = 3 * len(masses) - len(modes)
    if projected >= rigid:
        return "translations_rotations", []
    if projected == 3:
        projection, space = "translations", rotation_space
    elif projected == 0:
        projection = "none"
        space = _basis(np.array(translations + rotations).T * weights[:, None])
    else:
        raise ParserError(
            f"Psi4 returned {len(modes)} modes for {len(masses)} atoms, which no "
            "projection of translations and rotations leaves."
        )
    weighted = modes.reshape(len(modes), -1) * weights
    weighted /= np.linalg.norm(weighted, axis=1)[:, None]
    share = np.linalg.norm(weighted @ space, axis=1)
    external = set(np.argsort(-share)[: space.shape[1]].tolist())
    return projection, [i in external for i in range(len(modes))]


def _basis(vectors: np.ndarray) -> np.ndarray:
    """An orthonormal basis (columns) of the space the columns of ``vectors`` span."""
    u, s, _ = np.linalg.svd(vectors, full_matrices=False)
    return u[:, s > 1e-8 * s.max()]


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
