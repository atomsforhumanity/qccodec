"""Sella: minimizations, transition-state searches, and IRCs driving another program.

Sella (https://github.com/zadorlab/sella) is a Python optimizer built on ASE. It asks
an ASE calculator for energies and forces; here that calculator is
``qccodec.ase.ProgramCalculator``, so every step is a full ``gradient`` calculation of
the program named in ``subprograms``, run by qccompute like any other calculation and
kept in the trajectory. Any codec with gradients works as the child, and no ASE
calculator is involved.

Inputs map to Sella as follows:

- ``subprograms`` holds one ``ProgramSpec`` with ``calctype="gradient"``: the child
  program, its model, and its own keywords (passed to that program unchanged).
  Optionally a second one with ``calctype="hessian"`` becomes Sella's
  ``hessian_function``: Sella then starts from that program's Hessian instead of
  estimating curvature from gradients, and recomputes it every ``diag_every_n`` steps
  if that keyword is set. The Hessian program needs analytic or cheap Hessians to pay
  off (xtb, AIMNet2, MACE, ...).
- ``keywords`` are Sella's own, passed unchanged in Sella's units (eV and angstrom):
  ``fmax`` and ``steps`` go to ``Sella.run``, everything else (``eta``, ``gamma``,
  ``internal``, ``delta0``, ``diag_every_n``, ``nsteps_per_diag``, ...) to the
  ``Sella`` constructor. Sella's defaults apply to anything left out, including
  ``fmax=0.05`` eV/angstrom and an effectively unlimited ``steps``, so set ``steps``.
- ``constraints`` maps Sella's ``Constraints`` methods to their calls:
  ``fix_translation``, ``fix_bond``, ``fix_angle``, ``fix_dihedral``. Each call is the
  first argument (atom indices) or a dictionary of keyword arguments, e.g.
  ``{"fix_translation": [0], "fix_bond": [[0, 1], {"indices": [1, 2], "target": 1.0}]}``.
  Atom indices are 0-based and targets in angstrom and degrees, as in Sella.
- The codec sets ``order`` from the calculation type and keeps Sella's log on stdout;
  ``order``, ``logfile``, ``trajectory``, ``restart``, ``append_trajectory``,
  ``hessian_function``, and ``cmdline_args`` are not accepted.

Calculation types: ``optimization`` (``order=0``), ``transition_state``
(``order=1``), and ``irc`` (Sella's ``IRC`` from the transition state in
``structure``, both directions). For an IRC, ``fmax``, ``fmax_inner``, and ``steps`` go
to ``IRC.run`` and the rest (``dx``, ``keep_going``, ``eta``, ``gamma``, ...) to
``IRC``; ``direction`` is set by the codec. The result is a ``PathData``: frames from
the end of the reverse branch, through the transition state (``ts_index``), to the
end of the forward branch. Two corrections to Sella keep the path meaningful. Sella's
IRC keeps overall translation and rotation, so a branch drifts and can spin at its
minimum: each frame is moved onto the previous one by the mass-weighted best fit,
its gradient rotated with it, which keeps the path in the input's frame. And a
branch ends when the curvature of the vibrations is positive, not of the full
Hessian, whose near-zero rigid-motion eigenvalues noise can push negative (Sella
would otherwise keep rotating the molecule until ``steps`` runs out). With
constraints, Sella's own test applies.

The trajectory holds one frame per optimizer step. Gradients Sella evaluates at
displaced geometries to probe the curvature are not steps and are left out, unless one
fails. Each Hessian from the Hessian program is attached to the frame at its geometry
(``frame.results.hessian``). Sella removes overall translation and rotation from its
steps, so every frame stays in the input's frame. A run that uses up ``steps`` without
meeting ``fmax`` fails, with its trajectory attached.

With qccompute, ``propagate_wfn=True`` starts each child calculation from the previous
step's orbitals when the child codec supports it (TeraChem). The optimization declares
the threads and GPUs its child does, so lists of optimizations share the machine
correctly.
"""

from __future__ import annotations

import traceback
from copy import deepcopy
from importlib import metadata
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from qcdata import (
    CalcType,
    Data,
    Frame,
    OptimizationData,
    PathData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    SinglePointData,
    Structure,
)
from qcref import constants as c
from qcref import periodic_table

from ..codec import Codec
from ..exceptions import CalculationError, EncoderError
from ..native import ComputeChild, NativeOutput, PythonInput, PythonOutput
from .geometric import _attach_hessians, _child_resources


class SellaOutcome(TypedDict):
    """The recorded trajectory, Sella's version, and why it stopped, if not converged."""

    trajectory: list[Frame]
    ts_index: int | None
    version: str | None
    error: str | None


# Arguments of Sella.run (IRC.run also takes fmax_inner); every other keyword goes to
# the Sella or IRC constructor.
RUN_KEYWORDS = frozenset({"fmax", "steps"})
IRC_RUN_KEYWORDS = RUN_KEYWORDS | {"fmax_inner"}
# Set by the codec or meaningless here.
OWNED_KEYWORDS = frozenset(
    {
        "atoms",
        "order",
        "logfile",
        "trajectory",
        "restart",
        "append_trajectory",
        "hessian_function",
        "master",
        "direction",
    }
)
# Sella Constraints methods taking atom indices, and what each call names.
# (argument name, number of atoms; None for one atom or several)
CONSTRAINT_METHODS: dict[str, tuple[str, int | None]] = {
    "fix_translation": ("index", None),
    "fix_bond": ("indices", 2),
    "fix_angle": ("indices", 3),
    "fix_dihedral": ("indices", 4),
}


def encode(inp: ProgramInput) -> PythonInput:
    """Prepare Sella's arguments; the child programs come from ``subprograms``."""
    try:
        gradient_spec = inp.get_subprogram(CalcType.gradient)
    except ValueError as exc:
        raise EncoderError(str(exc)) from exc
    hessian_spec = next(
        (s for s in inp.subprograms if s.calctype == CalcType.hessian), None
    )
    others = [
        s
        for s in inp.subprograms
        if s.calctype not in (CalcType.gradient, CalcType.hessian)
    ]
    if others:
        raise EncoderError(
            "Sella takes a gradient subprogram and optionally a hessian subprogram."
        )
    if inp.cmdline_args:
        raise EncoderError("Sella's Python interface takes no command-line arguments.")
    keywords = deepcopy(inp.keywords)
    owned = sorted(OWNED_KEYWORDS & set(keywords))
    if owned:
        raise EncoderError(f"The sella codec sets these keywords itself: {owned}.")
    constraints = _constraints(
        keywords.pop("constraints", None), len(inp.require_structure().symbols)
    )
    irc = inp.calctype == CalcType.irc
    run_keywords = IRC_RUN_KEYWORDS if irc else RUN_KEYWORDS
    run_kwargs = {k: keywords.pop(k) for k in run_keywords & set(keywords)}
    threads, gpus = _child_resources(gradient_spec, inp)
    return PythonInput(
        function=invoke_sella,
        kwargs={
            "structure": inp.require_structure().model_copy(deep=True),
            "structures": deepcopy(inp.structures),
            "gradient_spec": gradient_spec.model_copy(deep=True),
            "hessian_spec": hessian_spec.model_copy(deep=True)
            if hessian_spec
            else None,
            "order": 1 if inp.calctype == CalcType.transition_state else 0,
            "irc": irc,
            "constraints": constraints,
            "init_kwargs": keywords,
            "run_kwargs": run_kwargs,
        },
        files=dict(inp.files),
        child_callback="compute_child",
        threads=threads,
        gpus=gpus,
    )


def _constraints(value: Any, natoms: int) -> dict[str, list[dict[str, Any]]]:
    """Validate the ``constraints`` keyword; return each call as keyword arguments."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise EncoderError("constraints must map Sella Constraints methods to calls.")
    calls: dict[str, list[dict[str, Any]]] = {}
    for method, items in value.items():
        if method not in CONSTRAINT_METHODS:
            raise EncoderError(
                f"Unknown Sella constraint {method!r}; use one of "
                f"{sorted(CONSTRAINT_METHODS)}."
            )
        if not isinstance(items, list):
            raise EncoderError(f"constraints[{method!r}] must be a list of calls.")
        name, count = CONSTRAINT_METHODS[method]
        calls[method] = []
        for item in items:
            kwargs = dict(item) if isinstance(item, dict) else {name: item}
            indices = kwargs.get(name)
            flat = [indices] if isinstance(indices, int) and count is None else indices
            if (
                not isinstance(flat, list)
                or not all(isinstance(i, int) and 0 <= i < natoms for i in flat)
                or (count is not None and len(flat) != count)
            ):
                raise EncoderError(
                    f"constraints[{method!r}] needs {count or 'one or more'} 0-based "
                    f"atom indices below {natoms}, not {indices!r}."
                )
            if isinstance(indices, list):
                kwargs[name] = tuple(indices)
            calls[method].append(kwargs)
    return calls


def invoke_sella(
    *,
    structure: Structure,
    structures: dict[str, Structure],
    gradient_spec: ProgramSpec,
    hessian_spec: ProgramSpec | None,
    order: int,
    irc: bool,
    constraints: dict[str, list[dict[str, Any]]],
    init_kwargs: dict[str, Any],
    run_kwargs: dict[str, Any],
    compute_child: ComputeChild,
) -> SellaOutcome:
    """Run Sella, retaining its trajectory even when a child or Sella fails.

    For an IRC, both directions run from the transition state with one Sella ``IRC``,
    so the transition state is analyzed once; the trajectory is then the path from the
    reverse end, through the transition state, to the forward end.
    """
    from sella import IRC, Constraints, Sella

    from ..ase import ProgramCalculator, atoms_structure, hessian_to_ase

    trajectory: list[Frame] = []
    recorded: set[int] = set()  # ids of the outputs already in the trajectory
    hessians: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    # Where record_step adds frames: the trajectory, or an IRC's current branch.
    target: list[list[Frame]] = [trajectory]
    branches: dict[str, list[Frame]] = {}
    outcome: SellaOutcome = {
        "trajectory": trajectory,
        "ts_index": None,
        "version": metadata.version("sella"),
        "error": None,
    }
    previous: list[ProgramOutput[ProgramInput, Data]] = []

    def compute(child: ProgramInput) -> ProgramOutput[ProgramInput, Data]:
        """Run a child, starting from the last one's wavefunction when propagating."""
        output = compute_child(child, previous=previous[-1] if previous else None)
        if output.success:
            previous[:] = [output]
        return output

    calculator = ProgramCalculator(
        gradient_spec, compute, structure=structure, structures=structures
    )
    try:
        from ase import Atoms

        atoms = Atoms(
            symbols=list(structure.symbols),
            positions=structure.geometry * c.BOHR_TO_ANGSTROM,
        )
        atoms.calc = calculator
        sella_constraints = None
        if constraints:
            sella_constraints = Constraints(atoms)
            for method, calls in constraints.items():
                for kwargs in calls:
                    getattr(sella_constraints, method)(**kwargs)

        def hessian_function(at: Atoms) -> NDArray[np.float64]:
            """Compute the Hessian program at Sella's geometry, in eV/angstrom^2."""
            assert hessian_spec is not None
            child = ProgramInput.from_spec(
                hessian_spec, atoms_structure(at, structure), structures=structures
            )
            output = compute(child)
            data = output.results
            if (
                not output.success
                or not isinstance(data, SinglePointData)
                or data.hessian is None
            ):
                trajectory.append(Frame.from_output(output))
                recorded.add(id(output))
                raise RuntimeError(
                    f"The {hessian_spec.program} Hessian calculation failed."
                )
            hessians.append(
                (child.require_structure().geometry.flatten(), np.array(data.hessian))
            )
            return hessian_to_ase(data.hessian)

        hessian = hessian_function if hessian_spec else None
        optimizer: Any
        if irc:
            optimizer = _vibrational_irc(IRC, constrained=bool(constraints))(
                atoms,
                constraints=sella_constraints,
                hessian_function=hessian,
                **init_kwargs,
            )
        else:
            optimizer = Sella(
                atoms,
                order=order,
                constraints=sella_constraints,  # type: ignore[arg-type]  # None is default
                hessian_function=hessian,
                **init_kwargs,
            )

        def record_step() -> None:
            """Add the calculation at Sella's current geometry to the trajectory."""
            here = atoms.get_positions().flatten() * c.ANGSTROM_TO_BOHR
            for output in reversed(calculator.outputs):
                geometry = output.input_data.require_structure().geometry.flatten()
                if np.allclose(geometry, here, atol=1e-8):
                    target[0].append(Frame.from_output(output))
                    recorded.add(id(output))
                    return

        optimizer.attach(record_step, interval=1)
        if irc:
            for direction in ("reverse", "forward"):
                target[0] = branches[direction] = []
                if not optimizer.run(direction=direction, **run_kwargs):
                    outcome["error"] = (
                        f"Sella's IRC did not reach a minimum in the {direction} "
                        f"direction within {optimizer.nsteps} steps."
                    )
                    break
        elif not optimizer.run(**run_kwargs):
            outcome["error"] = (
                f"Sella did not converge to fmax={optimizer.fmax} eV/angstrom in "
                f"{optimizer.nsteps} steps."
            )
    except Exception:
        outcome["error"] = traceback.format_exc()
    finally:
        # A failed gradient ends the run without becoming a step; keep it.
        if calculator.outputs and not calculator.outputs[-1].success:
            failed = calculator.outputs[-1]
            if id(failed) not in recorded:
                target[0].append(Frame.from_output(failed))
        if irc:
            trajectory[:], outcome["ts_index"] = _irc_path(branches)
        _attach_hessians(trajectory, hessians)
    return outcome


def _vibrational_irc(base: type[Any], *, constrained: bool) -> type[Any]:
    """Sella's IRC, judging arrival at a minimum on the vibrations alone.

    Sella ends a direction once its Hessian's lowest eigenvalue is positive. An IRC
    keeps overall translation and rotation in that Hessian, and their near-zero
    eigenvalues can fall below zero from numerical noise: at the minimum Sella then
    keeps stepping along a rotation until ``steps`` runs out. Here the eigenvalues
    are those of the Hessian projected onto the vibrations. With constraints, rigid
    motions are real and Sella's own test applies.
    """

    class VibrationalIRC(base):  # type: ignore[misc, valid-type]
        def converged(self, forces: Any = None) -> bool:
            if self.first or constrained:
                return bool(super().converged(forces))
            if self.pes.H.evals is None or not self.pes.converged(self.fmax)[0]:
                return False
            rigid = _rigid_basis(self.atoms.get_positions())
            u, _, _ = np.linalg.svd(rigid, full_matrices=True)
            vibrations = u[:, rigid.shape[1] :]
            hessian = np.asarray(self.pes.H.asarray(), dtype=float)
            lowest = np.linalg.eigvalsh(vibrations.T @ hessian @ vibrations)[0]
            return bool(lowest > 0)

    return VibrationalIRC


def _rigid_basis(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """An orthonormal basis of the rigid translations and rotations (3N columns)."""
    centered = positions - positions.mean(axis=0)
    vectors = []
    for axis in np.eye(3):
        vectors.append(np.tile(axis, len(positions)))
        vectors.append(np.cross(axis, centered).ravel())
    u, s, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
    return u[:, s > 1e-6 * s.max()]


def _irc_path(branches: dict[str, list[Frame]]) -> tuple[list[Frame], int | None]:
    """Join the branches: reverse end, ..., transition state, ..., forward end.

    Each branch starts with the transition state, where Sella evaluates first.
    """
    reverse = branches.get("reverse", [])
    forward = branches.get("forward", [])
    ts = reverse[:1] or forward[:1]
    if not ts:
        return [], None
    # Both branches align onto the same transition-state frame.
    reverse = _aligned(ts + reverse[1:])
    forward = _aligned(ts + forward[1:])
    path = reverse[:0:-1] + ts + forward[1:]
    return path, len(reverse) - 1


def _aligned(branch: list[Frame]) -> list[Frame]:
    """Remove the rigid drift of an IRC branch: each frame onto the one before it.

    Sella's IRC keeps overall translation and rotation, so a branch can drift and, at
    its minimum, spin. Each frame is moved onto the previous (already aligned) frame
    by the mass-weighted best fit (centers of mass together, then the rotation that
    minimizes the mass-weighted displacement, as an IRC step has no rigid part), its
    gradient rotated with it. The path then stays in the transition state's frame,
    which is the input's.
    """
    if not branch:
        return branch
    masses = np.array(
        [float(periodic_table[s].mass) for s in branch[0].structure.symbols]
    )[:, None]
    aligned = [branch[0]]
    previous = np.asarray(branch[0].structure.geometry, dtype=float)
    for frame in branch[1:]:
        coords = np.asarray(frame.structure.geometry, dtype=float)
        center = (masses * coords).sum(axis=0) / masses.sum()
        target = (masses * previous).sum(axis=0) / masses.sum()
        p = coords - center
        u, _, vt = np.linalg.svd((masses * p).T @ (previous - target))
        rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        previous = p @ rotation + target
        results = frame.results
        if results.gradient is not None:
            gradient = np.asarray(results.gradient) @ rotation
            results = results.model_copy(update={"gradient": gradient})
        structure = frame.structure.model_copy(update={"geometry": previous})
        aligned.append(
            Frame.model_validate(
                {**dict(frame), "structure": structure, "results": results}
            )
        )
    return aligned


def decode(inp: ProgramInput, out: NativeOutput) -> OptimizationData | PathData:
    """Wrap the recorded trajectory; raise with it attached if Sella failed."""
    assert isinstance(out, PythonOutput)
    native = out.value or {
        "trajectory": [],
        "ts_index": None,
        "version": None,
        "error": None,
    }
    provenance = Provenance(program="sella", program_version=native["version"])
    data: OptimizationData | PathData
    if inp.calctype == CalcType.irc:
        data = PathData(
            provenance=provenance,
            frames=native["trajectory"],
            ts_index=native["ts_index"],
        )
    else:
        data = OptimizationData(provenance=provenance, frames=native["trajectory"])
    if native["error"]:
        raise CalculationError(native["error"], data=data)
    return data


codec = Codec(
    "sella",
    calctypes={CalcType.optimization, CalcType.transition_state, CalcType.irc},
    encode=encode,
    decode=decode,
)
