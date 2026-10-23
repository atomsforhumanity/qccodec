"""Sella: minimizations and transition-state searches driving another program.

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

Calculation types: ``optimization`` (``order=0``) and ``transition_state``
(``order=1``).

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
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from qcdata import (
    CalcType,
    Data,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    SinglePointData,
    Structure,
)
from qcref import constants as c

from ..codec import Codec
from ..exceptions import CalculationError, EncoderError
from ..native import ComputeChild, NativeOutput, PythonInput, PythonOutput
from .geometric import _attach_hessians, _child_resources


class SellaOutcome(TypedDict):
    """The recorded trajectory, Sella's version, and why it stopped, if not converged."""

    trajectory: list[ProgramOutput[ProgramInput, SinglePointData]]
    version: str | None
    error: str | None


# Arguments of Sella.run; every other keyword goes to the Sella constructor.
RUN_KEYWORDS = frozenset({"fmax", "steps"})
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
        keywords.pop("constraints", None), len(inp.structure.symbols)
    )
    run_kwargs = {k: keywords.pop(k) for k in RUN_KEYWORDS & set(keywords)}
    threads, gpus = _child_resources(gradient_spec, inp)
    return PythonInput(
        function=invoke_sella,
        kwargs={
            "structure": inp.structure.model_copy(deep=True),
            "structures": deepcopy(inp.structures),
            "gradient_spec": gradient_spec.model_copy(deep=True),
            "hessian_spec": hessian_spec.model_copy(deep=True)
            if hessian_spec
            else None,
            "order": 1 if inp.calctype == CalcType.transition_state else 0,
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
    constraints: dict[str, list[dict[str, Any]]],
    init_kwargs: dict[str, Any],
    run_kwargs: dict[str, Any],
    compute_child: ComputeChild,
) -> SellaOutcome:
    """Run Sella, retaining its trajectory even when a child or Sella fails."""
    from sella import Constraints, Sella

    from ..ase import ProgramCalculator, atoms_structure, hessian_to_ase

    trajectory: list[ProgramOutput[ProgramInput, SinglePointData]] = []
    hessians: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    outcome: SellaOutcome = {
        "trajectory": trajectory,
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
                trajectory.append(
                    cast(ProgramOutput[ProgramInput, SinglePointData], output)
                )
                raise RuntimeError(
                    f"The {hessian_spec.program} Hessian calculation failed."
                )
            hessians.append(
                (child.structure.geometry.flatten(), np.array(data.hessian))
            )
            return hessian_to_ase(data.hessian)

        optimizer = Sella(
            atoms,
            order=order,
            constraints=sella_constraints,  # type: ignore[arg-type]  # None is its default
            hessian_function=hessian_function if hessian_spec else None,
            **init_kwargs,
        )

        def record_step() -> None:
            """Add the calculation at Sella's current geometry to the trajectory."""
            here = atoms.get_positions().flatten() * c.ANGSTROM_TO_BOHR
            for output in reversed(calculator.outputs):
                geometry = output.input_data.structure.geometry.flatten()
                if np.allclose(geometry, here, atol=1e-8):
                    trajectory.append(
                        cast(ProgramOutput[ProgramInput, SinglePointData], output)
                    )
                    return

        optimizer.attach(record_step, interval=1)
        converged = optimizer.run(**run_kwargs)
        if not converged:
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
            if not any(frame is failed for frame in trajectory):
                trajectory.append(
                    cast(ProgramOutput[ProgramInput, SinglePointData], failed)
                )
        _attach_hessians(trajectory, hessians)
    return outcome


def decode(inp: ProgramInput, out: NativeOutput) -> OptimizationData:
    """Wrap the recorded trajectory; raise with it attached if Sella failed."""
    assert isinstance(out, PythonOutput)
    native = out.value or {"trajectory": [], "version": None, "error": None}
    data = OptimizationData(
        provenance=Provenance(program="sella", program_version=native["version"]),
        trajectory=native["trajectory"],
    )
    if native["error"]:
        raise CalculationError(native["error"], data=data)
    return data


codec = Codec(
    "sella",
    calctypes={CalcType.optimization, CalcType.transition_state},
    encode=encode,
    decode=decode,
)
