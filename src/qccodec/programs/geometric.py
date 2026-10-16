"""geomeTRIC: optimizations and transition-state searches driving another program.

geomeTRIC runs in Python and asks for gradients through a callback: every step is a
full ``gradient`` calculation of the program named in ``subprograms``, run by qccompute
like any other calculation and kept in the trajectory. Any codec with gradients works
as the child, including ones registered with ``qccodec.register``.

Inputs map to geomeTRIC as follows:

- ``subprograms`` must hold one ``ProgramSpec`` with ``calctype="gradient"``: the child
  program, its model, and its own keywords (passed to that program unchanged).
- ``keywords`` are geomeTRIC's optimizer parameters (``geometric.params.OptParams``):
  ``maxiter``, ``convergence_set``, ``convergence_energy``, ``trust``, ``hessian``,
  ``enforce``, ... Two are handled by the codec: ``coordsys`` (``tric`` by default, or
  ``cart``, ``prim``, ``dlc``, ``hdlc``) and ``constraints`` (geomeTRIC's constraint
  dictionary, e.g. ``{"freeze": [{"type": "xyz", "indices": [0]}]}``, with ``set``
  distances in bohr; scans are not supported).
- ``cmdline_args`` are not accepted.

Calculation types: ``optimization`` and ``transition_state`` (``transition=True``;
geomeTRIC computes a starting Hessian from finite differences of child gradients).

The trajectory holds one frame per optimizer step. Gradients at the displaced
geometries of a finite-difference Hessian are not steps, so they are left out, unless
one fails. Each Hessian geomeTRIC computes is attached to the frame at its geometry
(``frame.results.hessian``): the first frame for ``hessian="first"``, the default for
transition states, and also the last frame for ``"first+last"`` or ``"last"``.
Atom indices in ``constraints`` are 0-based, as in geomeTRIC's Python interface.

With qccompute, ``propagate_wfn=True`` starts each child calculation from the previous
step's orbitals when the child codec supports it (TeraChem).
"""

from __future__ import annotations

import traceback
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypedDict, cast

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

from ..codec import Codec
from ..exceptions import CalculationError, EncoderError
from ..native import ComputeChild, NativeOutput, PythonInput, PythonOutput

if TYPE_CHECKING:
    from geometric.engine import Engine as NativeEngine
    from geometric.molecule import Molecule


class GeometricOutcome(TypedDict):
    """Optimizer completion state plus the independently recorded child outputs."""

    trajectory: list[ProgramOutput[ProgramInput, SinglePointData]]
    version: str | None
    error: str | None


# geomeTRIC computes finite-difference Hessians in directories below this one.
HESSIAN_DISPLACEMENT = "hessian/displace"

# coordsys -> (geometric.internal class, connect, addcart), as in geomeTRIC's run_json.
COORDINATE_SYSTEMS: dict[str, tuple[str, bool, bool]] = {
    "cart": ("CartesianCoordinates", False, False),
    "prim": ("PrimitiveInternalCoordinates", True, False),
    "dlc": ("DelocalizedInternalCoordinates", True, False),
    "hdlc": ("DelocalizedInternalCoordinates", False, True),
    "tric": ("DelocalizedInternalCoordinates", False, False),
}


def encode(inp: ProgramInput) -> PythonInput:
    """Prepare optimizer arguments; the gradient child comes from ``subprograms``."""
    try:
        gradient_spec = inp.get_subprogram(CalcType.gradient)
    except ValueError as exc:
        raise EncoderError(str(exc)) from exc
    if inp.cmdline_args:
        raise EncoderError(
            "geomeTRIC's Python interface does not accept command-line arguments."
        )
    keywords = deepcopy(inp.keywords)
    keywords["transition"] = inp.calctype == CalcType.transition_state
    coordsys = keywords.pop("coordsys", "tric")
    if coordsys not in COORDINATE_SYSTEMS:
        raise EncoderError(f"Unsupported coordinate system: {coordsys}")
    constraints = keywords.pop("constraints", None)
    if constraints and "scan" in constraints:
        raise EncoderError(
            "Constraint scans are not supported by this geomeTRIC integration."
        )
    filename = "initial_geometry.xyz"
    if filename in inp.files:
        raise EncoderError(f"Supplied files conflict with generated input {filename}.")
    return PythonInput(
        function=invoke_geometric,
        kwargs={
            "structure": inp.structure.model_copy(deep=True),
            "structures": deepcopy(inp.structures),
            "gradient_spec": gradient_spec.model_copy(deep=True),
            "coordsys": coordsys,
            "constraints": constraints,
            "keywords": keywords,
            "geometry_file": filename,
        },
        files={**inp.files, filename: inp.structure.to_xyz()},
        child_callback="compute_child",
        loggers=("geometric",),
    )


def make_engine(
    geometric: ModuleType,
    molecule: Molecule,
    structure: Structure,
    structures: dict[str, Structure],
    gradient_spec: ProgramSpec,
    compute_child: ComputeChild,
    trajectory: list[ProgramOutput[ProgramInput, SinglePointData]],
) -> NativeEngine:
    """Bind geomeTRIC's native engine interface to independent child calculations."""
    base_engine: Any = geometric.engine.Engine

    class Engine(base_engine):
        def calc_new(
            self, coords: NDArray[np.float64], *args: object
        ) -> dict[str, float | NDArray[np.float64]]:
            """Evaluate a geometry; optimizer steps join the trajectory.

            Gradients at the displaced geometries of a finite-difference Hessian
            (geomeTRIC runs them in ``hessian/displace/...``) are not optimizer steps,
            so they stay out of the trajectory unless they fail.
            """
            dirname = str(args[0]) if args else ""
            displacement = HESSIAN_DISPLACEMENT in dirname.replace("\\", "/")
            updated = Structure.model_validate(
                {**structure.model_dump(), "geometry": coords}
            )
            child = ProgramInput.from_spec(
                gradient_spec, updated, structures=structures
            )
            output = compute_child(
                child,
                previous=cast(ProgramOutput[ProgramInput, Data], trajectory[-1])
                if trajectory
                else None,
            )
            if not displacement or not output.success:
                trajectory.append(
                    cast(ProgramOutput[ProgramInput, SinglePointData], output)
                )
            if not output.success:
                raise RuntimeError(
                    "A child calculation failed during geomeTRIC optimization."
                )
            data = output.results
            if (
                not isinstance(data, SinglePointData)
                or data.energy is None
                or data.gradient is None
            ):
                raise RuntimeError("geomeTRIC requires both an energy and a gradient.")
            return {"energy": data.energy, "gradient": data.gradient.flatten()}

    return Engine(molecule)


def invoke_geometric(
    *,
    structure: Structure,
    structures: dict[str, Structure],
    gradient_spec: ProgramSpec,
    coordsys: str,
    constraints: dict[str, Any] | None,
    keywords: dict[str, Any],
    geometry_file: str,
    compute_child: ComputeChild,
) -> GeometricOutcome:
    """Run the optimizer, retaining its trajectory even when a child or optimizer fails."""
    import geometric

    trajectory: list[ProgramOutput[ProgramInput, SinglePointData]] = []
    hessians: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    compute_hessian = None
    outcome: GeometricOutcome = {
        "trajectory": trajectory,
        "version": geometric.__version__,
        "error": None,
    }
    try:
        molecule = geometric.molecule.Molecule(fnm=geometry_file)
        class_name, connect, addcart = COORDINATE_SYSTEMS[coordsys]
        coord_class = getattr(geometric.internal, class_name)
        parsed_constraints, values = None, None
        if constraints:
            parsed_constraints, values = geometric.prepare.parse_constraints(
                molecule, geometric.run_json.make_constraints_string(constraints)
            )
        coords = coord_class(
            molecule,
            build=True,
            connect=connect,
            addcart=addcart,
            constraints=parsed_constraints,
            cvals=values[0] if values is not None else None,
        )
        engine = make_engine(
            geometric,
            molecule,
            structure,
            structures,
            gradient_spec,
            compute_child,
            trajectory,
        )
        # Record every Cartesian Hessian geomeTRIC computes, with its geometry.
        compute_hessian = geometric.optimize.calc_cartesian_hessian

        def recording_hessian(
            coords: NDArray[np.float64], *args: Any, **kwargs: Any
        ) -> NDArray[np.float64]:
            """Compute the Hessian as geomeTRIC would and remember it."""
            hessian = compute_hessian(coords, *args, **kwargs)
            hessians.append((np.array(coords, copy=True), np.array(hessian, copy=True)))
            return hessian

        geometric.optimize.calc_cartesian_hessian = recording_hessian
        optimizer = geometric.optimize.Optimizer(
            structure.geometry.flatten(),
            molecule,
            coords,
            engine=engine,
            dirname=Path.cwd(),
            params=geometric.params.OptParams(
                **{"xyzout": "qcdata_optim.xyz", **keywords}
            ),
        )
        optimizer.optimizeGeometry()
    except Exception:
        outcome["error"] = traceback.format_exc()
    finally:
        if compute_hessian is not None:
            geometric.optimize.calc_cartesian_hessian = compute_hessian
        _attach_hessians(trajectory, hessians)
    return outcome


def _attach_hessians(
    trajectory: list[ProgramOutput[ProgramInput, SinglePointData]],
    hessians: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> None:
    """Put each Hessian on the trajectory frame computed at the same geometry.

    With ``hessian="first"`` (the default for transition states) that is the first
    frame; ``"last"`` and ``"first+last"`` also give the final frame one.
    """
    for coords, hessian in hessians:
        for i in range(len(trajectory) - 1, -1, -1):
            frame = trajectory[i]
            geometry = frame.input_data.structure.geometry.flatten()
            if geometry.shape == coords.shape and np.allclose(geometry, coords):
                results = frame.results.model_copy(update={"hessian": hessian})
                trajectory[i] = frame.model_copy(update={"results": results})
                break


def decode(inp: ProgramInput, out: NativeOutput) -> OptimizationData:
    """Wrap the recorded trajectory; raise with it attached if the optimizer failed."""
    assert isinstance(out, PythonOutput)
    native = out.value or {"trajectory": [], "version": None, "error": None}
    data = OptimizationData(
        provenance=Provenance(program="geometric", program_version=native["version"]),
        trajectory=native["trajectory"],
    )
    if native["error"]:
        raise CalculationError(native["error"], data=data)
    return data


codec = Codec(
    "geometric",
    calctypes={CalcType.optimization, CalcType.transition_state},
    encode=encode,
    decode=decode,
)
