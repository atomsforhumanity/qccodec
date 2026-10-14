"""geomeTRIC's native engine bridge; child gradients run through an injected callback."""

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
    if coordsys not in {"cart", "prim", "dlc", "hdlc", "tric"}:
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
            """Evaluate a trial geometry and keep its child output in the trajectory."""
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
    outcome: GeometricOutcome = {
        "trajectory": trajectory,
        "version": geometric.__version__,
        "error": None,
    }
    try:
        molecule = geometric.molecule.Molecule(fnm=geometry_file)
        coord_classes = {
            "cart": (geometric.internal.CartesianCoordinates, False, False),
            "prim": (geometric.internal.PrimitiveInternalCoordinates, True, False),
            "dlc": (geometric.internal.DelocalizedInternalCoordinates, True, False),
            "hdlc": (geometric.internal.DelocalizedInternalCoordinates, False, False),
            "tric": (geometric.internal.DelocalizedInternalCoordinates, False, False),
        }
        coord_class, connect, addcart = coord_classes[coordsys]
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
    return outcome


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
