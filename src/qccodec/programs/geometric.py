"""geomeTRIC: optimizations, transition-state searches, and IRCs driving another program.

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
  distances in bohr; scans are not supported). geomeTRIC itself ignores parameters
  it does not know, so the codec refuses them first (kind ``input``, code
  ``unknown_key``, with the closest parameter named), as it refuses those it sets
  itself (``transition``, ``irc``, ``xyzout``), those that only write files it does
  not return (``qdata``, ``write_cart_hess``, ``wigner``), ``meci`` and ``bigchem``;
  ``irc_direction`` outside an IRC is ``not_applicable``.
- ``cmdline_args`` are not accepted.

Calculation types: ``optimization``, ``transition_state`` (``transition=True``;
geomeTRIC computes a starting Hessian from finite differences of child gradients), and
``irc`` (``irc=True``, from the transition state in ``structure``; geomeTRIC computes
its Hessian the same way). An IRC returns a ``PathData``: the path geomeTRIC accepted,
from the end of its backward branch, through the transition state (``ts_index``), to
the end of its forward branch, each frame the child calculation at that point. Its
frames are the accepted points only; rejected trial steps are left out. geomeTRIC
finishes each branch by optimizing to the minimum once the IRC has converged, so the
ends are minima (its ``irc_direction`` keyword can limit the IRC to one branch).

The trajectory holds one frame per structure the optimizer evaluated, the starting
structure first, so n steps give n + 1 frames. Gradients at the displaced
geometries of a finite-difference Hessian are not steps, so they are left out, unless
one fails. Each Hessian geomeTRIC computes is attached to the frame at its geometry
(``frame.results.hessian``): the first frame for ``hessian="first"``, the default for
transition states, and also the last frame for ``"first+last"`` or ``"last"``.
Atom indices in ``constraints`` are 0-based, as in geomeTRIC's Python interface.

Each step is a calculation of the subprogram through the runner (``ctx.compute``), so
the step's program runs wherever the runner places it. Called directly, the program
hands itself to ``qccompute.compute``. The optimization plans the threads and GPUs its
subprogram asks for (its ``threads`` keyword, else its program's plan, whatever program
it is), so lists of optimizations share the machine correctly.

geomeTRIC runs in the caller's process: it writes its files in the calculation
directory, its log (the ``geometric.nifty`` logger) goes to the output's ``logs``
rather than the caller's handlers, and several optimizations run side by side in
threads. Its subprograms are therefore found as any calculation in that process finds
them, including programs registered there with ``qccompute.register_program``. A
timeout stops its subprograms, not geomeTRIC's own Python.

Failures and their kinds: running out of iterations (``maxiter``) is ``convergence``
(code ``max_iter``); an IRC started from a structure that is not a first-order saddle
point is ``input`` (``not_first_order_saddle``); parameters geomeTRIC rejects are
``input``; non-finite gradients are ``numerical``; a failed subprogram is ``program``,
with its error as the cause. The trajectory so far is attached to every failure, and
geomeTRIC's traceback is in the output's ``traceback``.
"""

from __future__ import annotations

import threading
import traceback
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypedDict

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

from ..codec import Codec
from ..exceptions import CalculationError, EncoderError
from ..native import ComputeChild, NativeOutput, PythonInput, PythonOutput
from ..program import CodecProgram
from ._common import align_path, reject_unknown

if TYPE_CHECKING:
    from geometric.engine import Engine as NativeEngine
    from geometric.molecule import Molecule


class GeometricOutcome(TypedDict):
    """Optimizer completion state plus the independently recorded child outputs."""

    trajectory: list[Frame]
    path: list[Frame]
    ts_index: int | None
    version: str | None
    error: str | None
    traceback: str | None


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


# geomeTRIC's optimizer parameters (geometric.params.OptParams 1.1) the codec passes
# on, and its own two.
SETTINGS = frozenset(
    {
        "irc_direction", "enforce", "epsilon", "check", "verbose", "trust", "tmax",
        "tmin", "usedmax", "maxiter", "bothre", "hessian", "hess_data", "frequency",
        "thermo", "ignore_modes", "reset", "subfrctor", "converge", "qccnv",
        "molcnv", "convergence_set", "convergence_energy", "convergence_grms",
        "convergence_gmax", "convergence_drms", "convergence_dmax",
        "convergence_molpro_gmax", "convergence_molpro_dmax", "convergence_cmax",
        "coordsys", "constraints",
    }
)  # fmt: skip
# OptParams the codec refuses, and why.
REFUSED = {
    "transition": "the calctype sets it (transition_state)",
    "irc": "the calctype sets it (irc)",
    "xyzout": "the codec names the trajectory file itself",
    "qdata": "it only writes a file the codec does not return",
    "write_cart_hess": "it only writes a file the codec does not return",
    "wigner": "it only writes files the codec does not return",
    "meci": "a conical-intersection search needs two states, not a gradient program",
    "bigchem": "subprograms run through qccompute",
}
IRC_DIRECTIONS = ("both", "forward", "backward")


def check_keywords(inp: ProgramInput) -> None:
    """Refuse keywords geomeTRIC would ignore or the codec does not pass on.

    Raises:
        EncoderError: Kind ``input``: an unknown keyword (code ``unknown_key``, with
            the closest one), one the codec sets or does not pass on, ``irc_direction``
            outside an IRC (``not_applicable``) or a value geomeTRIC does not take.
    """
    refused = sorted(REFUSED.keys() & inp.keywords.keys())
    if refused:
        raise EncoderError(
            "; ".join(
                f"geomeTRIC keyword {k!r} is not accepted: {REFUSED[k]}"
                for k in refused
            )
            + ".",
            code="unknown_key",
        )
    reject_unknown("geomeTRIC", inp.keywords, SETTINGS)
    direction = inp.keywords.get("irc_direction")
    if direction is not None:
        if inp.calctype != CalcType.irc:
            raise EncoderError(
                "irc_direction applies to calctype irc only.", code="not_applicable"
            )
        if direction not in IRC_DIRECTIONS:
            raise EncoderError(
                f"irc_direction is one of {list(IRC_DIRECTIONS)}, not {direction!r}."
            )
    coordsys = inp.keywords.get("coordsys", "tric")
    if coordsys not in COORDINATE_SYSTEMS:
        raise EncoderError(
            f"coordsys is one of {sorted(COORDINATE_SYSTEMS)}, not {coordsys!r}."
        )


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
    check_keywords(inp)
    keywords = deepcopy(inp.keywords)
    keywords["transition"] = inp.calctype == CalcType.transition_state
    keywords["irc"] = inp.calctype == CalcType.irc
    coordsys = keywords.pop("coordsys", "tric")
    constraints = keywords.pop("constraints", None)
    if constraints and "scan" in constraints:
        raise EncoderError(
            "Constraint scans are not supported by this geomeTRIC integration.",
            kind="unsupported",
        )
    threads, gpus = _child_resources(gradient_spec, inp)
    filename = "initial_geometry.xyz"
    if filename in inp.files:
        raise EncoderError(f"Supplied files conflict with generated input {filename}.")
    return PythonInput(
        function=invoke_geometric,
        kwargs={
            "structure": inp.require_structure().model_copy(deep=True),
            "structures": deepcopy(inp.structures),
            "gradient_spec": gradient_spec.model_copy(deep=True),
            "coordsys": coordsys,
            "constraints": constraints,
            "keywords": keywords,
            "geometry_file": filename,
        },
        files={**inp.files, filename: inp.require_structure().to_xyz()},
        child_callback="compute_child",
        # Every path is in the directory, so geomeTRIC runs in the caller's process.
        directory_keyword="directory",
        # geomeTRIC logs everything through this one logger.
        loggers=("geometric.nifty",),
        # Each step runs the child, so the optimization occupies what it declares.
        threads=threads,
        gpus=gpus,
    )


def _child_resources(spec: ProgramSpec, inp: ProgramInput) -> tuple[int | None, int]:
    """Return the threads and GPUs the gradient subprogram asks for.

    The subprogram's ``threads`` keyword, else its program's plan (None: all the
    runner has), whatever program it is (an external one, linus, axtb, a
    ``FunctionProgram``); one thread and no GPU when it cannot be planned, which
    computing it then reports.
    """
    from qccompute.program import find_program, plan_of

    try:
        child = ProgramInput.from_spec(spec, inp.structure, structures=inp.structures)
        program = find_program(child.program)
        if program is None:
            return 1, 0
        resources = plan_of(program, child).resources
    except Exception:  # noqa: BLE001 - computing the child reports any problem
        return 1, 0
    threads = child.keywords.get("threads")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        threads = resources.threads
    return threads, resources.gpus


def make_engine(
    geometric: ModuleType,
    molecule: Molecule,
    structure: Structure,
    structures: dict[str, Structure | list[Structure]],
    gradient_spec: ProgramSpec,
    compute_child: ComputeChild,
    trajectory: list[Frame],
) -> NativeEngine:
    """Bind geomeTRIC's native engine interface to independent child calculations."""
    base_engine: Any = geometric.engine.Engine
    # The last step's full output, to restart the next child from its wavefunction.
    last_step: list[ProgramOutput[ProgramInput, Data]] = []

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
            output = compute_child(child, previous=last_step[-1] if last_step else None)
            if not displacement or not output.success:
                trajectory.append(Frame.from_output(output))
                last_step[:] = [output]
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

        def copy_scratch(self, src: str, dest: str) -> None:
            """Nothing to copy: each step is a calculation with its own directory.

            geomeTRIC calls this for every displaced gradient of a finite-difference
            Hessian; its default only logs that it is not implemented.
            """

    return Engine(molecule)


# The Hessians geomeTRIC computes in this context's run, with their coordinates.
_hessians: ContextVar[list[tuple[NDArray[np.float64], NDArray[np.float64]]] | None] = (
    ContextVar("geometric_hessians", default=None)
)
_recording_lock = threading.Lock()


def _record_hessians(geometric: ModuleType) -> None:
    """Wrap geomeTRIC's Cartesian Hessian once, to keep each one a run computes.

    The wrapper stays installed and changes nothing outside a run of this codec: it
    records into the list of the run in its own context, so runs in several threads
    keep their Hessians apart.
    """
    with _recording_lock:
        compute = geometric.optimize.calc_cartesian_hessian
        if getattr(compute, "_qccodec_recording", False):
            return

        def recording_hessian(
            coords: NDArray[np.float64], *args: Any, **kwargs: Any
        ) -> NDArray[np.float64]:
            """Compute the Hessian as geomeTRIC would and remember it."""
            hessian = compute(coords, *args, **kwargs)
            recorded = _hessians.get()
            if recorded is not None:
                recorded.append(
                    (np.array(coords, copy=True), np.array(hessian, copy=True))
                )
            return hessian

        recording_hessian._qccodec_recording = True  # type: ignore[attr-defined]
        geometric.optimize.calc_cartesian_hessian = recording_hessian


def invoke_geometric(
    *,
    structure: Structure,
    structures: dict[str, Structure | list[Structure]],
    gradient_spec: ProgramSpec,
    coordsys: str,
    constraints: dict[str, Any] | None,
    keywords: dict[str, Any],
    geometry_file: str,
    compute_child: ComputeChild,
    directory: Path,
) -> GeometricOutcome:
    """Run the optimizer, retaining its trajectory even when a child or optimizer fails.

    Every file geomeTRIC reads or writes is in ``directory``, so this changes no
    process-wide state and may run in the caller's process, in several threads.
    """
    import geometric

    trajectory: list[Frame] = []
    hessians: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    outcome: GeometricOutcome = {
        "trajectory": trajectory,
        "path": [],
        "ts_index": None,
        "version": geometric.__version__,
        "error": None,
        "traceback": None,
    }
    optimizer = None
    _record_hessians(geometric)
    token = _hessians.set(hessians)
    try:
        molecule = geometric.molecule.Molecule(fnm=str(directory / geometry_file))
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
        optimizer = geometric.optimize.Optimizer(
            structure.geometry.flatten(),
            molecule,
            coords,
            engine=engine,
            dirname=str(directory),  # geomeTRIC edits it as text between IRC branches
            params=geometric.params.OptParams(
                **{"xyzout": str(directory / "qcdata_optim.xyz"), **keywords}
            ),
        )
        optimizer.optimizeGeometry()
    except Exception as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}".strip()
        outcome["traceback"] = traceback.format_exc()
    finally:
        _hessians.reset(token)
        _attach_hessians(trajectory, hessians)
        if keywords.get("irc") and optimizer is not None:
            outcome["path"], outcome["ts_index"] = _irc_path(
                optimizer, trajectory, structure
            )
    return outcome


def _irc_path(
    optimizer: Any, trajectory: list[Frame], structure: Structure
) -> tuple[list[Frame], int | None]:
    """Return the IRC geomeTRIC accepted, as frames, and the transition state's index.

    geomeTRIC's ``progress`` holds the accepted points: after both branches, the
    forward branch reversed, the transition state, then the backward branch. Each point
    is matched to the child calculation at its geometry; the path is returned from the
    backward end to the forward end.
    """
    points = [
        np.asarray(xyz, dtype=float).flatten() * c.ANGSTROM_TO_BOHR
        for xyz in optimizer.progress.xyzs
    ]
    path: list[Frame] = []
    for point in points:
        for frame in reversed(trajectory):
            geometry = frame.structure.geometry.flatten()
            if geometry.shape == point.shape and np.allclose(
                geometry, point, atol=1e-6
            ):
                path.append(frame)
                break
    path.reverse()
    start = structure.geometry.flatten()
    ts_index = next(
        (
            i
            for i, f in enumerate(path)
            if np.allclose(f.structure.geometry.flatten(), start, atol=1e-6)
        ),
        None,
    )
    return align_path(path, ts_index), ts_index


def _attach_hessians(
    trajectory: list[Frame],
    hessians: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> None:
    """Put each Hessian on the trajectory frame computed at the same geometry.

    With ``hessian="first"`` (the default for transition states) that is the first
    frame; ``"last"`` and ``"first+last"`` also give the final frame one.
    """
    for coords, hessian in hessians:
        for i in range(len(trajectory) - 1, -1, -1):
            frame = trajectory[i]
            geometry = frame.structure.geometry.flatten()
            if geometry.shape == coords.shape and np.allclose(geometry, coords):
                results = frame.results.model_copy(update={"hessian": hessian})
                trajectory[i] = frame.model_copy(update={"results": results})
                break


def decode(inp: ProgramInput, out: NativeOutput) -> OptimizationData | PathData:
    """Wrap the recorded trajectory; raise with it attached if the optimizer failed."""
    assert isinstance(out, PythonOutput)
    native = out.value or {
        "trajectory": [],
        "path": [],
        "ts_index": None,
        "version": None,
        "error": None,
        "traceback": None,
    }
    provenance = Provenance(program="geometric", program_version=native["version"])
    data: OptimizationData | PathData
    if inp.calctype == CalcType.irc:
        data = PathData(
            provenance=provenance,
            frames=native.get("path", []),
            ts_index=native.get("ts_index"),
        )
    else:
        data = OptimizationData(provenance=provenance, frames=native["trajectory"])
    if native["error"]:
        error = CalculationError(native["error"], data=data)
        if native.get("traceback"):
            error.add_note(f"geomeTRIC's traceback:\n{native['traceback']}")
        raise error
    return data


codec = Codec(
    "geometric",
    calctypes={CalcType.optimization, CalcType.transition_state, CalcType.irc},
    encode=encode,
    decode=decode,
    # geomeTRIC's exceptions, as the decoder reports them ("<name>: <message>").
    errors=[
        (r"^GeomOptNotConvergedError", "convergence", "max_iter"),
        (
            r"^IRCError: (No imaginary mode|There are more than one imaginary)",
            "input",
            "not_first_order_saddle",
        ),
        (r"^(InputError|ParamError|LinearTorsionError|CheckCoordError)", "input", None),
        (r"Hessian command line argument can only be", "input", None),
        (
            r"No Hessian data file found|hess_data passed in via OptParams",
            "input",
            None,
        ),
        (r"Constraints only work with|Cannot continue a constrained", "input", None),
        (r"(Gradient|Hessian) contains nan", "numerical", None),
    ],
)

#: The program qccompute runs (the ``qccompute.programs`` entry point 'geometric').
#: It runs in the caller's process, so its subprograms are found there too, and plans
#: the threads its subprogram does (None when that one asks for all).
program = CodecProgram(codec, library="geometric", threads=None)
