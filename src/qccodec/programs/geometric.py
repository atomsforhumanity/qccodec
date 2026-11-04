"""geomeTRIC: optimizations, transition states, IRCs, and NEB driving another program.

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
  distances in bohr; scans are not supported). geomeTRIC ignores names it does not
  read, so the codec refuses them rather than let a typo run with a default.
  ``transition`` and ``irc`` follow from the calculation type and are refused, as are
  ``bigchem`` (the runner computes children) and ``meci``.
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

``path_optimization`` runs geomeTRIC's nudged elastic band over
``structures["initial_chain"]`` (endpoints included, in order; geomeTRIC uses
``images`` of them, 11 by default, picked evenly, and does not interpolate). Its
``keywords`` are ``geometric.params.NEBParams`` (``images``, ``maxg``, ``avgg``,
``nebk``, ``neb_maxcyc``, ``climb``, ``optep``, ``align``, ``trust``, ...); other names
and ``bigchem`` are refused. Each iteration's images are computed together: all go to
the runner's ``compute`` at once, which batches machine-learned potentials and runs
programs side by side (one at a time with runners before qccompute 0.23). The result is a
``PathOptimizationData`` with the chain at every iteration; each path's ``ts_index``
is the climbing image, or the highest interior image before climbing starts. A band
that has not met ``maxg`` and ``avgg`` within ``neb_maxcyc`` fails, with its paths.

The trajectory holds one frame per optimizer step. Gradients at the displaced
geometries of a finite-difference Hessian are not steps, so they are left out, unless
one fails. Each Hessian geomeTRIC computes is attached to the frame at its geometry
(``frame.results.hessian``): the first frame for ``hessian="first"``, the default for
transition states, and also the last frame for ``"first+last"`` or ``"last"``.
Atom indices in ``constraints`` are 0-based, as in geomeTRIC's Python interface.

With qccompute, ``propagate_wfn=True`` starts each child calculation from the orbitals
of the last successful one when the child codec supports it (TeraChem); each NEB image
starts from its own orbitals of the previous iteration. The optimization declares the
threads and GPUs its child does, so lists of optimizations share the machine correctly.
"""

from __future__ import annotations

import traceback
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qcdata import (
    CalcType,
    Frame,
    ProgramInput,
    ProgramSpec,
    SinglePointData,
    Structure,
)
from qcref import constants as c

from ..codec import Codec
from ..exceptions import EncoderError
from ..native import Compute, ComputeChild, PythonInput
from ._driver import (
    COMPUTE,
    COMPUTE_CHILD,
    Outcome,
    Recorder,
    align_path,
    check_keywords,
    child_resources,
    driver_decoder,
    outcome,
    subprograms,
)

if TYPE_CHECKING:
    from geometric.engine import Engine as NativeEngine
    from geometric.molecule import Molecule


# Names geometric.params.OptParams and NEBParams read (geomeTRIC 1.1); both silently
# ignore any other. tests/test_native_codecs.py checks them against the installed
# geomeTRIC. bigchem and meci are left out: the runner computes the children, and
# the codec's engine has no second state for a conical intersection.
OPT_KEYWORDS = frozenset(
    {
        "bothre",
        "check",
        "converge",
        "convergence_cmax",
        "convergence_dmax",
        "convergence_drms",
        "convergence_energy",
        "convergence_gmax",
        "convergence_grms",
        "convergence_molpro_dmax",
        "convergence_molpro_gmax",
        "convergence_set",
        "enforce",
        "epsilon",
        "frequency",
        "hess_data",
        "hessian",
        "ignore_modes",
        "irc_direction",
        "maxiter",
        "molcnv",
        "qccnv",
        "qdata",
        "reset",
        "subfrctor",
        "thermo",
        "tmax",
        "tmin",
        "trust",
        "usedmax",
        "verbose",
        "wigner",
        "write_cart_hess",
        "xyzout",
    }
)
NEB_KEYWORDS = frozenset(
    {
        "align",
        "avgg",
        "climb",
        "epsilon",
        "guessk",
        "guessw",
        "images",
        "maxg",
        "ncimg",
        "neb_history",
        "neb_maxcyc",
        "nebk",
        "optep",
        "plain",
        "prefix",
        "skip",
        "tmax",
        "tmin",
        "trust",
        "verbose",
        "xyzout",
    }
)
# Keywords of the codec itself, and ones it sets from the calculation type.
CODEC_KEYWORDS = frozenset({"coordsys", "constraints"})
OWNED_KEYWORDS = frozenset({"transition", "irc"})

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
    gradient_spec = subprograms(inp, CalcType.gradient)[CalcType.gradient]
    if inp.cmdline_args:
        raise EncoderError(
            "geomeTRIC's Python interface does not accept command-line arguments."
        )
    if "bigchem" in inp.keywords:
        raise EncoderError(
            "The geometric codec computes child calculations through the runner "
            "(batched or in parallel); bigchem is not accepted."
        )
    if inp.calctype == CalcType.path_optimization:
        return _encode_neb(inp, gradient_spec)
    check_keywords(
        "geometric",
        inp.keywords,
        owned=OWNED_KEYWORDS,
        known=OPT_KEYWORDS | CODEC_KEYWORDS,
    )
    keywords = deepcopy(inp.keywords)
    keywords["transition"] = inp.calctype == CalcType.transition_state
    keywords["irc"] = inp.calctype == CalcType.irc
    coordsys = keywords.pop("coordsys", "tric")
    if coordsys not in COORDINATE_SYSTEMS:
        raise EncoderError(f"Unsupported coordinate system: {coordsys}")
    constraints = keywords.pop("constraints", None)
    if constraints and "scan" in constraints:
        raise EncoderError(
            "Constraint scans are not supported by this geomeTRIC integration."
        )
    threads, gpus = child_resources(inp, [gradient_spec])
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
        compute_callback=COMPUTE,
        child_callback=COMPUTE_CHILD,
        loggers=("geometric",),
        # Each step runs the child, so the optimization occupies what it declares.
        threads=threads,
        gpus=gpus,
    )


def _encode_neb(inp: ProgramInput, gradient_spec: ProgramSpec) -> PythonInput:
    """Prepare geomeTRIC's nudged elastic band over ``structures["initial_chain"]``."""
    check_keywords("geometric", inp.keywords, known=NEB_KEYWORDS)
    keywords = deepcopy(inp.keywords)
    chain = [s.model_copy(deep=True) for s in inp.initial_chain]
    others = {k: v for k, v in inp.structures.items() if k != "initial_chain"}
    threads, gpus = child_resources(inp, [gradient_spec])
    return PythonInput(
        function=invoke_geometric_neb,
        kwargs={
            "chain": chain,
            "structures": deepcopy(others),
            "gradient_spec": gradient_spec.model_copy(deep=True),
            "keywords": keywords,
        },
        files=dict(inp.files),
        compute_callback=COMPUTE,
        child_callback=COMPUTE_CHILD,
        loggers=("geometric",),
        threads=threads,
        gpus=gpus,
    )


def make_engine(
    geometric: ModuleType,
    molecule: Molecule,
    structure: Structure,
    structures: dict[str, Structure | list[Structure]],
    gradient_spec: ProgramSpec,
    recorder: Recorder,
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
            output = recorder.compute(child, step=not displacement)
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
    structures: dict[str, Structure | list[Structure]],
    gradient_spec: ProgramSpec,
    coordsys: str,
    constraints: dict[str, Any] | None,
    keywords: dict[str, Any],
    geometry_file: str,
    compute: Compute | None = None,
    compute_child: ComputeChild | None = None,
) -> Outcome:
    """Run the optimizer, retaining its trajectory even when a child or optimizer fails."""
    import geometric

    recorder = Recorder("geometric", compute, compute_child)
    compute_hessian = None
    result = outcome(geometric.__version__)
    optimizer = None
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
            recorder,
        )
        # Record every Cartesian Hessian geomeTRIC computes, with its geometry.
        compute_hessian = geometric.optimize.calc_cartesian_hessian

        def recording_hessian(
            coords: NDArray[np.float64], *args: Any, **kwargs: Any
        ) -> NDArray[np.float64]:
            """Compute the Hessian as geomeTRIC would and remember it."""
            hessian = compute_hessian(coords, *args, **kwargs)
            recorder.add_hessian(coords, hessian)
            return hessian

        geometric.optimize.calc_cartesian_hessian = recording_hessian
        optimizer = geometric.optimize.Optimizer(
            structure.geometry.flatten(),
            molecule,
            coords,
            engine=engine,
            dirname=str(Path.cwd()),  # geomeTRIC edits it as text between IRC branches
            params=geometric.params.OptParams(
                **{"xyzout": "qcdata_optim.xyz", **keywords}
            ),
        )
        optimizer.optimizeGeometry()
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if compute_hessian is not None:
            geometric.optimize.calc_cartesian_hessian = compute_hessian
        recorder.attach_hessians(recorder.frames)
        result["frames"] = recorder.frames
        if keywords.get("irc") and optimizer is not None:
            result["frames"], result["ts_index"] = _irc_path(
                optimizer, recorder.frames, structure
            )
    return result


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


def invoke_geometric_neb(
    *,
    chain: list[Structure],
    structures: dict[str, Structure | list[Structure]],
    gradient_spec: ProgramSpec,
    keywords: dict[str, Any],
    compute: Compute | None = None,
    compute_child: ComputeChild | None = None,
) -> Outcome:
    """Run geomeTRIC's NEB, computing each iteration's images together.

    geomeTRIC evaluates every image in ``Chain.ComputeEnergyGradient`` (its "parallel
    point"); that method is replaced for the run so all images go to the runner's
    ``compute`` at once, which batches them or runs them side by side. The path is
    recorded at each iteration geomeTRIC saves.
    """
    import geometric
    from geometric.neb import Chain, ElasticBand, OptimizeChain

    recorder = Recorder("geometric", compute, compute_child)
    report = outcome(geometric.__version__)
    template = chain[0]
    images: dict[int, list[Frame]] = {}  # id(chain object) -> its images' frames

    def evaluate(band: Any, cyc: Any = None, result: Any = None) -> None:
        """Compute the energy and gradient of every image of a chain at once."""
        if result is not None:
            original(band, cyc=None, result=result)
            return
        children = [
            ProgramInput.from_spec(
                gradient_spec,
                template.model_copy(
                    update={"geometry": np.asarray(s.cartesians).reshape(-1, 3)}
                ),
                structures=structures,
            )
            for s in band.Structures
        ]
        outputs = recorder.compute_images(children)
        frames = [Frame.from_output(output) for output in outputs]
        images[id(band)] = frames
        failed = [i for i, output in enumerate(outputs) if not output.success]
        if failed:
            report["paths"].append((frames, None))
            raise RuntimeError(f"Child calculations failed for images {failed}.")
        results = []
        for frame in frames:
            data = frame.results
            if data.energy is None or data.gradient is None:
                raise RuntimeError("geomeTRIC's NEB needs an energy and a gradient.")
            results.append(
                {"energy": data.energy, "gradient": np.asarray(data.gradient).ravel()}
            )
        original(band, cyc=None, result=results)

    def record(band: Any) -> None:
        """Keep the path of an iteration geomeTRIC saves."""
        frames = images.get(id(band))
        if frames is None:
            return
        report["paths"].append((frames, _highest_image(band, frames)))

    original = Chain.ComputeEnergyGradient
    Chain.ComputeEnergyGradient = evaluate  # type: ignore[method-assign]
    try:
        params = geometric.params.NEBParams(**keywords)
        molecule = geometric.molecule.Molecule()
        molecule.elem = list(template.symbols)
        molecule.charge = template.charge
        molecule.mult = template.multiplicity
        picks = np.linspace(0, len(chain) - 1, min(params.images, len(chain)))
        molecule.xyzs = [
            np.asarray(chain[int(round(i))].geometry) * c.BOHR_TO_ANGSTROM
            for i in picks
        ]
        molecule.build_topology()
        engine = geometric.engine.Engine(molecule)
        band = ElasticBand(
            molecule,
            engine=engine,
            tmpdir=str(Path.cwd()),
            params=params,
            plain=params.plain,
        )
        final, _ = OptimizeChain(band, engine, params, save_callback=record)
        if not (final.maxg < params.maxg and final.avgg < params.avgg):
            report["error"] = (
                f"geomeTRIC's NEB did not converge in {params.maxcyc} cycles "
                f"(largest force {final.maxg:.3g}, average {final.avgg:.3g} eV/angstrom; "
                f"targets {params.maxg}, {params.avgg})."
            )
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        Chain.ComputeEnergyGradient = original  # type: ignore[method-assign]
    return report


def _highest_image(band: Any, frames: list[Frame]) -> int | None:
    """The climbing image, if geomeTRIC chose one, else the highest interior image."""
    climbers = list(getattr(band, "climbers", []) or [])
    if getattr(band, "climbSet", False) and climbers:
        return int(climbers[0])
    energies = [f.results.energy for f in frames[1:-1]]
    if not energies or any(e is None for e in energies):
        return None
    return 1 + int(np.argmax(np.array(energies, dtype=float)))


codec = Codec(
    "geometric",
    calctypes={
        CalcType.optimization,
        CalcType.transition_state,
        CalcType.irc,
        CalcType.path_optimization,
    },
    encode=encode,
    decode=driver_decoder("geometric"),
)
