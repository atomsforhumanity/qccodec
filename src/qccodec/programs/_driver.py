"""Helpers shared by driver codecs: programs whose steps are other programs' calculations.

A driver (geomeTRIC, Sella) runs in Python and asks for energies and gradients. Each
request is a child calculation of a program named in ``subprograms``, computed by the
runner through the ``Compute`` it injects (``PythonInput.compute_callback``), so
children run like any other calculation: through their codecs, batched or side by side
when computed together, with failures kept. What every driver does the same way lives
here:

- encoding: picking subprograms by calculation type, refusing keywords the codec sets
  or the program would ignore, and declaring the resources of the children;
- running: a ``Recorder`` that computes the children, starts each from the previous
  wavefunction, and keeps the frames the result reports, including a failed child;
- decoding: one ``Outcome`` shape, which ``driver_decoder`` turns into
  ``OptimizationData``, ``PathData``, or ``PathOptimizationData``;
- ``align_path``, which removes an IRC path's rigid drift.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from qcdata import (
    CalcType,
    Data,
    Frame,
    OptimizationData,
    PathData,
    PathOptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
)
from qcref import periodic_table

from ..exceptions import CalculationError, EncoderError
from ..native import Compute, ComputeChild, NativeOutput, PythonOutput, sequential
from ._python import reject_unknown

Output = ProgramOutput[ProgramInput, Data]

# Keyword names of the callbacks a driver's function receives. Drivers name both in
# their PythonInput: runners before qccompute 0.23 inject only the one-input callback.
COMPUTE = "compute"
COMPUTE_CHILD = "compute_child"


# ---------------------------------------------------------------- encoding


def subprograms(
    inp: ProgramInput,
    required: CalcType,
    optional: Collection[CalcType] = (),
) -> dict[CalcType, ProgramSpec]:
    """Return the driver's subprograms by calculation type (qcdata keeps them unique).

    Raises:
        EncoderError: The required subprogram is missing, or one has a calculation
            type the driver does not use.
    """
    found = {spec.calctype: spec for spec in inp.subprograms}
    if required in found and set(found) <= {required, *optional}:
        return found
    wanted = f"one {required.value} subprogram"
    if optional:
        wanted += " and optionally one " + " or ".join(c.value for c in optional)
    given = [s.calctype.value for s in inp.subprograms]
    raise EncoderError(f"{inp.program} takes {wanted}, not {given}.")


def check_keywords(
    program: str,
    keywords: Mapping[str, Any],
    *,
    owned: Collection[str] = (),
    known: Collection[str] | None = None,
) -> None:
    """Refuse keywords the codec sets itself and, given ``known``, any other name.

    ``known`` is for programs that ignore names they do not read, so a typo would
    silently run with the default; programs that reject them need no list.

    Raises:
        EncoderError: A keyword is owned by the codec or unknown.
    """
    taken = sorted(set(owned) & set(keywords))
    if taken:
        raise EncoderError(f"The {program} codec sets these keywords itself: {taken}.")
    if known is not None:
        reject_unknown(program, keywords, frozenset(known))


def child_resources(inp: ProgramInput, specs: Iterable[ProgramSpec]) -> tuple[int, int]:
    """Return the most threads and GPUs any child declares, for the driver to declare.

    Each child is encoded at the input's structure, or the first of its chain. One that
    cannot be encoded declares nothing: computing it reports the problem.
    """
    from ..registry import get_codec

    threads, gpus = 1, 0
    for spec in specs:
        try:
            structure = inp.structure or inp.initial_chain[0]
            others = {k: v for k, v in inp.structures.items() if k != "initial_chain"}
            child = ProgramInput.from_spec(spec, structure, structures=others)
            native = get_codec(child).encode(child)
        except Exception:  # noqa: BLE001 - computing the child reports any problem
            continue
        threads, gpus = max(threads, native.threads), max(gpus, native.gpus)
    return threads, gpus


# ----------------------------------------------------------------- running


class Recorder:
    """Computes a driver's children and keeps the frames its result reports.

    Each child starts from the wavefunction of the last successful child of the same
    program, which the runner applies when it propagates wavefunctions; images of a
    chain each start from the same image's last output. A failed child always becomes
    a frame, since it explains why the driver stopped.

    Attributes:
        frames: The frames recorded so far. A driver may point it at a new list, such
            as one branch of an IRC.
        outputs: Every child computed one at a time, in order.
    """

    def __init__(
        self,
        program: str,
        compute: Compute | None,
        compute_child: ComputeChild | None = None,
    ) -> None:
        """Bind the runner's callback.

        Args:
            program: The driver's name, for errors.
            compute: The runner's ``Compute``.
            compute_child: The one-input callback of runners before qccompute 0.23,
                used when ``compute`` is missing; children of a list then run one at
                a time.

        Raises:
            RuntimeError: The runner provided neither callback.
        """
        if compute is None:
            if compute_child is None:
                raise RuntimeError(
                    f"{program} computes child calculations, but its runner provided "
                    "no compute callback (PythonInput.compute_callback)."
                )
            compute = sequential(compute_child)
        self._compute = compute
        self.frames: list[Frame] = []
        self.outputs: list[Output] = []
        self._previous: dict[str, Output] = {}
        self._images: list[Output | None] = []
        self._hessians: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    def compute(self, child: ProgramInput, *, step: bool = True) -> Output:
        """Compute one child; a step, or any failure, becomes the next frame.

        Args:
            child: The calculation.
            step: Whether the child is a step of the driver. Probes, such as the
                displaced gradients of a finite-difference Hessian, are not.
        """
        output = self._compute(child, previous=self._previous.get(child.program))
        self.outputs.append(output)
        if output.success:
            self._previous[child.program] = output
        if step or not output.success:
            self.record(output)
        return output

    def compute_images(self, children: Sequence[ProgramInput]) -> list[Output]:
        """Compute a chain's images together, each from its own last output.

        No frames are recorded: a chain's frames belong to the path of its iteration.
        """
        starts = self._images
        if len(starts) != len(children):
            starts = [None] * len(children)
        outputs = self._compute(list(children), previous=starts)
        self._images = [
            new if new.success else old
            for new, old in zip(outputs, starts, strict=True)
        ]
        return outputs

    def record(self, output: Output) -> None:
        """Add a computed output to the frames."""
        self.frames.append(Frame.from_output(output))

    def record_at(self, geometry: NDArray[np.float64]) -> None:
        """Add the latest child computed at ``geometry`` (bohr) to the frames.

        For drivers that tell steps from probes only after computing them, such as an
        optimizer that reports each accepted geometry.
        """
        here = np.asarray(geometry, dtype=float).flatten()
        for output in reversed(self.outputs):
            computed = output.input_data.require_structure().geometry.flatten()
            if computed.shape == here.shape and np.allclose(computed, here, atol=1e-8):
                self.record(output)
                return

    def add_hessian(
        self, geometry: NDArray[np.float64], hessian: NDArray[np.float64]
    ) -> None:
        """Remember a Hessian (Hartree/bohr^2) the driver computed at ``geometry``."""
        self._hessians.append(
            (np.array(geometry, dtype=float).flatten(), np.array(hessian, dtype=float))
        )

    def attach_hessians(self, frames: list[Frame]) -> None:
        """Put each Hessian on the last of ``frames`` computed at the same geometry.

        Frames are replaced in place; a Hessian at no frame's geometry is dropped.
        """
        for coords, hessian in self._hessians:
            for i in range(len(frames) - 1, -1, -1):
                geometry = frames[i].structure.geometry.flatten()
                if geometry.shape == coords.shape and np.allclose(geometry, coords):
                    results = frames[i].results.model_copy(update={"hessian": hessian})
                    frames[i] = frames[i].model_copy(update={"results": results})
                    break


# ---------------------------------------------------------------- decoding


class Outcome(TypedDict):
    """What a driver's function returns; ``driver_decoder`` turns it into data.

    Attributes:
        frames: An optimization's trajectory, or an IRC's path.
        ts_index: The transition state's frame in an IRC path.
        paths: A path optimization's chain at each iteration, with its highest image.
        version: The driver's version.
        error: Why the driver stopped without converging, or None.
    """

    frames: list[Frame]
    ts_index: int | None
    paths: list[tuple[list[Frame], int | None]]
    version: str | None
    error: str | None


def outcome(version: str | None) -> Outcome:
    """Return an empty outcome for a driver at ``version``."""
    return {
        "frames": [],
        "ts_index": None,
        "paths": [],
        "version": version,
        "error": None,
    }


def driver_decoder(
    program: str,
) -> Callable[
    [ProgramInput, NativeOutput], OptimizationData | PathData | PathOptimizationData
]:
    """Build the decoder that wraps a driver's ``Outcome`` in its calculation's data.

    A run that stopped with an error, or whose function raised, raises
    ``CalculationError`` with the data recorded so far attached.
    """

    def decode(
        inp: ProgramInput, out: NativeOutput
    ) -> OptimizationData | PathData | PathOptimizationData:
        assert isinstance(out, PythonOutput)
        native: Outcome = out.value or outcome(None)
        provenance = Provenance(program=program, program_version=native["version"])
        data: OptimizationData | PathData | PathOptimizationData
        if inp.calctype == CalcType.path_optimization:
            data = PathOptimizationData(
                provenance=provenance,
                paths=[
                    PathData(provenance=provenance, frames=frames, ts_index=ts_index)
                    for frames, ts_index in native["paths"]
                ],
            )
        elif inp.calctype == CalcType.irc:
            data = PathData(
                provenance=provenance,
                frames=native["frames"],
                ts_index=native["ts_index"],
            )
        else:
            data = OptimizationData(provenance=provenance, frames=native["frames"])
        error = native["error"]
        if error is None and out.error is not None:
            error = f"{type(out.error).__name__}: {out.error}"
        if error:
            raise CalculationError(error, data=data)
        return data

    return decode


# ------------------------------------------------------------------- paths


def align_path(frames: list[Frame], ts_index: int | None) -> list[Frame]:
    """Remove a path's rigid drift: frames aligned outward from the transition state.

    IRC programs step in Cartesian or internal coordinates without always removing
    overall translation and rotation, so a branch can drift and, at its minimum, spin.
    Working outward from ``ts_index`` (or the first frame), each frame is moved onto its
    already-aligned neighbor by the mass-weighted best fit: centers of mass together,
    then the rotation minimizing the mass-weighted displacement, as an IRC step has no
    rigid part. Gradients and Hessians are rotated with their frames, so the path stays
    in the transition state's frame, which is the input's.
    """
    if not frames:
        return frames
    start = ts_index if ts_index is not None else 0
    backward = _align_branch(frames[start::-1])
    forward = _align_branch(frames[start:])
    return backward[:0:-1] + forward


def _align_branch(branch: list[Frame]) -> list[Frame]:
    """Align each frame of a branch onto the one before it (the first stays put)."""
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
        update: dict[str, Any] = {}
        if frame.results.gradient is not None:
            update["gradient"] = np.asarray(frame.results.gradient) @ rotation
        if frame.results.hessian is not None:
            block = np.kron(np.eye(len(masses)), rotation)
            update["hessian"] = block.T @ np.asarray(frame.results.hessian) @ block
        structure = frame.structure.model_copy(update={"geometry": previous})
        results = frame.results.model_copy(update=update)
        aligned.append(
            Frame.model_validate(
                {**dict(frame), "structure": structure, "results": results}
            )
        )
    return aligned
