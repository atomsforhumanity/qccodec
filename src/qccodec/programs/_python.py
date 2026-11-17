"""Helpers shared by codecs that call a Python library in the calling process.

These codecs return a ``PythonInput`` whose function imports the library, runs one
single point, and returns a plain snapshot (:class:`SinglePoint`) in atomic units; the
codec's decoder turns the snapshot into ``SinglePointData``. Libraries are imported
inside the function so encoding and decoding work, and are testable, without them.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

from qcdata import CalcType, ProgramInput, Provenance, SinglePointData, Structure
from qcref import constants as c

from ..exceptions import CalculationError, EncoderError
from ..model_cache import cached_model as cached_model
from ..native import NativeOutput, PythonOutput


class SinglePoint(TypedDict, total=False):
    """What an in-process single point returns, in atomic units.

    ``energy`` and ``version`` are always present; the rest only when computed.
    ``device`` is where a GPU-capable library ran (``cuda``, ``cpu``): the adapter
    warns ``backend_fallback`` when a calculation granted a GPU ran on the CPU.
    """

    energy: float
    gradient: list[list[float]]
    hessian: list[list[float]]
    atomic_charges: list[float]
    dipole_moment: list[float]
    extras: dict[str, Any]
    version: str | None
    device: str


def positions_angstrom(structure: Structure) -> list[list[float]]:
    """Return the structure's coordinates in angstrom, the unit most libraries take."""
    return (structure.geometry * c.BOHR_TO_ANGSTROM).tolist()


def from_ev(
    energy: float,
    forces: Any = None,
    hessian: Any = None,
) -> SinglePoint:
    """Convert eV-based results to Hartree, Hartree/bohr, and Hartree/bohr^2.

    Args:
        energy: Energy in eV.
        forces: Forces in eV/angstrom, shape (n_atoms, 3); the gradient is their
            negative.
        hessian: Second derivatives of the energy in eV/angstrom^2, any shape with
            (3 n_atoms)^2 elements ordered atom-major (x1, y1, z1, x2, ...).

    Returns:
        A snapshot without ``version``.
    """
    import numpy as np

    result: SinglePoint = {"energy": float(energy) * c.EV_TO_HARTREE}
    if forces is not None:
        scale = c.EV_TO_HARTREE / c.ANGSTROM_TO_BOHR
        result["gradient"] = (-np.asarray(forces, dtype=float) * scale).tolist()
    if hessian is not None:
        values = np.asarray(hessian, dtype=float)
        n = int(round(values.size**0.5))
        scale = c.EV_TO_HARTREE / c.ANGSTROM_TO_BOHR**2
        result["hessian"] = (values.reshape(n, n) * scale).tolist()
    return result


@functools.cache
def package_version(distribution: str) -> str | None:
    """Return an installed distribution's version, or None if it is not installed.

    Read once per process: every output names it, and a distribution does not change
    under a running process without a restart to load it.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def pop_device(keywords: dict[str, Any]) -> str | None:
    """Remove and return the ``device`` keyword (``"cpu"``, ``"cuda"``, ``"cuda:1"``).

    Raises:
        EncoderError: ``device`` is not a string.
    """
    device = keywords.pop("device", None)
    if device is not None and not isinstance(device, str):
        raise EncoderError(f"device must be a string such as 'cuda', not {device!r}.")
    return device


def gpu_count(device: str | None) -> int:
    """Return the GPUs a calculation on ``device`` declares for scheduling.

    ``"cpu"`` uses none. ``"cuda"`` or no device (a GPU when one is present) uses
    one, which a runner may pin with ``CUDA_VISIBLE_DEVICES``. An explicit index such
    as ``"cuda:1"`` declares none: pinning would renumber the devices, so callers
    who choose devices themselves also manage sharing them.
    """
    if device is None or device == "cuda":
        return 1
    return 0


#: The keyword into which the adapter hands a library the device qccompute granted.
GRANTED = "granted"


def resolve_device(device: str | None, granted: str | None = None) -> str:
    """Return the device a library runs on.

    Args:
        device: The ``device`` keyword (``"cpu"``, ``"cuda"``, ``"cuda:1"``), or None
            to leave it to the codec.
        granted: The device qccompute granted the calculation (``cuda:N``, its GPU
            as this process's CUDA device, or ``cpu`` when it granted none), injected
            through ``PythonInput.device_keyword``; None when called directly.

    Returns:
        ``device`` as given, except that ``cuda`` runs on the granted GPU. Left to
        the codec, the granted device, or the CPU when PyTorch cannot use the
        granted GPU (the adapter warns ``backend_fallback``); called directly, CUDA
        when PyTorch sees a GPU and the CPU otherwise.
    """
    if device == "cuda" and granted is not None and granted != "cpu":
        return granted
    if device:
        return device
    if granted == "cpu":
        return granted
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return granted or "cuda"


def model_file(spec: Any) -> Path | None:
    """The model file a spec's ``model.method`` names, when it is a path to one.

    For the numerics identity of ML potentials that load a model from a path
    (`qccodec.CodecProgram` ``weights``): the file's digest tells a retrained
    checkpoint from the one before it.

    Args:
        spec: A `qcdata.ProgramInput` or `qcdata.ProgramSpec`.

    Returns:
        The file's absolute path, or None when the method names no existing file.
    """
    model = getattr(spec, "model", None)
    method = getattr(model, "method", None)
    if not method:
        return None
    path = Path(method).expanduser()
    return path.resolve() if path.is_file() else None


def require_model(inp: ProgramInput, example: str) -> str:
    """Return ``model.method``, rejecting a model with a basis set.

    Raises:
        EncoderError: The model is missing or has a basis set.
    """
    if inp.model is None:
        raise EncoderError(
            f"{inp.program} requires a model, e.g. {{'method': '{example}'}}."
        )
    if inp.model.basis:
        raise EncoderError(
            f"{inp.program} models have no basis set; remove basis {inp.model.basis!r}."
        )
    return inp.model.method


def single_point_decoder(
    program: str,
) -> Callable[[ProgramInput, NativeOutput], SinglePointData]:
    """Build the decoder that turns a :class:`SinglePoint` snapshot into data.

    The calculation type decides which fields are kept: an energy never carries a
    gradient a library happened to compute.
    """

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        assert isinstance(out, PythonOutput)
        snapshot = cast(SinglePoint, out.value or {})
        provenance = Provenance(
            program=program, program_version=snapshot.get("version")
        )
        if out.error is not None or not snapshot:
            reason = (
                f"{type(out.error).__name__}: {out.error}"
                if out.error is not None
                else "no results were returned"
            )
            raise CalculationError(
                f"{program} failed: {reason}",
                data=SinglePointData(provenance=provenance),
            )
        fields: dict[str, Any] = {
            "energy": snapshot["energy"],
            "atomic_charges": snapshot.get("atomic_charges"),
            "dipole_moment": snapshot.get("dipole_moment"),
            "extras": snapshot.get("extras", {}),
        }
        if inp.calctype in (CalcType.gradient, CalcType.hessian):
            fields["gradient"] = snapshot.get("gradient")
        if inp.calctype == CalcType.hessian:
            fields["hessian"] = snapshot.get("hessian")
        return SinglePointData(provenance=provenance, **fields)

    return decode


def batch_signature(inp: ProgramInput) -> Hashable | None:
    """Return the batch key of an input: everything a batch shares, all but the structure.

    Hessians return None: libraries differentiate one structure at a time.
    """
    if inp.calctype == CalcType.hessian:
        return None
    return json.dumps(
        {
            "program": inp.program,
            "calctype": inp.calctype.value,
            "model": inp.model.model_dump(mode="json") if inp.model else None,
            "keywords": inp.keywords,
        },
        sort_keys=True,
        default=str,
    )


def split_atoms(values: Any, counts: Sequence[int]) -> list[Any]:
    """Split per-atom rows of a batch (all molecules' atoms in order) per molecule."""
    import numpy as np

    array = np.asarray(values)
    bounds = np.cumsum([0, *counts])
    return [array[bounds[i] : bounds[i + 1]] for i in range(len(counts))]
