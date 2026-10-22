"""Helpers shared by codecs that call a Python library in the calling process.

These codecs return a ``PythonInput`` whose function imports the library, runs one
single point, and returns a plain snapshot (:class:`SinglePoint`) in atomic units; the
codec's decoder turns the snapshot into ``SinglePointData``. Libraries are imported
inside the function so encoding and decoding work, and are testable, without them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypedDict, cast

from qcdata import CalcType, ProgramInput, Provenance, SinglePointData, Structure
from qcref import constants as c

from ..exceptions import CalculationError, EncoderError
from ..model_cache import cached_model as cached_model
from ..native import NativeOutput, PythonOutput


class SinglePoint(TypedDict, total=False):
    """What an in-process single point returns, in atomic units.

    ``energy`` and ``version`` are always present; the rest only when computed.
    """

    energy: float
    gradient: list[list[float]]
    hessian: list[list[float]]
    atomic_charges: list[float]
    scf_dipole_moment: list[float]
    extras: dict[str, Any]
    version: str | None


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


def package_version(distribution: str) -> str | None:
    """Return an installed distribution's version, or None if it is not installed."""
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


def resolve_device(device: str | None) -> str:
    """Return ``device``, or CUDA when PyTorch sees a GPU and CPU otherwise."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


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


def reject_unknown(
    program: str, keywords: Mapping[str, Any], known: set[str] | frozenset[str]
) -> None:
    """Reject keywords the codec does not know, so typos fail instead of being ignored.

    Raises:
        EncoderError: Some keyword names are not in ``known``.
    """
    unknown = sorted(set(keywords) - known)
    if unknown:
        raise EncoderError(
            f"Unknown {program} keywords {unknown}; supported: {sorted(known)}."
        )


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
            "scf_dipole_moment": snapshot.get("scf_dipole_moment"),
            "extras": snapshot.get("extras", {}),
        }
        if inp.calctype in (CalcType.gradient, CalcType.hessian):
            fields["gradient"] = snapshot.get("gradient")
        if inp.calctype == CalcType.hessian:
            fields["hessian"] = snapshot.get("hessian")
        return SinglePointData(provenance=provenance, **fields)

    return decode
