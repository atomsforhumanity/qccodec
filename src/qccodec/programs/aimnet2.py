"""AIMNet2: machine-learned potentials from the Isayev group, through ``aimnet``.

The codec calls ``aimnet.calculators.AIMNet2Calculator`` in the calling process; install
it next to qccompute with ``pip install aimnet``. Models download on first use and are
loaded once per process, then reused for every later calculation with the same model
and device.

Inputs map to AIMNet2 as follows:

- ``model.method`` is a model name from aimnet's registry: ``aimnet2`` (ωB97M-D3,
  14 elements, any charge), ``aimnet2-rxn`` (reactions, H C N O), ``aimnet2-nse``
  (open-shell species), ``aimnet2-2025`` (B97-3c), ``aimnet2-b973c``, ``aimnet2-pd``,
  or a member such as ``aimnet2-wb97m-d3_1``; or a path to a model file. Models have
  no basis set.
- ``structure.charge`` goes to the model. ``structure.multiplicity`` goes to the
  model only for spin-aware models (``aimnet2-nse``); the others are closed-shell,
  so the codec raises for a multiplicity other than 1.
- ``keywords``: ``device`` (``"cpu"``, ``"cuda"``, ``"cuda:1"``; CUDA when available
  by default) and ``validate_species`` (``True`` by default: aimnet refuses elements
  or charges a model was not trained on). Nothing else is accepted. A model on a GPU
  declares one for qccompute, which gives each calculation of a list its own; an
  explicit index such as ``"cuda:1"`` opts out of that.
- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy``, ``gradient``, and ``hessian`` (by automatic
differentiation). Every result records the model's partial charges in
``atomic_charges``. AIMNet2 predicts energies in eV; the codec converts them, and the
forces and Hessian, to Hartree and bohr.

AIMNet2 does not move the molecule, so results are in the input's frame. No input
field carries atom indices.
"""

from __future__ import annotations

from typing import Any

from qcdata import CalcType, ProgramInput

from ..codec import Codec
from ..exceptions import EncoderError
from ..native import PythonInput
from ._python import (
    SinglePoint,
    cached_model,
    from_ev,
    gpu_count,
    package_version,
    pop_device,
    positions_angstrom,
    reject_unknown,
    require_model,
    resolve_device,
    single_point_decoder,
)


def encode(inp: ProgramInput) -> PythonInput:
    """Collect the model name, device, and molecule in angstrom.

    Raises:
        EncoderError: The model is missing or has a basis, a keyword is unknown, or
            ``cmdline_args`` or ``files`` were given.
    """
    model = require_model(inp, "aimnet2")
    if inp.cmdline_args or inp.files:
        raise EncoderError("AIMNet2 runs in Python; cmdline_args and files are unused.")
    keywords = dict(inp.keywords)
    device = pop_device(keywords)
    reject_unknown("aimnet2", keywords, frozenset({"validate_species"}))
    return PythonInput(
        gpus=gpu_count(device),
        function=invoke_aimnet2,
        kwargs={
            "model": model,
            "device": device,
            "numbers": inp.structure.atomic_numbers,
            "positions": positions_angstrom(inp.structure),
            "charge": inp.structure.charge,
            "multiplicity": inp.structure.multiplicity,
            "hessian": inp.calctype == CalcType.hessian,
            "forces": inp.calctype != CalcType.energy,
            "validate_species": bool(keywords.get("validate_species", True)),
        },
    )


def invoke_aimnet2(
    *,
    model: str,
    device: str | None,
    numbers: list[int],
    positions: list[list[float]],
    charge: int,
    multiplicity: int,
    forces: bool,
    hessian: bool,
    validate_species: bool,
) -> SinglePoint:
    """Run one AIMNet2 single point with a cached model."""
    from aimnet.calculators import AIMNet2Calculator

    device = resolve_device(device)
    calculator = cached_model(
        ("aimnet2", model, device), lambda: AIMNet2Calculator(model, device=device)
    )
    data: dict[str, Any] = {"coord": positions, "numbers": numbers, "charge": charge}
    if calculator.is_nse:
        data["mult"] = multiplicity
    elif multiplicity != 1:
        raise ValueError(
            f"AIMNet2 model {model!r} is closed-shell; use 'aimnet2-nse' for "
            f"multiplicity {multiplicity}."
        )
    out = calculator.eval(
        data, forces=forces, hessian=hessian, validate_species=validate_species
    )
    result = from_ev(
        out["energy"].item(),
        out["forces"].detach().cpu().numpy() if forces else None,
        out["hessian"].detach().cpu().numpy() if hessian else None,
    )
    result["atomic_charges"] = out["charges"].detach().cpu().numpy().ravel().tolist()
    result["version"] = package_version("aimnet")
    return result


codec = Codec(
    "aimnet2",
    calctypes={CalcType.energy, CalcType.gradient, CalcType.hessian},
    encode=encode,
    decode=single_point_decoder("aimnet2"),
)
