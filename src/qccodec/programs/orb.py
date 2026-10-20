"""Orb: Orbital Materials' machine-learned potentials, through ``orb-models``.

The codec calls Orb's models directly in the calling process (``model.predict`` on the
graph Orb's adapter builds), as Orb's ASE calculator does internally; install it next to
qccompute with ``pip install orb-models``. ``orb-models`` and ``fairchem-core`` need
separate environments. Weights download on first use, and each model is loaded once per
process and reused.

Inputs map to Orb as follows:

- ``model.method`` is a name from ``orb_models.forcefield.pretrained
  .ORB_PRETRAINED_MODELS``: ``orb-v3-conservative-omol`` (trained on OMol25, smooth
  conservative forces; the usual choice for molecules), ``orb-v3-direct-omol``
  (faster, direct non-conservative forces), ``orbmol-v2``, or the materials models
  (``orb-v3-conservative-inf-omat``, ...). Models have no basis set.
- Models trained on OMol25 get ``structure.charge`` and ``structure.multiplicity``
  through ``atoms.info["charge"]`` and ``atoms.info["spin"]``. The others describe
  neutral closed-shell systems only, so the codec raises for anything else.
- ``keywords``: ``device`` (``"cpu"``, ``"cuda"``; CUDA when available by default)
  and ``precision`` (``"float32-high"`` by default, ``"float32-highest"``, or
  ``"float64"``). Nothing else is accepted.
- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy`` and ``gradient``. Orb provides no Hessian; use
geomeTRIC with Orb gradients for optimizations and transition states. Orb predicts
energies in eV; the codec converts them, and the forces, to Hartree and bohr.

Orb does not move the molecule, so gradients are in the input's frame. No input field
carries atom indices.
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
    package_version,
    pop_device,
    positions_angstrom,
    reject_unknown,
    require_model,
    resolve_device,
    single_point_decoder,
)

PRECISIONS = ("float32-high", "float32-highest", "float64")


def encode(inp: ProgramInput) -> PythonInput:
    """Collect the model, device, precision, and molecule in angstrom.

    Raises:
        EncoderError: The model is missing or has a basis, a keyword is unknown or
            invalid, or ``cmdline_args`` or ``files`` were given.
    """
    model = require_model(inp, "orb-v3-conservative-omol")
    if inp.cmdline_args or inp.files:
        raise EncoderError("Orb runs in Python; cmdline_args and files are unused.")
    keywords = dict(inp.keywords)
    device = pop_device(keywords)
    reject_unknown("orb", keywords, frozenset({"precision"}))
    precision = keywords.get("precision", "float32-high")
    if precision not in PRECISIONS:
        raise EncoderError(
            f"precision must be one of {list(PRECISIONS)}, not {precision!r}."
        )
    return PythonInput(
        function=invoke_orb,
        kwargs={
            "model": model,
            "device": device,
            "precision": precision,
            "symbols": inp.structure.symbols,
            "positions": positions_angstrom(inp.structure),
            "charge": inp.structure.charge,
            "multiplicity": inp.structure.multiplicity,
            "forces": inp.calctype == CalcType.gradient,
        },
    )


def invoke_orb(
    *,
    model: str,
    device: str | None,
    precision: str,
    symbols: list[str],
    positions: list[list[float]],
    charge: int,
    multiplicity: int,
    forces: bool,
) -> SinglePoint:
    """Run one Orb single point on the cached model and its graph adapter."""
    from ase import Atoms
    from orb_models.common.models.nn_util import ChargeSpinConditioner
    from orb_models.forcefield import pretrained

    if model not in pretrained.ORB_PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown Orb model {model!r}; use one of "
            f"{sorted(pretrained.ORB_PRETRAINED_MODELS)}."
        )
    device = resolve_device(device)

    def load() -> Any:
        return pretrained.ORB_PRETRAINED_MODELS[model](
            device=device, precision=precision
        )

    regressor, adapter = cached_model(("orb", model, device, precision), load)
    # D3-corrected models wrap the regressor that carries the heads and conditioner.
    inner = getattr(regressor, "xc_model", regressor)
    atoms = Atoms(symbols=symbols, positions=positions)
    if isinstance(inner.model.conditioner, ChargeSpinConditioner):
        atoms.info.update({"charge": charge, "spin": multiplicity})
    elif charge != 0 or multiplicity != 1:
        raise ValueError(
            f"Orb model {model!r} describes neutral closed-shell systems only; use "
            "an OMol25 model such as 'orb-v3-conservative-omol'."
        )
    # ASE Atoms is only the container Orb's graph builder reads; no ASE calculator runs.
    batch = adapter.from_ase_atoms(atoms, device=device)
    out = regressor.predict(batch)
    # Conservative models name their heads and return forces as the energy gradient;
    # direct models key their outputs by head ("energy", "forces").
    energy_key = getattr(inner, "energy_name", "energy")
    force_key = getattr(inner, "grad_forces_name", "forces")
    snapshot = from_ev(
        out[energy_key].detach().cpu().double().reshape(-1)[0].item(),
        out[force_key].detach().cpu().double().numpy() if forces else None,
    )
    snapshot["version"] = package_version("orb-models")
    return snapshot


codec = Codec(
    "orb",
    calctypes={CalcType.energy, CalcType.gradient},
    encode=encode,
    decode=single_point_decoder("orb"),
)
