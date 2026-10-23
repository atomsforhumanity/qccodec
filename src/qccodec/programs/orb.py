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
  ``"float64"``). Nothing else is accepted. A model on a GPU declares one for
  qccompute, which gives each calculation of a list its own; an explicit index such
  as ``"cuda:1"`` opts out of that.
- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy`` and ``gradient``. Orb provides no Hessian; use
geomeTRIC with Orb gradients for optimizations and transition states. Orb predicts
energies in eV; the codec converts them, and the forces, to Hartree and bohr.

Orb does not move the molecule, so gradients are in the input's frame. No input field
carries atom indices.

Energies and gradients of several structures can be computed as one batch:
``qccompute.compute`` does this for a list of inputs that share a model, keywords, and
calculation type, evaluating them in one pass instead of one call each. Hessians are
computed one structure at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qcdata import CalcType, ProgramInput

from ..codec import Codec
from ..exceptions import EncoderError
from ..native import PythonInput
from ._python import (
    SinglePoint,
    batch_signature,
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
    split_atoms,
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
        gpus=gpu_count(device),
        function=invoke_orb,
        kwargs={
            "model": model,
            "device": device,
            "precision": precision,
            "symbols": inp.require_structure().symbols,
            "positions": positions_angstrom(inp.require_structure()),
            "charge": inp.require_structure().charge,
            "multiplicity": inp.require_structure().multiplicity,
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
    return invoke_orb_batch(
        model=model,
        device=device,
        precision=precision,
        forces=forces,
        molecules=[(symbols, positions, charge, multiplicity)],
    )[0]


def encode_batch(inputs: Sequence[ProgramInput]) -> PythonInput:
    """Encode inputs that share a batch key as one Orb evaluation.

    Raises:
        EncoderError: An input is invalid on its own (see ``encode``).
    """
    singles = [encode(inp).kwargs for inp in inputs]
    first = singles[0]
    return PythonInput(
        gpus=gpu_count(first["device"]),
        function=invoke_orb_batch,
        kwargs={
            **{k: first[k] for k in ("model", "device", "precision", "forces")},
            "molecules": [
                (k["symbols"], k["positions"], k["charge"], k["multiplicity"])
                for k in singles
            ],
        },
    )


def invoke_orb_batch(
    *,
    model: str,
    device: str | None,
    precision: str,
    forces: bool,
    molecules: list[tuple[list[str], list[list[float]], int, int]],
) -> list[SinglePoint]:
    """Evaluate molecules in one Orb pass, as one batched graph."""
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
    charged = isinstance(inner.model.conditioner, ChargeSpinConditioner)
    atoms_list = []
    for symbols, positions, charge, multiplicity in molecules:
        atoms = Atoms(symbols=symbols, positions=positions)
        if charged:
            atoms.info.update({"charge": charge, "spin": multiplicity})
        elif charge != 0 or multiplicity != 1:
            raise ValueError(
                f"Orb model {model!r} describes neutral closed-shell systems only; "
                "use an OMol25 model such as 'orb-v3-conservative-omol'."
            )
        atoms_list.append(atoms)
    # ASE Atoms is only the container Orb's graph builder reads; no ASE calculator runs.
    if len(atoms_list) == 1:
        batch = adapter.from_ase_atoms(atoms_list[0], device=device)
    else:
        batch = adapter.from_ase_atoms_list(atoms_list, device=device)
    out = regressor.predict(batch)
    # Conservative models name their heads and return forces as the energy gradient;
    # direct models key their outputs by head ("energy", "forces").
    energy_key = getattr(inner, "energy_name", "energy")
    force_key = getattr(inner, "grad_forces_name", "forces")
    energies = out[energy_key].detach().cpu().double().numpy().reshape(-1)
    counts = [len(symbols) for symbols, _, _, _ in molecules]
    per_forces = (
        split_atoms(out[force_key].detach().cpu().double().numpy(), counts)
        if forces
        else [None] * len(molecules)
    )
    version = package_version("orb-models")
    results: list[SinglePoint] = []
    for energy, f in zip(energies, per_forces, strict=True):
        result = from_ev(float(energy), f)
        result["version"] = version
        results.append(result)
    return results


codec = Codec(
    "orb",
    calctypes={CalcType.energy, CalcType.gradient},
    encode=encode,
    decode=single_point_decoder("orb"),
    encode_batch=encode_batch,
    batch_key=batch_signature,
)
