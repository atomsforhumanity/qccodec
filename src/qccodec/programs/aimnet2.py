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
  Unset, it runs on a GPU when one is visible, else on the CPU; it warns
  ``backend_fallback`` only when qccompute granted it a GPU and it still ran
  on the CPU (a CPU-only PyTorch). ``resolved.settings`` names the device used.
- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy``, ``gradient``, and ``hessian`` (by automatic
differentiation). Every result records the model's partial charges in
``atomic_charges``. AIMNet2 predicts energies in eV; the codec converts them, and the
forces and Hessian, to Hartree and bohr.

AIMNet2 does not move the molecule, so results are in the input's frame. No input
field carries atom indices.

The numerics identity of a calculation is aimnet's version and the sha256 of the
model's weights file: the file given as ``model.method``, or the one a registry name
(aimnet 0.2's ``model_registry.yaml``, aliases included) names in aimnet's cache
(``AIMNET_CACHE_DIR``, else ``~/.cache/aimnet``) once it is downloaded. Before the
first download, and for Hugging Face repositories, aimnet's version stands alone.

Energies and gradients of several structures can be computed as one batch:
``qccompute.compute`` does this for a list of inputs that share a model, keywords, and
calculation type, evaluating them in one pass instead of one call each. Hessians are
computed one structure at a time.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qcdata import CalcType, ProgramInput

from ..codec import Codec
from ..exceptions import EncoderError
from ..native import PythonInput
from ..program import CodecProgram
from ._common import reject_unknown
from ._python import (
    SinglePoint,
    batch_signature,
    cached_model,
    from_ev,
    gpu_count,
    model_file,
    package_version,
    pop_device,
    positions_angstrom,
    require_model,
    resolve_device,
    single_point_decoder,
    split_atoms,
)

#: The keywords the codec takes; nothing else is accepted.
KEYWORDS = frozenset({"device", "validate_species"})


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
    reject_unknown("aimnet2", keywords, KEYWORDS)
    return PythonInput(
        gpus=gpu_count(device),
        function=invoke_aimnet2,
        kwargs={
            "model": model,
            "device": device,
            "numbers": inp.require_structure().atomic_numbers,
            "positions": positions_angstrom(inp.require_structure()),
            "charge": inp.require_structure().charge,
            "multiplicity": inp.require_structure().multiplicity,
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
    result["device"] = device
    return result


def encode_batch(inputs: Sequence[ProgramInput]) -> PythonInput:
    """Encode inputs that share a batch key as one AIMNet2 evaluation.

    Raises:
        EncoderError: An input is invalid on its own (see ``encode``).
    """
    singles = [encode(inp).kwargs for inp in inputs]
    first = singles[0]
    return PythonInput(
        gpus=gpu_count(first["device"]),
        function=invoke_aimnet2_batch,
        kwargs={
            "model": first["model"],
            "device": first["device"],
            "forces": first["forces"],
            "validate_species": first["validate_species"],
            "molecules": [
                (k["numbers"], k["positions"], k["charge"], k["multiplicity"])
                for k in singles
            ],
        },
    )


def invoke_aimnet2_batch(
    *,
    model: str,
    device: str | None,
    forces: bool,
    validate_species: bool,
    molecules: list[tuple[list[int], list[list[float]], int, int]],
) -> list[SinglePoint]:
    """Evaluate several molecules in one AIMNet2 pass: atoms concatenated, mol_idx."""
    import numpy as np
    from aimnet.calculators import AIMNet2Calculator

    device = resolve_device(device)
    calculator = cached_model(
        ("aimnet2", model, device), lambda: AIMNet2Calculator(model, device=device)
    )
    counts = [len(numbers) for numbers, _, _, _ in molecules]
    data: dict[str, Any] = {
        "coord": np.concatenate([np.asarray(p) for _, p, _, _ in molecules]),
        "numbers": np.concatenate([np.asarray(n) for n, _, _, _ in molecules]),
        "charge": np.array([float(q) for _, _, q, _ in molecules]),
        "mol_idx": np.repeat(np.arange(len(molecules)), counts),
    }
    multiplicities = [m for _, _, _, m in molecules]
    if calculator.is_nse:
        data["mult"] = np.array(multiplicities, dtype=float)
    elif any(m != 1 for m in multiplicities):
        raise ValueError(
            f"AIMNet2 model {model!r} is closed-shell; use 'aimnet2-nse' for open "
            "shells."
        )
    out = calculator.eval(data, forces=forces, validate_species=validate_species)
    energies = out["energy"].detach().cpu().numpy().reshape(-1)
    charges = split_atoms(out["charges"].detach().cpu().numpy().reshape(-1), counts)
    per_forces = (
        split_atoms(out["forces"].detach().cpu().numpy().reshape(-1, 3), counts)
        if forces
        else [None] * len(molecules)
    )
    version = package_version("aimnet")
    results: list[SinglePoint] = []
    for energy, f, q in zip(energies, per_forces, charges, strict=True):
        result = from_ev(float(energy), f)
        result["atomic_charges"] = q.tolist()
        result["version"] = version
        result["device"] = device
        results.append(result)
    return results


def weights(spec: Any) -> Path | None:
    """The file of a spec's model weights on disk, for its numerics identity.

    Args:
        spec: A `qcdata.ProgramInput` or `qcdata.ProgramSpec`.

    Returns:
        The model file given as ``model.method``; else the file a registry name or
        alias names in aimnet's cache, once downloaded; else None.
    """
    path = model_file(spec)
    if path is not None or spec.model is None:
        return path
    registry = _registry()
    name = registry.get("aliases", {}).get(spec.model.method, spec.model.method)
    entry = registry.get("models", {}).get(name)
    if not isinstance(entry, dict) or not entry.get("file"):
        return None
    cache = os.environ.get("AIMNET_CACHE_DIR") or Path.home() / ".cache" / "aimnet"
    path = Path(cache) / str(entry["file"])
    return path if path.is_file() else None


@functools.cache
def _registry() -> dict[str, Any]:
    """Aimnet's model registry, read without importing aimnet (and PyTorch).

    Empty when aimnet is not installed or has no ``calculators/model_registry.yaml``
    (versions before 0.2).
    """
    found = importlib.util.find_spec("aimnet")
    if found is None or not found.submodule_search_locations:
        return {}
    path = Path(
        found.submodule_search_locations[0], "calculators", "model_registry.yaml"
    )
    try:
        import yaml  # an aimnet dependency

        loaded = yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 - no PyYAML, no file or bad YAML: no registry
        return {}
    return loaded if isinstance(loaded, dict) else {}


codec = Codec(
    "aimnet2",
    calctypes={CalcType.energy, CalcType.gradient, CalcType.hessian},
    encode=encode,
    decode=single_point_decoder("aimnet2"),
    encode_batch=encode_batch,
    batch_key=batch_signature,
    converges=False,  # a model has no convergence settings
    # The model cannot describe the structure's charge or spin.
    errors=[(r"is closed-shell; use 'aimnet2-nse'", "input", "not_modeled")],
)

#: The program qccompute runs (the ``qccompute.programs`` entry point 'aimnet2').
program = CodecProgram(
    codec,
    in_process=False,
    gpu=True,
    library="aimnet",
    weights=weights,
    keywords=KEYWORDS,
)
