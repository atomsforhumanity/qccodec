"""FAIR-Chem: Meta's UMA and eSEN machine-learned potentials, through ``fairchem-core``.

The codec calls FAIR-Chem's predict unit, the model interface its ASE calculator wraps,
in the calling process; install it next to qccompute with ``pip install fairchem-core``. The pretrained weights are gated
on Hugging Face: request access at https://huggingface.co/facebook/UMA (UMA) or
https://huggingface.co/facebook/OMol25 (eSEN), then log in with ``hf auth login``, or
point ``model.method`` at a local checkpoint. Each model loads once per process and is
reused.

Inputs map to FAIR-Chem as follows:

- ``model.method`` is a pretrained checkpoint or a path to a checkpoint file. FAIR-Chem
  ships three architectures: UMA, one universal model with a head per dataset
  (``uma-s-1p2p1``, ``uma-s-1p2``, ``uma-s-1p1``, ``uma-m-1p1``); eSEN, one model per
  dataset (``esen-sm-conserving-all-omol``, ``esen-md-direct-all-omol``, and OC25 and
  ODAC25 variants); and AllScAIP (``allscaip-md-conserving-all-omol``, ...).
  ``fairchem.core.calculate.pretrained_mlip.available_models`` lists them all. Models
  have no basis set.
- ``structure.charge`` and ``structure.multiplicity`` reach the model through
  ``atoms.info["charge"]`` and ``atoms.info["spin"]``; only ``omol`` heads read them.
- ``keywords``:

  - ``task``: the dataset head to evaluate. UMA models have several: ``omol``
    (molecules, the default), ``omat``, ``oc20``, ``odac``, and ``omc``. eSEN and
    AllScAIP checkpoints are trained on one dataset, which is used unless ``task``
    says otherwise. The head used is recorded in ``extras["fairchem_task"]``.
  - ``device``: ``"cpu"`` or ``"cuda"`` (CUDA when available by default). A model on
    a GPU declares one for qccompute, which gives each calculation of a list its
    own; an explicit index such as ``"cuda:1"`` opts out of that.
  - ``inference_settings``: one of fairchem's presets by name (``"default"``,
    ``"turbo"``, ``"batch"``, ...) or a dictionary of ``InferenceSettings`` fields. The
    codec's default is ``{"compile": False, "merge_mole": False,
    "activation_checkpointing": False}``: the model as trained, ready in seconds for any
    molecule. fairchem's own ``"default"`` compiles the model, which pays off only for
    long runs on one system: compiling takes tens of seconds in every process and again
    for each new molecule size.

- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy``, ``gradient``, and ``hessian`` (by automatic
differentiation; the codec requests it with ``predict_untrained_hessian``). Models
predict energies in eV; the codec converts them, and the forces and Hessian, to Hartree
and bohr.

FAIR-Chem does not move the molecule, so results are in the input's frame. No input
field carries atom indices.

Energies and gradients of several structures can be computed as one batch:
``qccompute.compute`` does this for a list of inputs that share a model, keywords, and
calculation type, evaluating them in one pass instead of one call each. Hessians are
computed one structure at a time.
"""

from __future__ import annotations

import copy
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

DEFAULT_INFERENCE_SETTINGS: dict[str, Any] = {
    "compile": False,
    "merge_mole": False,
    "activation_checkpointing": False,
}


def encode(inp: ProgramInput) -> PythonInput:
    """Collect the model, task, device, inference settings, and molecule in angstrom.

    Raises:
        EncoderError: The model is missing or has a basis, a keyword is unknown or
            invalid, or ``cmdline_args`` or ``files`` were given.
    """
    model = require_model(inp, "uma-s-1p2p1")
    if inp.cmdline_args or inp.files:
        raise EncoderError(
            "FAIR-Chem runs in Python; cmdline_args and files are unused."
        )
    keywords = dict(inp.keywords)
    device = pop_device(keywords)
    reject_unknown("fairchem", keywords, frozenset({"task", "inference_settings"}))
    settings = keywords.get("inference_settings", DEFAULT_INFERENCE_SETTINGS)
    if not isinstance(settings, (str, dict)):
        raise EncoderError(
            "inference_settings must be a fairchem preset name or a dictionary of "
            f"InferenceSettings fields, not {settings!r}."
        )
    return PythonInput(
        gpus=gpu_count(device),
        function=invoke_fairchem,
        kwargs={
            "model": model,
            "task": keywords.get("task"),
            "device": device,
            "inference_settings": settings,
            "numbers": inp.structure.atomic_numbers,
            "positions": positions_angstrom(inp.structure),
            "charge": inp.structure.charge,
            "multiplicity": inp.structure.multiplicity,
            "forces": inp.calctype != CalcType.energy,
            "hessian": inp.calctype == CalcType.hessian,
        },
    )


def invoke_fairchem(
    *,
    model: str,
    task: str | None,
    device: str | None,
    inference_settings: str | dict[str, Any],
    numbers: list[int],
    positions: list[list[float]],
    charge: int,
    multiplicity: int,
    forces: bool,
    hessian: bool,
) -> SinglePoint:
    """Run one FAIR-Chem single point with a cached predict unit."""
    device = resolve_device(device)

    def predict_unit(hessian_task: str | None) -> Any:
        return _predict_unit(model, device, inference_settings, hessian_task)

    unit = predict_unit(task if hessian else None)
    if task is None:
        task = default_task(list(unit.dataset_to_tasks))
        if hessian:
            unit = predict_unit(task)
    # The predict unit returns each property of the input's dataset by property name.
    tasks = {t.property for t in unit.dataset_to_tasks[task]}
    wanted = (
        ["energy"] + (["forces"] if forces else []) + (["hessian"] if hessian else [])
    )
    missing = [p for p in wanted if p not in tasks]
    if missing:
        raise ValueError(
            f"FAIR-Chem model {model!r} predicts no {missing} for {task!r}."
        )
    data = atomic_data(
        numbers, positions, charge, multiplicity, task, unit.inference_settings
    )
    prediction = unit.predict(data)
    values = {p: prediction[p].detach().cpu().double().numpy() for p in wanted}
    snapshot = from_ev(
        float(values["energy"].reshape(-1)[0]),
        values.get("forces"),
        values.get("hessian"),
    )
    snapshot["extras"] = {"fairchem_task": task}
    snapshot["version"] = package_version("fairchem-core")
    return snapshot


def atomic_data(
    numbers: list[int],
    positions: list[list[float]],
    charge: int,
    multiplicity: int,
    task: str,
    settings: Any,
) -> Any:
    """Build FAIR-Chem's input for one isolated molecule, as ``AtomicData.from_ase`` does.

    No periodic cell, no precomputed edges (the model builds its own graph), no
    fixed atoms or tags; charge and spin multiplicity as the ``omol`` head expects.
    """
    import torch
    from fairchem.core.datasets.atomic_data import AtomicData

    dtype = settings.base_precision_dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    natoms = len(numbers)
    return AtomicData(
        pos=torch.tensor(positions, dtype=dtype),
        atomic_numbers=torch.tensor(numbers, dtype=torch.long),
        cell=torch.zeros((1, 3, 3), dtype=dtype),
        pbc=torch.zeros((1, 3), dtype=torch.bool),
        natoms=torch.tensor([natoms], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        cell_offsets=torch.empty((0, 3), dtype=dtype),
        nedges=torch.tensor([0], dtype=torch.long),
        charge=torch.tensor([charge], dtype=torch.long),
        spin=torch.tensor([multiplicity], dtype=torch.long),
        fixed=torch.zeros(natoms, dtype=torch.long),
        tags=torch.zeros(natoms, dtype=torch.long),
        dataset=task,
    )


def _predict_unit(
    model: str,
    device: str,
    inference_settings: str | dict[str, Any],
    hessian_task: str | None,
) -> Any:
    """Return the cached predict unit, with a Hessian head for ``hessian_task`` if given."""
    import os

    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import (
        InferenceSettings,
        guess_inference_settings,
    )

    def load() -> Any:
        if isinstance(inference_settings, dict):
            settings = InferenceSettings(**inference_settings)
        else:
            settings = copy.copy(guess_inference_settings(inference_settings))
        if hessian_task is not None:
            settings.predict_untrained_hessian = {hessian_task}
        if model in pretrained_mlip.available_models:
            return pretrained_mlip.get_predict_unit(
                model, inference_settings=settings, device=device
            )
        if os.path.isfile(model):
            return pretrained_mlip.load_predict_unit(
                model, inference_settings=settings, device=device
            )
        raise ValueError(
            f"{model!r} is neither a FAIR-Chem model name "
            f"({sorted(pretrained_mlip.available_models)}) nor a checkpoint file."
        )

    settings_key = (
        repr(sorted(inference_settings.items()))
        if isinstance(inference_settings, dict)
        else inference_settings
    )
    return cached_model(("fairchem", model, device, settings_key, hessian_task), load)


def encode_batch(inputs: Sequence[ProgramInput]) -> PythonInput:
    """Encode inputs that share a batch key as one FAIR-Chem prediction.

    Raises:
        EncoderError: An input is invalid on its own (see ``encode``).
    """
    singles = [encode(inp).kwargs for inp in inputs]
    first = singles[0]
    return PythonInput(
        gpus=gpu_count(first["device"]),
        function=invoke_fairchem_batch,
        kwargs={
            **{k: first[k] for k in ("model", "task", "device", "inference_settings")},
            "forces": first["forces"],
            "molecules": [
                (k["numbers"], k["positions"], k["charge"], k["multiplicity"])
                for k in singles
            ],
        },
    )


def invoke_fairchem_batch(
    *,
    model: str,
    task: str | None,
    device: str | None,
    inference_settings: str | dict[str, Any],
    forces: bool,
    molecules: list[tuple[list[int], list[list[float]], int, int]],
) -> list[SinglePoint]:
    """Evaluate several molecules in one FAIR-Chem prediction (one batched graph)."""
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    device = resolve_device(device)
    unit = _predict_unit(model, device, inference_settings, None)
    task = task or default_task(list(unit.dataset_to_tasks))
    wanted = ["energy"] + (["forces"] if forces else [])
    missing = [
        p for p in wanted if p not in {t.property for t in unit.dataset_to_tasks[task]}
    ]
    if missing:
        raise ValueError(
            f"FAIR-Chem model {model!r} predicts no {missing} for {task!r}."
        )
    batch = atomicdata_list_to_batch(
        [
            atomic_data(n, p, q, m, task, unit.inference_settings)
            for n, p, q, m in molecules
        ]
    )
    prediction = unit.predict(batch)
    energies = prediction["energy"].detach().cpu().double().numpy().reshape(-1)
    counts = [len(n) for n, _, _, _ in molecules]
    per_forces = (
        split_atoms(prediction["forces"].detach().cpu().double().numpy(), counts)
        if forces
        else [None] * len(molecules)
    )
    version = package_version("fairchem-core")
    results: list[SinglePoint] = []
    for energy, f in zip(energies, per_forces, strict=True):
        result = from_ev(float(energy), f)
        result["extras"] = {"fairchem_task": task}
        result["version"] = version
        results.append(result)
    return results


def default_task(datasets: list[str]) -> str:
    """Pick the task head when none is given: a model's only one, else ``omol``.

    Raises:
        ValueError: The model has several heads and none is ``omol``.
    """
    if len(datasets) == 1:
        return datasets[0]
    if "omol" in datasets:
        return "omol"
    raise ValueError(
        f"This model has several task heads {sorted(datasets)}; choose one with the "
        "'task' keyword."
    )


codec = Codec(
    "fairchem",
    calctypes={CalcType.energy, CalcType.gradient, CalcType.hessian},
    encode=encode,
    decode=single_point_decoder("fairchem"),
    encode_batch=encode_batch,
    batch_key=batch_signature,
)
