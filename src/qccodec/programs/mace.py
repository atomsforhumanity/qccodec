"""MACE: the MACE foundation models and custom MACE potentials, through ``mace-torch``.

The codec loads MACE's raw PyTorch models (the loaders' ``return_raw_model=True``) and
calls them directly in the calling process, as MACE's ASE calculator does internally;
install MACE next to qccompute with ``pip install mace-torch``. ``mace-torch`` and ``fairchem-core`` pin
incompatible versions of ``e3nn``, so they need separate environments. Models download
on first use, and each is loaded once per process and reused.

Inputs map to MACE as follows:

- ``model.method`` names a foundation model as ``<family>`` or ``<family>-<model>``,
  where ``<model>`` is the argument MACE's loader takes:

  - ``mace-off``: MACE-OFF23 organic force field (``mace-off-small``,
    ``mace-off-medium``, ``mace-off-large``; medium by default);
  - ``mace-mp``: Materials Project models (``mace-mp-medium``, ``mace-mp-medium-mpa-0``,
    ...; see ``mace.calculators.mace_mp``);
  - ``mace-omol``: MACE-OMOL-0, trained on OMol25 (``mace-omol-extra_large``, the
    default), the one family that reads charge and spin;

  or a path to a ``.model`` file. Models have no basis set. The codec uses a model's
  ``omol`` head for MACE-OMOL, its only head, or its ``default`` head.
- ``mace-omol`` gets ``structure.charge`` and ``structure.multiplicity`` through
  ``atoms.info["charge"]`` and ``atoms.info["spin"]``. The other models describe
  neutral closed-shell molecules only, so the codec raises for anything else.
- ``keywords``: ``device`` (``"cpu"``, ``"cuda"``; CUDA when available by default)
  and ``default_dtype`` (``"float64"`` or ``"float32"``; MACE's default for the model
  otherwise: float64 for MACE-OFF and MACE-OMOL, float32 for MACE-MP). Nothing else is
  accepted. A model on a GPU declares one for qccompute, which gives each calculation
  of a list its own; an explicit index such as ``"cuda:1"`` opts out of that.
- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy``, ``gradient``, and ``hessian`` (by automatic
differentiation, for ``MACE`` models). MACE predicts energies in eV; the codec
converts them, and the forces and Hessian, to Hartree and bohr.

MACE does not move the molecule, so results are in the input's frame. No input field
carries atom indices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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

# model.method prefix -> mace.calculators loader; the longest prefix wins.
FAMILIES: dict[str, str] = {
    "mace-off": "mace_off",
    "mace-mp": "mace_mp",
    "mace-omol": "mace_omol",
}
# Families that read total charge and spin multiplicity from atoms.info.
CHARGED_FAMILIES = {"mace-omol"}
DTYPES = ("float64", "float32")


def split_method(method: str) -> tuple[str | None, str | None]:
    """Split ``mace-off-small`` into the family and MACE's model argument.

    Returns:
        ``(family, model)``: ``model`` is None for the family's default model;
        ``family`` is None when ``method`` is a path to a model file.
    """
    lower = method.lower()
    for family in sorted(FAMILIES, key=len, reverse=True):
        if lower == family:
            return family, None
        if lower.startswith(family + "-"):
            return family, method[len(family) + 1 :]
    return None, method


def encode(inp: ProgramInput) -> PythonInput:
    """Collect the model, device, precision, and molecule in angstrom.

    Raises:
        EncoderError: The model is missing or not a known family or model file path,
            the molecule is charged or open-shell for a model that cannot describe it,
            a keyword is unknown or invalid, or ``cmdline_args`` or ``files`` were
            given.
    """
    method = require_model(inp, "mace-off-medium")
    family, model = split_method(method)
    if family is None and not method.endswith(".model"):
        raise EncoderError(
            f"Unknown MACE model {method!r}; use one of {sorted(FAMILIES)}, optionally "
            "with a model suffix such as 'mace-off-small', or a path to a .model file."
        )
    if inp.cmdline_args or inp.files:
        raise EncoderError("MACE runs in Python; cmdline_args and files are unused.")
    charged = family in CHARGED_FAMILIES
    if not charged and (inp.structure.charge != 0 or inp.structure.multiplicity != 1):
        raise EncoderError(
            f"{method} describes neutral closed-shell molecules only; got charge "
            f"{inp.structure.charge} and multiplicity {inp.structure.multiplicity}. "
            "MACE-OMOL ('mace-omol') takes charge and spin."
        )
    keywords = dict(inp.keywords)
    device = pop_device(keywords)
    reject_unknown("mace", keywords, frozenset({"default_dtype"}))
    dtype = keywords.get("default_dtype")
    if dtype is not None and dtype not in DTYPES:
        raise EncoderError(
            f"default_dtype must be one of {list(DTYPES)}, not {dtype!r}."
        )
    return PythonInput(
        gpus=gpu_count(device),
        function=invoke_mace,
        kwargs={
            "loader": FAMILIES[family] if family else None,
            "model": model,
            "device": device,
            "default_dtype": dtype,
            "charge_and_spin": (
                (inp.structure.charge, inp.structure.multiplicity) if charged else None
            ),
            "numbers": inp.structure.atomic_numbers,
            "positions": positions_angstrom(inp.structure),
            "forces": inp.calctype != CalcType.energy,
            "hessian": inp.calctype == CalcType.hessian,
        },
    )


# MACE's own default precision for each family's models.
FAMILY_DTYPES = {"mace_off": "float64", "mace_mp": "float32", "mace_omol": "float64"}


def invoke_mace(
    *,
    loader: str | None,
    model: str | None,
    device: str | None,
    default_dtype: str | None,
    charge_and_spin: tuple[int, int] | None,
    numbers: list[int],
    positions: list[list[float]],
    forces: bool,
    hessian: bool,
) -> SinglePoint:
    """Run one MACE single point on the cached raw model."""
    import torch
    from mace import calculators, data
    from mace.tools import torch_geometric, torch_tools, utils

    device = resolve_device(device)
    dtype = default_dtype or (FAMILY_DTYPES[loader] if loader else None)

    def load() -> Any:
        if loader is None:
            network = torch.load(model, map_location=device, weights_only=False)
        else:
            options: dict[str, Any] = {"device": device, "return_raw_model": True}
            if model is not None:
                options["model"] = model
            network = getattr(calculators, loader)(**options)
        if dtype == "float64":
            network = network.double()
        elif dtype == "float32":
            network = network.float()
        network = network.to(device)
        for parameter in network.parameters():
            parameter.requires_grad = False
        return network

    network = cached_model(("mace", loader, model, device, dtype), load)
    heads = list(getattr(network, "heads", ["Default"]))
    head = select_head(heads, loader)
    model_dtype = next(network.parameters()).dtype
    properties: dict[str, Any] = {}
    if charge_and_spin is not None:
        properties = {
            "total_charge": charge_and_spin[0],
            "total_spin": charge_and_spin[1],
        }
    config = data.Configuration(
        atomic_numbers=np.array(numbers),
        positions=np.array(positions),
        properties=properties,
        property_weights={name: 1.0 for name in properties},
        pbc=(False, False, False),
        cell=np.zeros((3, 3)),
        head=head,
    )
    with torch_tools.default_dtype(model_dtype):
        graph = data.AtomicData.from_config(
            config,
            z_table=utils.AtomicNumberTable([int(z) for z in network.atomic_numbers]),
            cutoff=float(network.r_max),
            heads=heads,
        )
    batch = torch_geometric.Batch.from_data_list([graph]).to(device)
    inputs = {
        key: value.to(model_dtype) if torch.is_floating_point(value) else value
        for key, value in batch.to_dict().items()
        if torch.is_tensor(value)
    }
    out = network(
        inputs,
        compute_force=forces or hessian,
        compute_hessian=hessian,
        compute_stress=False,
        training=False,
    )
    snapshot = from_ev(
        out["energy"].detach().cpu().double().item(),
        out["forces"].detach().cpu().double().numpy() if forces else None,
        out["hessian"].detach().cpu().double().numpy() if hessian else None,
    )
    snapshot["version"] = package_version("mace-torch")
    return snapshot


def select_head(heads: list[str], loader: str | None) -> str:
    """Pick the model head as MACE's calculator does.

    MACE-OMOL uses its ``omol`` head; otherwise a model's only head, or the one named
    ``default``.

    Raises:
        ValueError: Several heads and none of them is the default.
    """
    if loader == "mace_omol" and "omol" in heads:
        return "omol"
    if len(heads) == 1:
        return heads[0]
    for head in heads:
        if head.lower() == "default":
            return head
    raise ValueError(f"This MACE model has several heads {heads} and no default.")


codec = Codec(
    "mace",
    calctypes={CalcType.energy, CalcType.gradient, CalcType.hessian},
    encode=encode,
    decode=single_point_decoder("mace"),
)
