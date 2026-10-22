"""TorchANI: the ANI family of machine-learned potentials.

The codec calls TorchANI (https://github.com/aiqm/torchani) in the calling process;
install it next to qccompute with ``pip install torchani``. Parameters download on
first use, and each model is loaded once per process and reused.

Inputs map to TorchANI as follows:

- ``model.method`` names a model in ``torchani.models``: ``ANI2x`` (ωB97X/6-31G(d),
  H C N O F S Cl), ``ANI1x``, ``ANI1ccx`` (CCSD(T)/CBS, H C N O), ``ANI2xr``,
  ``ANI2dr``, ``ANImbis``, or ``SnnANI2xr``. Case and dashes are ignored, so
  ``ani-2x`` works too. Models have no basis set.
- ANI models describe neutral closed-shell molecules only, so the codec raises for a
  charge other than 0 or a multiplicity other than 1.
- ``keywords``: ``device`` (``"cpu"``, ``"cuda"``; CUDA when available by default),
  ``dtype`` (``"float64"`` by default, for smooth gradients and Hessians; or
  ``"float32"``, faster), and ``model_index`` (one member of the ensemble instead of
  its average). Nothing else is accepted. A model on a GPU declares one for
  qccompute, which gives each calculation of a list its own; an explicit index such
  as ``"cuda:1"`` opts out of that.
- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy``, ``gradient``, and ``hessian`` (by automatic
differentiation). TorchANI predicts energies in Hartree with coordinates in angstrom;
the codec converts gradients and Hessians to bohr. ``ANImbis`` also fills
``atomic_charges``.

TorchANI does not move the molecule, so results are in the input's frame. No input
field carries atom indices.
"""

from __future__ import annotations

from qcdata import CalcType, ProgramInput
from qcref import constants as c

from ..codec import Codec
from ..exceptions import EncoderError
from ..native import PythonInput
from ._python import (
    SinglePoint,
    cached_model,
    gpu_count,
    package_version,
    pop_device,
    positions_angstrom,
    reject_unknown,
    require_model,
    resolve_device,
    single_point_decoder,
)

MODELS = ("ANI1x", "ANI1ccx", "ANI2x", "ANI2xr", "ANI2dr", "ANImbis", "SnnANI2xr")
DTYPES = ("float64", "float32")


def model_name(method: str) -> str:
    """Return the ``torchani.models`` factory for a method name such as ``ani-2x``.

    Raises:
        EncoderError: No TorchANI model has that name.
    """
    lookup = {name.lower(): name for name in MODELS}
    key = method.lower().replace("-", "").replace("_", "")
    if key not in lookup:
        raise EncoderError(
            f"Unknown TorchANI model {method!r}; use one of {list(MODELS)}."
        )
    return lookup[key]


def encode(inp: ProgramInput) -> PythonInput:
    """Collect the model, device, precision, and molecule in angstrom.

    Raises:
        EncoderError: The model is missing or unknown, the molecule is charged or
            open-shell, a keyword is unknown or invalid, or ``cmdline_args`` or
            ``files`` were given.
    """
    model = model_name(require_model(inp, "ANI2x"))
    if inp.cmdline_args or inp.files:
        raise EncoderError(
            "TorchANI runs in Python; cmdline_args and files are unused."
        )
    if inp.structure.charge != 0 or inp.structure.multiplicity != 1:
        raise EncoderError(
            "ANI models describe neutral closed-shell molecules only; got charge "
            f"{inp.structure.charge} and multiplicity {inp.structure.multiplicity}."
        )
    keywords = dict(inp.keywords)
    device = pop_device(keywords)
    reject_unknown("torchani", keywords, frozenset({"dtype", "model_index"}))
    dtype = keywords.get("dtype", "float64")
    if dtype not in DTYPES:
        raise EncoderError(f"dtype must be one of {list(DTYPES)}, not {dtype!r}.")
    model_index = keywords.get("model_index")
    if model_index is not None and not isinstance(model_index, int):
        raise EncoderError(f"model_index must be an integer, not {model_index!r}.")
    return PythonInput(
        gpus=gpu_count(device),
        function=invoke_torchani,
        kwargs={
            "model": model,
            "model_index": model_index,
            "device": device,
            "dtype": dtype,
            "numbers": inp.structure.atomic_numbers,
            "positions": positions_angstrom(inp.structure),
            "forces": inp.calctype != CalcType.energy,
            "hessian": inp.calctype == CalcType.hessian,
        },
    )


def invoke_torchani(
    *,
    model: str,
    model_index: int | None,
    device: str | None,
    dtype: str,
    numbers: list[int],
    positions: list[list[float]],
    forces: bool,
    hessian: bool,
) -> SinglePoint:
    """Run one TorchANI single point with a cached model."""
    import torch
    import torchani

    device = resolve_device(device)
    torch_dtype = getattr(torch, dtype)
    network = cached_model(
        ("torchani", model, model_index, device, dtype),
        lambda: getattr(torchani.models, model)(
            model_index=model_index, device=device, dtype=torch_dtype
        ),
    )
    species = torch.tensor([numbers], device=device)
    coordinates = torch.tensor([positions], device=device, dtype=torch_dtype)
    out = torchani.grad.single_point(
        network,
        species,
        coordinates,
        forces=forces,
        hessians=hessian,
        atomic_charges=model == "ANImbis",
    )
    result: SinglePoint = {"energy": out["energies"].item()}
    if forces:
        gradient = -out["forces"][0].detach().cpu().double() / c.ANGSTROM_TO_BOHR
        result["gradient"] = gradient.tolist()
    if hessian:
        hess = out["hessians"][0].detach().cpu().double() / c.ANGSTROM_TO_BOHR**2
        result["hessian"] = hess.tolist()
    if "atomic_charges" in out:
        result["atomic_charges"] = out["atomic_charges"][0].detach().cpu().tolist()
    result["version"] = package_version("torchani")
    return result


codec = Codec(
    "torchani",
    calctypes={CalcType.energy, CalcType.gradient, CalcType.hessian},
    encode=encode,
    decode=single_point_decoder("torchani"),
)
