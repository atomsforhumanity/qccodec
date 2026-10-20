"""tblite: GFN1-xTB, GFN2-xTB, and IPEA1-xTB through tblite's Python API.

tblite (https://tblite.readthedocs.io) is the library implementation of the extended
tight-binding methods. The codec calls ``tblite.interface.Calculator`` in the calling
process, so there is no program to find on ``PATH``: install it next to qccompute
with ``pip install tblite`` (or ``conda install -c conda-forge tblite-python``).

Inputs map to tblite as follows:

- ``model.method`` selects the parametrization: ``gfn2`` (``GFN2-xTB``), ``gfn1``
  (``GFN1-xTB``), or ``ipea1`` (``IPEA1-xTB``); tblite's own names work too. Models
  have no basis set.
- ``structure.charge`` is the total charge and ``structure.multiplicity - 1`` the
  number of unpaired electrons (tblite's ``uhf``).
- ``keywords`` use tblite's own names. Calculator settings go to
  ``Calculator.set``: ``accuracy``, ``guess``, ``max-iter``, ``mixer-damping``,
  ``mixer-memory``, ``temperature`` (electronic temperature in **Hartree**, as tblite
  takes it; multiply kelvin by 3.166808578545117e-6), ``annealing``, and
  ``verbosity``. Interactions go to ``Calculator.add``, with a single value or a list
  of arguments: ``alpb-solvation`` and ``gbsa-solvation`` (a solvent name, or
  ``[solvent, state]`` with state ``gsolv``, ``bar1mol``, or ``reference``),
  ``ddX-solvation``, ``gbe-solvation``, ``gb-solvation``, ``electric-field`` (a
  vector in atomic units), and ``spin-polarization`` (a scaling factor, usually
  ``1.0``).
- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy`` and ``gradient``. tblite has no analytic Hessian; use
``program="xtb"`` for GFN2-xTB Hessians, or geomeTRIC with tblite gradients for
optimizations. Every result records Mulliken partial charges in ``atomic_charges``
and the dipole moment (e bohr) in ``scf_dipole_moment``.

tblite never moves or reorients the molecule, so gradients are in the input's frame.
No input field carries atom indices. An SCF that does not converge raises
``CalculationError`` with tblite's message.
"""

from __future__ import annotations

from typing import Any

from qcdata import CalcType, ProgramInput

from ..codec import Codec
from ..exceptions import EncoderError
from ..native import PythonInput
from ._python import (
    SinglePoint,
    package_version,
    reject_unknown,
    require_model,
    single_point_decoder,
)

METHODS: dict[str, str] = {
    "gfn2": "GFN2-xTB",
    "gfn1": "GFN1-xTB",
    "ipea1": "IPEA1-xTB",
}
# Calculator.set attributes and Calculator.add interactions, by tblite's names.
SETTINGS = frozenset(
    {
        "accuracy",
        "guess",
        "max-iter",
        "mixer-damping",
        "mixer-memory",
        "temperature",
        "annealing",
        "verbosity",
    }
)
INTERACTIONS = frozenset(
    {
        "alpb-solvation",
        "gbsa-solvation",
        "ddX-solvation",
        "gbe-solvation",
        "gb-solvation",
        "electric-field",
        "spin-polarization",
    }
)


def method_name(method: str) -> str:
    """Return tblite's name for a method: ``gfn2`` and ``GFN2-xTB`` give ``GFN2-xTB``.

    Raises:
        EncoderError: The method is not one tblite provides.
    """
    key = method.lower().replace("-", "").replace("_", "").removesuffix("xtb")
    if key not in METHODS:
        raise EncoderError(
            f"Unknown tblite method {method!r}; use one of {sorted(METHODS)}."
        )
    return METHODS[key]


def encode(inp: ProgramInput) -> PythonInput:
    """Collect the calculator's arguments, settings, and interactions.

    Raises:
        EncoderError: The model is missing or unknown, a keyword is not a tblite
            setting or interaction, or ``cmdline_args`` or ``files`` were given.
    """
    method = method_name(require_model(inp, "gfn2"))
    if inp.cmdline_args or inp.files:
        raise EncoderError("tblite runs in Python; cmdline_args and files are unused.")
    reject_unknown("tblite", inp.keywords, SETTINGS | INTERACTIONS)
    interactions: list[tuple[str, list[Any]]] = []
    for name, value in inp.keywords.items():
        if name in INTERACTIONS:
            vector = name == "electric-field"
            args = [value] if vector or not isinstance(value, list) else list(value)
            interactions.append((name, args))
    return PythonInput(
        function=invoke_tblite,
        kwargs={
            "method": method,
            "numbers": inp.structure.atomic_numbers,
            "positions": inp.structure.geometry.tolist(),
            "charge": inp.structure.charge,
            "uhf": inp.structure.multiplicity - 1,
            "settings": {k: v for k, v in inp.keywords.items() if k in SETTINGS},
            "interactions": interactions,
        },
    )


def invoke_tblite(
    *,
    method: str,
    numbers: list[int],
    positions: list[list[float]],
    charge: int,
    uhf: int,
    settings: dict[str, Any],
    interactions: list[tuple[str, list[Any]]],
) -> SinglePoint:
    """Run one tblite single point; tblite works in bohr and Hartree throughout."""
    import numpy as np
    from tblite.interface import Calculator

    calculator = Calculator(
        method, np.array(numbers), np.array(positions), charge=float(charge), uhf=uhf
    )
    for name, value in settings.items():
        calculator.set(name, value)
    for name, args in interactions:
        if name == "electric-field":
            args = [np.asarray(args[0], dtype=float)]
        calculator.add(name, *args)
    result = calculator.singlepoint()
    return {
        "energy": float(result.get("energy")),
        "gradient": np.asarray(result.get("gradient")).tolist(),
        "atomic_charges": np.asarray(result.get("charges")).tolist(),
        "scf_dipole_moment": np.asarray(result.get("dipole")).tolist(),
        "version": package_version("tblite"),
    }


codec = Codec(
    "tblite",
    calctypes={CalcType.energy, CalcType.gradient},
    encode=encode,
    decode=single_point_decoder("tblite"),
)
