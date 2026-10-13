"""xTB translation and native Python binding; imports xTB only when invoked."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qcdata import (
    CalcType,
    ProgramInput,
    Provenance,
    SinglePointData,
    Wavefunction,
)

from ..definition import CodecDefinition
from ..exceptions import EncoderError
from ..native import NativeOutput, PythonInput, PythonOutput


def encode(inp: ProgramInput) -> PythonInput:
    """Prepare native arguments and files without executing the scientific program."""
    if inp.model is None:
        raise EncoderError("xtb requires a scientific model.")
    if inp.files or inp.cmdline_args:
        raise EncoderError(
            "xtb's Python interface does not accept files or command-line arguments."
        )
    return PythonInput(
        function=invoke_xtb,
        environment={"OMP_NUM_THREADS": "1", "GFORTRAN_UNBUFFERED_ALL": "1"},
        kwargs={
            "method": inp.model.method,
            "numbers": list(inp.structure.atomic_numbers),
            "positions": inp.structure.geometry.copy(),
            "charge": inp.structure.charge,
            "uhf": inp.structure.multiplicity - 1,
            "settings": deepcopy(inp.keywords),
        },
    )


def invoke_xtb(
    *,
    method: str,
    numbers: list[int],
    positions: NDArray[np.float64],
    charge: int,
    uhf: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Run the native calculator and snapshot results before releasing its output unit."""
    from importlib.metadata import version

    import numpy as np
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_FULL

    if method not in Param.__members__:
        raise EncoderError(
            f"Unsupported xTB method {method!r}: {list(Param.__members__)}"
        )
    calc = Calculator(
        getattr(Param, method), np.asarray(numbers), positions, charge, uhf
    )
    output_file = Path("xtb.log")
    try:
        # libxtb's Fortran stdout may flush after FD capture has ended. Binding an
        # output unit gives release_output() a file it can close synchronously.
        calc.set_output(str(output_file))
        calc.set_verbosity(VERBOSITY_FULL)
        for name, value in settings.items():
            getattr(calc, f"set_{name}")(value)
        result = calc.singlepoint()
        return {
            "energy": result.get_energy(),
            "gradient": np.array(result.get_gradient(), copy=True),
            "dipole": np.array(result.get_dipole(), copy=True),
            "orbital_eigenvalues": np.array(
                result.get_orbital_eigenvalues(), copy=True
            ),
            "version": version("xtb"),
        }
    finally:
        calc.release_output()
        if output_file.exists():
            print(output_file.read_text(errors="replace"), end="")


def decode(
    _definition: CodecDefinition, inp: ProgramInput, out: NativeOutput
) -> SinglePointData:
    """Interpret native observations and preserve recoverable scientific data."""
    assert isinstance(out, PythonOutput)
    if out.failed:
        return SinglePointData(provenance=Provenance(program="xtb"))
    values = out.value
    return SinglePointData(
        provenance=Provenance(program="xtb", program_version=values["version"]),
        energy=values["energy"],
        gradient=values["gradient"],
        scf_dipole_moment=values["dipole"],
        wavefunction=Wavefunction(scf_eigenvalues_a=values["orbital_eigenvalues"]),
    )


codec = CodecDefinition(
    "xtb", frozenset({CalcType.energy, CalcType.gradient}), encode, decode
)
