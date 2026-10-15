"""Psi4 encoding and the driver script's Psi4-free helpers."""

import json

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from qccodec import get_codec
from qccodec.exceptions import EncoderError
from qccodec.native import SubprocessInput
from qccodec.programs import _psi4_driver as driver

WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array([[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]),
)


def _encode(**kwargs: object) -> SubprocessInput:
    values: dict[str, object] = {
        "program": "psi4",
        "structure": WATER,
        "calctype": CalcType.gradient,
        "model": Model(method="hf", basis="sto-3g"),
        **kwargs,
    }
    native = get_codec("psi4").encode(ProgramInput(**values))  # type: ignore[arg-type]
    assert isinstance(native, SubprocessInput)
    return native


def _job(native: SubprocessInput) -> dict:
    return json.loads(native.files["job.json"])


def test_command_and_scratch() -> None:
    """Psi4 runs the driver without its preprocessor, logging to stdout."""
    native = _encode(cmdline_args=["-n", "4"])
    assert list(native.command) == ["psi4", "input.py", "-k", "-o", "stdout", "-n", "4"]
    assert native.environment["PSI_SCRATCH"] == "."
    assert native.files["input.py"] == open(driver.__file__).read()


def test_method_string() -> None:
    """The basis joins the method; without one the method stands alone."""
    assert _job(_encode())["method"] == "hf/sto-3g"
    assert (
        _job(_encode(model=Model(method="mp2/cc-pv[dt]z")))["method"]
        == "mp2/cc-pv[dt]z"
    )


def test_keywords_split_into_options_and_driver_arguments() -> None:
    """Psi4 options pass through untouched; driver arguments are set aside."""
    charges = [[-0.8, [0.0, 0.0, 5.0]]]
    job = _job(
        _encode(
            keywords={
                "scf_type": "pk",
                "scf__maxiter": 50,
                "external_potentials": charges,
                "dertype": 0,
            }
        )
    )
    assert job["options"] == {"scf_type": "pk", "scf__maxiter": 50}
    assert job["kwargs"] == {"external_potentials": charges, "dertype": 0}


def test_molecule_is_in_bohr() -> None:
    """The structure travels in bohr with its charge and multiplicity."""
    molecule = _job(_encode())["molecule"]
    assert molecule["geometry"] == WATER.geometry.tolist()
    assert (molecule["charge"], molecule["multiplicity"]) == (0, 1)


def test_model_required() -> None:
    """Psi4 needs a method."""
    inp = ProgramInput(program="psi4", structure=WATER, calctype=CalcType.energy)
    with pytest.raises(EncoderError, match="model"):
        get_codec("psi4").encode(inp)


def test_driver_molecule_block_fixes_the_frame() -> None:
    """The molecule block stops Psi4 from moving, rotating, or symmetrizing it."""
    block = driver.molecule_string(
        {
            "symbols": ["H", "H"],
            "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            "charge": 0,
            "multiplicity": 1,
        }
    )
    lines = block.splitlines()
    assert lines[0] == "0 1"
    assert lines[2] == "H 0.0 0.0 1.4"
    assert {"units bohr", "no_com", "no_reorient", "symmetry c1"} <= set(lines)


def test_driver_wavenumbers() -> None:
    """Imaginary frequencies become negative numbers."""
    assert driver.wavenumbers([1500.0 + 0j, 0 + 300.0j]) == [1500.0, -300.0]


def test_driver_to_plain() -> None:
    """Arrays and nested containers become JSON-ready lists."""
    value = {"a": np.arange(3), "b": (np.float64(1.5), [np.zeros((1, 2))])}
    assert driver.to_plain(value) == {"a": [0, 1, 2], "b": [1.5, [[[0.0, 0.0]]]]}
