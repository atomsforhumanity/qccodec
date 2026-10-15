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


def test_keywords_split_into_options_driver_arguments_and_molecule() -> None:
    """Psi4 options pass through untouched; the rest go where Psi4 expects them."""
    charges = [[-0.8, [0.0, 0.0, 5.0]]]
    job = _job(
        _encode(
            keywords={
                "scf_type": "pk",
                "scf__maxiter": 50,
                "external_potentials": charges,
                "dertype": 0,
                "project_rot": True,
                "symmetry": "c1",
            }
        )
    )
    assert job["options"] == {"scf_type": "pk", "scf__maxiter": 50}
    assert job["kwargs"] == {
        "external_potentials": charges,
        "dertype": 0,
        "project_rot": True,
    }
    assert job["molecule"]["symmetry"] == "c1"


def test_symmetry_defaults_to_psi4() -> None:
    """Without a symmetry keyword, Psi4 detects symmetry in the fixed frame."""
    assert _job(_encode())["molecule"]["symmetry"] is None


def test_transition_state_options() -> None:
    """A transition-state search sets opt_type and an exact starting Hessian."""
    job = _job(_encode(calctype=CalcType.transition_state))
    assert job["calctype"] == "transition_state"
    assert job["options"] == {"opt_type": "ts", "full_hess_every": 0}
    job = _job(
        _encode(calctype=CalcType.transition_state, keywords={"FULL_HESS_EVERY": 3})
    )
    assert job["options"] == {"FULL_HESS_EVERY": 3, "opt_type": "ts"}


@pytest.mark.parametrize(
    ("calctype", "opt_type"),
    [(CalcType.transition_state, "min"), (CalcType.optimization, "TS")],
)
def test_contradictory_opt_type(calctype: CalcType, opt_type: str) -> None:
    """opt_type must agree with the calculation type."""
    with pytest.raises(EncoderError, match="opt_type"):
        _encode(calctype=calctype, keywords={"opt_type": opt_type})


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
    """The molecule block stops Psi4 from moving or rotating the molecule."""
    molecule = {
        "symbols": ["H", "H"],
        "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        "charge": 0,
        "multiplicity": 1,
    }
    lines = driver.molecule_string(molecule).splitlines()
    assert lines[0] == "0 1"
    assert lines[2] == "H 0.0 0.0 1.4"
    assert {"units bohr", "no_com", "no_reorient"} <= set(lines)
    assert not any(line.startswith("symmetry") for line in lines)
    lines = driver.molecule_string({**molecule, "symmetry": "c1"}).splitlines()
    assert lines[-1] == "symmetry c1"


def test_driver_orbitals_merge_irreps_by_energy() -> None:
    """Per-irrep orbital blocks become one energy-ordered list."""
    energies, occupations = driver.orbitals(
        [[-1.0, 0.5], [], [-0.2]], [[2, 0], [], [2]]
    )
    assert energies == [-1.0, -0.2, 0.5]
    assert occupations == [2, 2, 0]


def test_driver_wavenumbers() -> None:
    """Imaginary frequencies become negative numbers."""
    assert driver.wavenumbers([1500.0 + 0j, 0 + 300.0j]) == [1500.0, -300.0]


def test_driver_to_plain() -> None:
    """Arrays and nested containers become JSON-ready lists."""
    value = {"a": np.arange(3), "b": (np.float64(1.5), [np.zeros((1, 2))])}
    assert driver.to_plain(value) == {"a": [0, 1, 2], "b": [1.5, [[[0.0, 0.0]]]]}


@pytest.mark.parametrize(
    "calctype",
    [CalcType.hessian, CalcType.optimization, CalcType.transition_state],
)
def test_point_charges_only_for_single_points(calctype: CalcType) -> None:
    """Psi4 1.11 mishandles point charges beyond energies and gradients."""
    charges = [[-0.8, [0.0, 0.0, 5.0]]]
    with pytest.raises(EncoderError, match="geomeTRIC"):
        _encode(calctype=calctype, keywords={"external_potentials": charges})
