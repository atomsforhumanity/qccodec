"""xtb runs through qccompute: GFN-xTB, GFN-FF, and g-xTB."""

import os
from pathlib import Path

import numpy as np
import pytest
from qcdata import CalcType, Model, OptimizationData, ProgramInput, Structure

from .conftest import Run, requires

pytestmark = [pytest.mark.integration, requires("xtb")]

# GFN2-xTB on the shared distorted water, measured with xtb 6.7.1.
ENERGY = -5.070218272202
GRADIENT = [
    [-0.010797157541, -0.008149198592, 0.005562728108],
    [0.004813610511, 0.006287808602, -0.000938419136],
    [0.005983547029, 0.00186138999, -0.004624308973],
]


def _input(
    structure: Structure, calctype: CalcType, method: str = "gfn2", **keywords: object
) -> ProgramInput:
    return ProgramInput(
        program="xtb",
        structure=structure,
        calctype=calctype,
        model=Model(method=method),
        keywords=keywords,
    )


def test_energy(run: Run, water: Structure) -> None:
    """GFN2-xTB energy and partial charges."""
    output = run(_input(water, CalcType.energy), case="water_energy")
    assert output.results.provenance.program == "xtb"
    assert np.isclose(output.results.energy, ENERGY, atol=1e-6)
    assert np.allclose(
        output.results.atomic_charges, [-0.561, 0.2805, 0.2805], atol=1e-3
    )


def test_gradient(run: Run, water: Structure) -> None:
    """GFN2-xTB gradient."""
    output = run(_input(water, CalcType.gradient), case="water_gradient")
    assert np.isclose(output.results.energy, ENERGY, atol=1e-6)
    assert np.allclose(output.results.gradient, GRADIENT, atol=1e-6)


def test_hessian(run: Run, water: Structure) -> None:
    """GFN2-xTB Hessian, frequencies, and normal modes."""
    output = run(_input(water, CalcType.hessian), case="water_hessian")
    hessian = np.asarray(output.results.hessian)
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-5)
    assert np.allclose(
        output.results.freqs_wavenumber, [1587.67, 3520.44, 3527.26], atol=1.0
    )
    assert np.asarray(output.results.normal_modes_cartesian).shape == (3, 3, 3)


@pytest.mark.parametrize("level", [None, "tight"])
def test_optimization(run: Run, water: Structure, level: str | None) -> None:
    """Optimizations reach the GFN2-xTB minimum at the requested level."""
    output = run(
        _input(water, CalcType.optimization, opt=level),
        case=None if level else "water_optimization",
    )
    assert isinstance(output.results, OptimizationData)
    energies = output.results.energies
    assert np.isclose(energies[0], ENERGY, atol=1e-6)
    assert np.isclose(energies[-1], -5.0705444, atol=1e-6)
    assert all(np.diff(energies) <= 1e-8)


def test_gfnff_gradient(run: Run, water: Structure) -> None:
    """GFN-FF writes its charges to a different file; both are parsed."""
    output = run(_input(water, CalcType.gradient, "gfnff"), case="water_gfnff_gradient")
    assert np.isclose(output.results.energy, -0.327530842695, atol=1e-6)
    assert output.results.atomic_charges is not None


def test_gfn1_with_implicit_solvent(run: Run, water: Structure) -> None:
    """Keywords pass through as flags: GFN1-xTB with ALPB water."""
    output = run(_input(water, CalcType.energy, "gfn1", alpb="water"))
    assert output.logs is not None and "alpb" in output.logs.lower()
    assert output.results.energy < -5.768446177392  # solvation lowers the energy


@pytest.mark.skipif(not os.environ.get("XTBPATH"), reason="GFN0 needs XTBPATH")
def test_gfn0(run: Run, water: Structure) -> None:
    """GFN0-xTB reads its parameters from XTBPATH."""
    output = run(_input(water, CalcType.gradient, "gfn0"))
    assert np.isclose(output.results.energy, -4.366772003256, atol=1e-6)


@pytest.fixture
def gxtb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the g-xTB build of xtb, named by GXTB_EXECUTABLE, first on PATH."""
    path = os.environ.get("GXTB_EXECUTABLE")
    if not path:
        pytest.skip("GXTB_EXECUTABLE does not name the g-xTB build of xtb")
    monkeypatch.setenv(
        "PATH", f"{Path(path).resolve().parent}{os.pathsep}{os.environ['PATH']}"
    )


@pytest.mark.usefixtures("gxtb")
@pytest.mark.parametrize("calctype", [CalcType.gradient, CalcType.hessian])
def test_gxtb(run: Run, water: Structure, calctype: CalcType) -> None:
    """g-xTB energies, gradients, and Hessians."""
    output = run(_input(water, calctype, "gxtb"), case=f"water_gxtb_{calctype.value}")
    assert output.logs is not None and "g-xTB" in output.logs
    assert np.isclose(output.results.energy, -76.437424197961, atol=1e-6)
    if calctype == CalcType.hessian:
        assert np.asarray(output.results.hessian).shape == (9, 9)


def test_unknown_option_fails(run: Run, water: Structure) -> None:
    """xtb only warns about unknown flags; the codec turns that into a failure."""
    with pytest.raises(Exception, match="unknown options"):
        run(_input(water, CalcType.energy, gfn2_typo=True))
