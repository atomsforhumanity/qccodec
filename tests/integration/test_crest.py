"""CREST runs through qccompute: single points, optimization, conformer search."""

import numpy as np
import pytest
from qcdata import (
    CalcType,
    ConformerSearchData,
    Model,
    OptimizationData,
    ProgramInput,
    SinglePointData,
    Structure,
)

from .conftest import Run, requires

pytestmark = [pytest.mark.integration, requires("crest")]

# GFN2-xTB on the shared distorted water; CREST 3.0.2 agrees with xtb 6.x here.
ENERGY = -5.070218277097775
GRADIENT = [
    [-0.010797589714437, -0.008149519966979, 0.00556295355879],
    [0.004813784661518, 0.006288115451175, -0.000938407000615],
    [0.005983805052918, 0.001861404515804, -0.004624546558175],
]


def _input(
    structure: Structure, calctype: CalcType, method: str = "gfn2", **kw: object
) -> ProgramInput:
    return ProgramInput(
        program="crest",
        structure=structure,
        calctype=calctype,
        model=Model(method=method),
        keywords={"threads": 1, **kw},
    )


@pytest.mark.parametrize("calctype", [CalcType.energy, CalcType.gradient])
def test_single_point(run: Run, water: Structure, calctype: CalcType) -> None:
    """Energies and gradients match GFN2-xTB reference values."""
    output = run(_input(water, calctype), case=f"water_{calctype.value}")
    results = output.results
    assert isinstance(results, SinglePointData)
    assert results.provenance.program == "crest"
    assert results.energy is not None
    assert np.isclose(results.energy, ENERGY, atol=1e-6)
    if calctype == CalcType.gradient:
        assert results.gradient is not None
        assert np.allclose(results.gradient, GRADIENT, atol=1e-6)


def test_hessian(run: Run, water: Structure) -> None:
    """The Hessian is symmetric and its frequencies are those of bent water."""
    output = run(_input(water, CalcType.hessian), case="water_hessian")
    hessian = np.asarray(output.results.hessian)
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-5)
    assert np.isclose(output.results.energy, ENERGY, atol=1e-6)
    assert np.allclose(
        output.results.freqs_wavenumber, [1588.38, 3520.98, 3527.24], atol=1.0
    )


def test_optimization(run: Run, water: Structure) -> None:
    """The optimization goes downhill to the GFN2-xTB minimum."""
    output = run(_input(water, CalcType.optimization), case="water_optimization")
    assert isinstance(output.results, OptimizationData)
    energies = output.results.energies
    assert len(output.results.trajectory) > 1
    assert np.isclose(energies[0], ENERGY, atol=1e-6)
    assert np.isclose(energies[-1], -5.0705443538, atol=1e-6)
    assert all(np.diff(energies) <= 1e-8)


def test_conformer_search_with_solvent(run: Run, water: Structure) -> None:
    """A GFN-FF conformer search passes the ALPB solvent through to CREST."""
    inp = _input(
        water,
        CalcType.conformer_search,
        method="gfnff",
        calculation={"level": [{"alpb": "acetonitrile"}]},
    )
    output = run(inp, case="water_conformers_alpb")
    assert isinstance(output.results, ConformerSearchData)
    assert len(output.results.conformers) == 1
    assert np.isclose(output.results.conformer_energies[0], -0.33568982, atol=1e-5)
    assert output.logs is not None
    assert "alpb" in output.logs and "acetonitrile" in output.logs
