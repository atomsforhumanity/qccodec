"""Orb runs in-process through qccompute: energies and gradients."""

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from .conftest import Run, finite_difference_check, requires_modules

pytestmark = [pytest.mark.integration, requires_modules("orb", "orb_models")]

# orb-v3-conservative-omol (float32-high) on the shared distorted water, orb-models
# 0.7.0 on a CPU.
ENERGY = -76.43353424098667
GRADIENT = [
    [-0.0069472726, -0.005334549009, 0.003513098738],
    [0.002615951892, 0.005903477506, 0.000940681075],
    [0.004331320709, -0.000568928352, -0.004453779741],
]


def _input(structure: Structure, calctype: CalcType) -> ProgramInput:
    return ProgramInput(
        program="orb",
        structure=structure,
        calctype=calctype,
        model=Model(method="orb-v3-conservative-omol"),
        keywords={"device": "cpu"},
    )


def test_gradient(run: Run, water: Structure) -> None:
    """Energy and gradient in atomic units."""
    results = run(_input(water, CalcType.gradient)).results
    assert results.provenance.program == "orb"
    assert np.isclose(results.energy, ENERGY, atol=1e-5)
    assert np.allclose(results.gradient, GRADIENT, atol=1e-5)


def test_gradient_matches_finite_differences(run: Run, water: Structure) -> None:
    """Gradients are in Hartree/bohr and the input's frame."""
    finite_difference_check(run, _input(water, CalcType.gradient), atol=1e-5)


def test_charge_and_spin_reach_the_model(run: Run) -> None:
    """The OMol25 model sees the charge: OH- and the neutral radical differ."""
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]])
    anion = Structure(symbols=["O", "H"], geometry=geometry, charge=-1)
    radical = Structure(symbols=["O", "H"], geometry=geometry, multiplicity=2)
    e_anion = run(_input(anion, CalcType.energy)).results.energy
    e_radical = run(_input(radical, CalcType.energy)).results.energy
    assert abs(e_anion - e_radical) > 1e-2
