"""MACE runs in-process through qccompute: energies, gradients, and Hessians."""

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from .conftest import Run, finite_difference_check, requires_modules

pytestmark = [pytest.mark.integration, requires_modules("mace", "mace")]

# MACE-OFF23 small (float64) on the shared distorted water, mace-torch 0.3.16.
ENERGY = -76.47960634641302
GRADIENT = [
    [-0.009377798695, -0.007125506981, 0.004803843676],
    [0.003309266232, 0.008543309447, 0.00180569182],
    [0.006068532464, -0.001417802466, -0.006609535496],
]


def _input(structure: Structure, calctype: CalcType) -> ProgramInput:
    return ProgramInput(
        program="mace",
        structure=structure,
        calctype=calctype,
        model=Model(method="mace-off-small"),
        keywords={"device": "cpu"},
    )


def test_gradient(run: Run, water: Structure) -> None:
    """Energy and gradient in atomic units."""
    results = run(_input(water, CalcType.gradient)).results
    assert results.provenance.program == "mace"
    assert np.isclose(results.energy, ENERGY, atol=1e-8)
    assert np.allclose(results.gradient, GRADIENT, atol=1e-8)


def test_gradient_matches_finite_differences(run: Run, water: Structure) -> None:
    """Gradients are in Hartree/bohr and the input's frame."""
    finite_difference_check(run, _input(water, CalcType.gradient), atol=1e-6)


def test_hessian_matches_finite_differences(run: Run, water: Structure) -> None:
    """The autograd Hessian is symmetric and agrees with differences of gradients."""
    hessian = np.asarray(run(_input(water, CalcType.hessian)).results.hessian)
    assert np.allclose(hessian, hessian.T, atol=1e-10)
    finite_difference_check(
        run, _input(water, CalcType.gradient), atol=1e-5, hessian=True
    )
