"""TorchANI runs in-process through qccompute: energies, gradients, and Hessians."""

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from .conftest import (
    Run,
    batch_molecules,
    check_batch_matches_single,
    finite_difference_check,
    requires_modules,
)

pytestmark = [pytest.mark.integration, requires_modules("torchani", "torchani")]

# ANI-2x in float64 on the shared distorted water, torchani 2.9.0.
ENERGY = -76.38826228686709
GRADIENT = [
    [-0.004054435464, -0.00319315389, 0.002011595265],
    [0.001283733915, 0.004405966143, 0.001334484637],
    [0.002770701549, -0.001212812253, -0.003346079902],
]


def _input(structure: Structure, calctype: CalcType) -> ProgramInput:
    return ProgramInput(
        program="torchani",
        structure=structure,
        calctype=calctype,
        model=Model(method="ANI2x"),
        keywords={"device": "cpu"},
    )


def test_gradient(run: Run, water: Structure) -> None:
    """Energy and gradient in atomic units."""
    results = run(_input(water, CalcType.gradient)).results
    assert results.provenance.program == "torchani"
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
        run, _input(water, CalcType.gradient), step=1e-5, atol=1e-6, hessian=True
    )


def test_batch_matches_single(run: Run) -> None:
    """One batched pass gives each molecule what computing it alone gives."""
    inputs = [_input(m, CalcType.gradient) for m in batch_molecules(charged=False)]
    check_batch_matches_single(run, inputs, atol=1e-8)
