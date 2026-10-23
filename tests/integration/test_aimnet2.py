"""AIMNet2 runs in-process through qccompute: energies, gradients, and Hessians."""

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

pytestmark = [pytest.mark.integration, requires_modules("aimnet2", "aimnet")]

# aimnet2 (wB97M-D3) on the shared distorted water, aimnet 0.2.0 on a CPU. The model
# runs in float32, so values agree to about 1e-6 across hardware.
ENERGY = -76.47684061317321
GRADIENT = [
    [-0.009852110226, -0.007473959076, 0.005053746728],
    [0.003530829885, 0.008767232815, 0.001724464767],
    [0.006321277443, -0.001293273015, -0.006778214103],
]


def _input(
    structure: Structure, calctype: CalcType, method: str = "aimnet2"
) -> ProgramInput:
    return ProgramInput(
        program="aimnet2",
        structure=structure,
        calctype=calctype,
        model=Model(method=method),
        keywords={"device": "cpu"},
    )


def test_gradient(run: Run, water: Structure) -> None:
    """Energy, gradient, and partial charges in atomic units."""
    results = run(_input(water, CalcType.gradient)).results
    assert results.provenance.program == "aimnet2"
    assert np.isclose(results.energy, ENERGY, atol=1e-5)
    assert np.allclose(results.gradient, GRADIENT, atol=1e-5)
    assert np.isclose(sum(results.atomic_charges), 0.0, atol=1e-4)


def test_gradient_matches_finite_differences(run: Run, water: Structure) -> None:
    """Gradients are in Hartree/bohr and the input's frame (float32 limits the match)."""
    finite_difference_check(run, _input(water, CalcType.gradient), atol=1e-4)


def test_hessian_matches_finite_differences(run: Run, water: Structure) -> None:
    """The autograd Hessian is symmetric and agrees with differences of gradients."""
    results = run(_input(water, CalcType.hessian)).results
    hessian = np.asarray(results.hessian)
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-6)
    finite_difference_check(
        run, _input(water, CalcType.gradient), atol=1e-4, hessian=True
    )


def test_closed_shell_models_refuse_open_shells(run: Run) -> None:
    """A doublet needs aimnet2-nse; the closed-shell model raises with that advice."""
    hydroxyl = Structure(
        symbols=["O", "H"],
        geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]]),
        multiplicity=2,
    )
    output = run(_input(hydroxyl, CalcType.energy), raise_exc=False)
    assert not output.success
    assert "aimnet2-nse" in str(output.traceback)


def test_batch_matches_single(run: Run) -> None:
    """One batched pass gives each molecule what computing it alone gives."""
    inputs = [_input(m, CalcType.gradient) for m in batch_molecules(charged=True)]
    check_batch_matches_single(run, inputs, atol=1e-6)
