"""FAIR-Chem runs in-process through qccompute: UMA energies, gradients, and Hessians."""

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from .conftest import (
    Run,
    finite_difference_check,
    requires_hugging_face,
    requires_modules,
)

pytestmark = [
    pytest.mark.integration,
    requires_modules("fairchem", "fairchem"),
    requires_hugging_face("fairchem"),
]


def _input(
    structure: Structure, calctype: CalcType, method: str = "uma-s-1p2p1"
) -> ProgramInput:
    return ProgramInput(
        program="fairchem",
        structure=structure,
        calctype=calctype,
        model=Model(method=method),
        keywords={"device": "cpu"},
    )


# uma-s-1p2p1 (omol head, uncompiled, float32) on the shared distorted water,
# fairchem-core 2.23 on a CPU; fairchem's own calculator agrees to 2e-8 Hartree.
ENERGY = -76.43351672972503


def test_gradient_matches_finite_differences(run: Run, water: Structure) -> None:
    """Energy against the reference; gradients in Hartree/bohr and the input's frame."""
    results = run(_input(water, CalcType.gradient)).results
    assert results.provenance.program == "fairchem"
    assert results.extras["fairchem_task"] == "omol"
    assert results.energy is not None and abs(results.energy - ENERGY) < 1e-5
    finite_difference_check(run, _input(water, CalcType.gradient), atol=1e-4)


def test_hessian_matches_finite_differences(run: Run, water: Structure) -> None:
    """The autograd Hessian is symmetric and agrees with differences of gradients."""
    hessian = np.asarray(run(_input(water, CalcType.hessian)).results.hessian)
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-5)
    finite_difference_check(
        run, _input(water, CalcType.gradient), atol=1e-4, hessian=True
    )


def test_charge_reaches_the_omol_head(run: Run) -> None:
    """OH- and the neutral OH radical get different energies."""
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]])
    anion = Structure(symbols=["O", "H"], geometry=geometry, charge=-1)
    radical = Structure(symbols=["O", "H"], geometry=geometry, multiplicity=2)
    e_anion = run(_input(anion, CalcType.energy)).results.energy
    e_radical = run(_input(radical, CalcType.energy)).results.energy
    assert abs(e_anion - e_radical) > 1e-2


# esen-sm-conserving-all-omol on the shared distorted water, fairchem-core 2.23 on a
# CPU; fairchem's own calculator agrees to 3e-9 Hartree.
ESEN_ENERGY = -76.43351520193589


def test_single_head_checkpoint(run: Run, water: Structure) -> None:
    """eSEN has one head (omol), which the codec uses without a task keyword."""
    inp = _input(water, CalcType.gradient, method="esen-sm-conserving-all-omol")
    results = run(inp).results
    assert results.extras["fairchem_task"] == "omol"
    assert results.energy is not None and abs(results.energy - ESEN_ENERGY) < 1e-5
    finite_difference_check(run, inp, atol=1e-4)
