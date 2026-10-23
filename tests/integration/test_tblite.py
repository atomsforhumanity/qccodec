"""tblite runs in-process through qccompute: GFN2-xTB energies and gradients."""

import numpy as np
import pytest
from qcdata import (
    CalcType,
    Model,
    OptimizationData,
    ProgramInput,
    ProgramSpec,
    Structure,
)

from .conftest import Run, finite_difference_check, requires_modules
from .test_xtb import ENERGY as XTB_ENERGY
from .test_xtb import GRADIENT as XTB_GRADIENT

pytestmark = [pytest.mark.integration, requires_modules("tblite", "tblite")]

# GFN2-xTB on the shared distorted water, measured with tblite 0.7.0.
ENERGY = -5.070218274765011
GRADIENT = [
    [-0.01079765445, -0.008149567808, 0.005562987502],
    [0.004813803088, 0.006288187121, -0.000938382952],
    [0.005983851362, 0.001861380686, -0.004624604551],
]


def _input(
    structure: Structure, calctype: CalcType, **keywords: object
) -> ProgramInput:
    return ProgramInput(
        program="tblite",
        structure=structure,
        calctype=calctype,
        model=Model(method="gfn2"),
        keywords=keywords,
    )


def test_gradient(run: Run, water: Structure) -> None:
    """GFN2-xTB energy, gradient, charges, and dipole; the xtb program agrees."""
    results = run(_input(water, CalcType.gradient)).results
    assert results.provenance.program == "tblite"
    assert np.isclose(results.energy, ENERGY, atol=1e-8)
    assert np.allclose(results.gradient, GRADIENT, atol=1e-8)
    # The same method in the xtb program, measured independently.
    assert np.isclose(results.energy, XTB_ENERGY, atol=1e-7)
    assert np.allclose(results.gradient, XTB_GRADIENT, atol=1e-5)
    assert np.allclose(results.atomic_charges, [-0.561, 0.2805, 0.2805], atol=1e-3)
    assert results.scf_dipole_moment is not None


def test_energy_has_no_gradient(run: Run, water: Structure) -> None:
    """An energy calculation returns only what was asked for."""
    results = run(_input(water, CalcType.energy)).results
    assert np.isclose(results.energy, ENERGY, atol=1e-8)
    assert results.gradient is None


def test_gradient_matches_finite_differences(run: Run, water: Structure) -> None:
    """Gradients are in Hartree/bohr and the input's frame."""
    finite_difference_check(run, _input(water, CalcType.gradient), atol=1e-6)


def test_implicit_solvent_and_settings(run: Run, water: Structure) -> None:
    """ALPB water lowers the energy; settings reach the calculator."""
    solvated = run(
        _input(water, CalcType.energy, **{"alpb-solvation": "water", "accuracy": 0.1})
    ).results
    assert solvated.energy < ENERGY - 1e-3


def test_geometric_optimization_with_tblite_gradients(
    run: Run, water: Structure
) -> None:
    """geomeTRIC drives tblite: the library codecs optimize through it."""
    pytest.importorskip("geometric")
    output = run(
        ProgramInput(
            program="geometric",
            calctype=CalcType.optimization,
            structure=water,
            subprograms=[
                ProgramSpec(
                    program="tblite",
                    calctype=CalcType.gradient,
                    model=Model(method="gfn2"),
                )
            ],
        )
    )
    assert isinstance(output.results, OptimizationData)
    trajectory = output.results.frames
    assert np.allclose(trajectory[0].structure.geometry, water.geometry)
    final = trajectory[-1].results
    assert final.energy is not None and final.energy < ENERGY
    assert final.gradient is not None and np.abs(final.gradient).max() < 5e-4
