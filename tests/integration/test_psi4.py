"""Psi4 runs through qccompute: single points, optimization, and QM/MM embedding."""

import numpy as np
import pytest
from qcdata import (
    CalcType,
    Model,
    OptimizationData,
    ProgramInput,
    SinglePointData,
    Structure,
)

from .conftest import Run, requires

pytestmark = [pytest.mark.integration, requires("psi4")]

# HF/STO-3G with exact (PK) integrals on the shared distorted water, Psi4 1.11.
# ORCA 6.0.0 agrees to 2e-7 Hartree and 1e-6 Hartree/bohr.
ENERGY = -74.96441401121291
GRADIENT = [
    [0.032075111496655206, 0.023483904382355314, -0.01694614596310079],
    [-0.013215589271685466, -0.021068551593565354, 0.0003665172769692221],
    [-0.018859522224969172, -0.0024153527887883497, 0.016579628686129333],
]
# Two point charges in Psi4's [charge, [x, y, z]] form, positions in bohr.
CHARGE_VALUES = [-0.8, 0.4]
POSITIONS = [[0.0, 0.0, 5.0], [1.0, 1.0, 6.0]]
CHARGES = [[q, xyz] for q, xyz in zip(CHARGE_VALUES, POSITIONS, strict=True)]


def _input(
    structure: Structure, calctype: CalcType, **keywords: object
) -> ProgramInput:
    return ProgramInput(
        program="psi4",
        structure=structure,
        calctype=calctype,
        model=Model(method="hf", basis="sto-3g"),
        keywords={"scf_type": "pk", **keywords},
    )


def test_energy(run: Run, water: Structure) -> None:
    """Energy plus the extra data Psi4 provides."""
    output = run(_input(water, CalcType.energy), case="water_energy")
    results = output.results
    assert isinstance(results, SinglePointData)
    assert results.provenance.program == "psi4"
    assert results.energy is not None
    assert np.isclose(results.energy, ENERGY, atol=1e-8)
    assert results.calcinfo_nbasis == 7 and results.calcinfo_nalpha == 5
    assert (
        results.atomic_charges is not None and abs(sum(results.atomic_charges)) < 1e-6
    )
    assert results.scf_dipole_moment is not None
    assert "SCF TOTAL ENERGY" in results.extras["psi4_variables"]


def test_gradient_in_input_frame(run: Run, water: Structure) -> None:
    """The gradient is in the input's frame: Psi4 must not move or rotate the molecule."""
    output = run(_input(water, CalcType.gradient), case="water_gradient")
    assert np.isclose(output.results.energy, ENERGY, atol=1e-8)
    assert np.allclose(output.results.gradient, GRADIENT, atol=1e-7)


def test_hessian(run: Run, water: Structure) -> None:
    """Hessian with frequencies, normal modes, and the gradient."""
    output = run(_input(water, CalcType.hessian), case="water_hessian")
    results = output.results
    hessian = np.asarray(results.hessian)
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-8)
    assert np.allclose(results.gradient, GRADIENT, atol=1e-7)
    assert results.freqs_wavenumber
    modes = np.asarray(results.normal_modes_cartesian)
    assert modes.shape == (len(results.freqs_wavenumber), 3, 3)


def test_optimization(run: Run, water: Structure) -> None:
    """optking goes downhill from exactly the input geometry; every step has a gradient."""
    output = run(_input(water, CalcType.optimization), case="water_optimization")
    results = output.results
    assert isinstance(results, OptimizationData)
    trajectory = results.trajectory
    assert np.array_equal(trajectory[0].input_data.structure.geometry, water.geometry)
    assert np.isclose(results.energies[0], ENERGY, atol=1e-8)
    assert np.isclose(results.energies[-1], -74.96590112165362, atol=1e-6)
    assert all(np.diff(results.energies) <= 1e-8)
    assert all(step.results.gradient is not None for step in trajectory)
    assert np.abs(np.asarray(trajectory[-1].results.gradient)).max() < 3e-4


def test_point_charges(run: Run, water: Structure) -> None:
    """QM/MM embedding: the gradient on the charges matches finite differences."""
    output = run(
        _input(water, CalcType.gradient, external_potentials=CHARGES),
        case="water_point_charges_gradient",
    )
    results = output.results
    assert not np.isclose(results.energy, ENERGY, atol=1e-4)  # the charges matter
    analytic = np.asarray(results.point_charge_gradient)
    assert analytic.shape == (2, 3)

    step = 1e-3
    numeric = np.zeros((2, 3))
    for charge in range(2):
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                positions = [list(xyz) for xyz in POSITIONS]
                positions[charge][axis] += sign * step
                moved = [
                    [q, xyz] for q, xyz in zip(CHARGE_VALUES, positions, strict=True)
                ]
                energies.append(
                    run(
                        _input(water, CalcType.energy, external_potentials=moved)
                    ).results.energy
                )
            numeric[charge, axis] = (energies[0] - energies[1]) / (2 * step)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_options_pass_through(run: Run, water: Structure) -> None:
    """Keywords reach psi4.set_options unchanged, including module-scoped options."""
    output = run(
        _input(water, CalcType.energy, scf__d_convergence=1e-10, e_convergence=1e-10)
    )
    assert output.logs is not None
    assert np.isclose(output.results.energy, ENERGY, atol=1e-8)


def test_invalid_option_fails(run: Run, water: Structure) -> None:
    """Psi4 rejects unknown options, and the error says so."""
    with pytest.raises(Exception, match="return code 1"):
        run(_input(water, CalcType.energy, not_a_psi4_option=1))
