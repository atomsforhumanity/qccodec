"""DFTB+ runs through qccompute: DFTB3/3ob, SCC-DFTB/mio, and GFN2-xTB."""

import os

import numpy as np
import pytest
from qcdata import CalcType, Model, OptimizationData, ProgramInput, Structure

from .conftest import Run, finite_difference_check, requires
from .test_xtb import ENERGY as XTB_GFN2_ENERGY

pytestmark = [
    pytest.mark.integration,
    requires("dftb+"),
    pytest.mark.skipif(
        not os.environ.get("QCCODEC_DFTB_PARAMETERS"),
        reason="QCCODEC_DFTB_PARAMETERS must hold the 3ob-3-1 and mio-1-1 sets",
    ),
]

# DFTB3/3ob-3-1 on the shared distorted water, measured with DFTB+ 25.1.
DFTB3_ENERGY = -4.07067291016514


def _input(
    structure: Structure,
    calctype: CalcType,
    method: str = "DFTB3",
    basis: str | None = "3ob-3-1",
    **keywords: object,
) -> ProgramInput:
    return ProgramInput(
        program="dftbplus",
        structure=structure,
        calctype=calctype,
        model=Model(method=method, basis=basis),
        # Explicit, so recorded cases encode the same anywhere (see
        # tests/data/dftbplus: the path is replaced by /opt/dftb-params).
        keywords={
            "slater_koster_dir": os.environ["QCCODEC_DFTB_PARAMETERS"],
            **keywords,
        }
        if basis
        else keywords,
    )


def test_dftb3_gradient(run: Run, water: Structure) -> None:
    """DFTB3/3ob energy, gradient, Mulliken charges, and dipole."""
    output = run(_input(water, CalcType.gradient), case="water_dftb3_gradient")
    results = output.results
    assert results.provenance.program == "dftbplus"
    assert np.isclose(results.energy, DFTB3_ENERGY, atol=1e-8)
    assert np.isclose(sum(results.atomic_charges), 0.0, atol=1e-6)
    assert results.scf_dipole_moment is not None


def test_dftb3_gradient_matches_finite_differences(run: Run, water: Structure) -> None:
    """Gradients are in Hartree/bohr and the input's frame."""
    finite_difference_check(run, _input(water, CalcType.gradient), atol=1e-6)


def test_dftb3_hessian(run: Run, water: Structure) -> None:
    """The SecondDerivatives Hessian agrees with differences of gradients."""
    output = run(_input(water, CalcType.hessian), case="water_dftb3_hessian")
    hessian = np.asarray(output.results.hessian)
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-3)
    finite_difference_check(
        run, _input(water, CalcType.gradient), step=1e-4, atol=1e-4, hessian=True
    )


def test_scc_dftb_mio_energy(run: Run, water: Structure) -> None:
    """SCC-DFTB with the mio set runs and differs from DFTB3."""
    output = run(
        _input(water, CalcType.energy, method="SCC-DFTB", basis="mio-1-1"),
        case="water_scc_dftb_energy",
    )
    assert output.results.energy is not None
    assert not np.isclose(output.results.energy, DFTB3_ENERGY, atol=1e-4)


def test_gfn2_xtb_agrees_with_xtb(run: Run, water: Structure) -> None:
    """DFTB+'s GFN2-xTB matches the xtb program's energy."""
    output = run(
        _input(water, CalcType.gradient, method="GFN2-xTB", basis=None),
        case="water_gfn2_gradient",
    )
    assert np.isclose(output.results.energy, XTB_GFN2_ENERGY, atol=1e-6)


def test_optimization(run: Run, water: Structure) -> None:
    """LBFGS goes downhill from the input structure and ends with a small gradient."""
    output = run(_input(water, CalcType.optimization), case="water_dftb3_optimization")
    assert isinstance(output.results, OptimizationData)
    trajectory = output.results.trajectory
    assert np.allclose(
        trajectory[0].input_data.structure.geometry, water.geometry, atol=1e-6
    )
    first_energy = trajectory[0].results.energy
    assert first_energy is not None
    assert np.isclose(first_energy, DFTB3_ENERGY, atol=1e-8)
    assert output.results.energies[-1] < output.results.energies[0]
    final = trajectory[-1].results
    assert final.gradient is not None and np.abs(final.gradient).max() < 1e-4


def test_unknown_keyword_fails(run: Run, water: Structure) -> None:
    """DFTB+ stops on input it does not recognize; the error says where."""
    output = run(
        _input(water, CalcType.energy, Hamiltonian={"NotAnOption": 1}),
        raise_exc=False,
    )
    assert not output.success
    assert "NotAnOption" in str(output.traceback)
