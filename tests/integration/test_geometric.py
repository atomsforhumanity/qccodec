"""geomeTRIC optimizations driving a real gradient program through qccompute.

These test the interface (child calculations, keywords, wavefunction propagation,
trajectories), not the optimizer, so each starts near its answer and converges in a
few steps. Starting geometries are the converged structures rounded to 0.01 bohr.
"""

from importlib.util import find_spec

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

from .conftest import Run, requires

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        find_spec("geometric") is None, reason="geometric not installed"
    ),
]


def _input(
    structure: Structure, calctype: CalcType, program: str, model: Model, **keywords
) -> ProgramInput:
    return ProgramInput(
        program="geometric",
        calctype=calctype,
        structure=structure,
        subprograms=[
            ProgramSpec(
                program=program,
                calctype=CalcType.gradient,
                model=model,
                keywords=keywords,
            )
        ],
    )


@requires("crest")
def test_optimization_with_crest(run: Run, water: Structure) -> None:
    """geomeTRIC reaches the GFN2-xTB minimum that CREST's own optimizer finds."""
    inp = _input(water, CalcType.optimization, "crest", Model(method="gfn2"), threads=1)
    output = run(inp)
    assert isinstance(output.results, OptimizationData)
    energies = output.results.energies
    assert len(energies) > 1
    assert energies[0] > energies[-1]
    assert np.isclose(energies[-1], -5.0705443538, atol=1e-5)
    assert all(step.input_data.program == "crest" for step in output.results.trajectory)


HF_STO3G = Model(method="hf", basis="sto-3g")


# Near the HF/STO-3G minimum of water (bohr).
NEAR_HF_WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array([[-0.04, -0.03, 0.02], [0.54, 1.69, 0.46], [1.16, -0.43, -1.36]]),
)


@requires("psi4")
def test_optimization_with_psi4(run: Run) -> None:
    """geomeTRIC drives Psi4 gradients to the HF/STO-3G minimum optking also finds."""
    inp = _input(NEAR_HF_WATER, CalcType.optimization, "psi4", HF_STO3G, scf_type="pk")
    output = run(inp)
    assert isinstance(output.results, OptimizationData)
    assert np.isclose(output.results.energies[-1], -74.96590112, atol=1e-5)


# Two weak point charges [charge, [x, y, z]] (bohr) and water near its QM/MM minimum,
# where it has drifted toward them.
QMMM_CHARGES = [[-0.2, [0.0, 0.0, 8.0]], [0.2, [2.0, 2.0, 9.0]]]
NEAR_QMMM_WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array(
        [[0.51, 0.37, -3.03], [-1.05, -0.66, -3.03], [1.14, -0.01, -1.31]]
    ),
)


@requires("psi4")
def test_qmmm_optimization_with_psi4(run: Run) -> None:
    """QM/MM geometry optimization: geomeTRIC moves the QM atoms; charges stay fixed."""
    inp = _input(
        NEAR_QMMM_WATER,
        CalcType.optimization,
        "psi4",
        HF_STO3G,
        scf_type="pk",
        external_potentials=QMMM_CHARGES,
    )
    output = run(inp)
    assert isinstance(output.results, OptimizationData)
    energies = output.results.energies
    assert len(energies) > 1 and energies[0] >= energies[-1]
    assert np.isclose(energies[-1], -74.96621433, atol=1e-6)
    for step in output.results.trajectory:
        assert step.input_data.keywords["external_potentials"] == QMMM_CHARGES
        assert step.results.point_charge_gradient is not None


@requires("terachem")
def test_optimization_with_terachem_propagates_wavefunction(
    run: Run, hydrogen: Structure
) -> None:
    """Each TeraChem step after the first starts from the previous wavefunction."""
    inp = _input(hydrogen, CalcType.optimization, "terachem", HF_STO3G, purify="no")
    output = run(inp, propagate_wfn=True)
    assert isinstance(output.results, OptimizationData)
    logs = output.results.trajectory[-1].logs
    assert logs is not None
    assert "Initial guess will be loaded from c0" in logs
    assert output.results.energies[0] > output.results.energies[-1]


# Ammonia near its planar inversion saddle point at GFN2-xTB (bohr).
NEAR_PLANAR_AMMONIA = Structure(
    symbols=["N", "H", "H", "H"],
    geometry=np.array(
        [[0.0, 0.0, 0.01], [1.88, 0.0, 0.01], [-0.94, 1.62, 0.01], [-0.94, -1.62, 0.01]]
    ),
)


@requires("crest")
def test_transition_state_with_crest(run: Run) -> None:
    """geomeTRIC's saddle-point search on GFN2-xTB gradients finds one imaginary mode."""
    inp = _input(
        NEAR_PLANAR_AMMONIA,
        CalcType.transition_state,
        "crest",
        Model(method="gfn2"),
        threads=1,
    )
    output = run(inp)
    assert isinstance(output.results, OptimizationData)
    assert np.isclose(output.results.energies[-1], -4.4165068, atol=1e-6)
    final = output.results.trajectory[-1].input_data.structure
    hessian = ProgramInput(
        program="crest",
        calctype=CalcType.hessian,
        structure=final,
        model=Model(method="gfn2"),
        keywords={"threads": 1},
    )
    frequencies = run(hessian).results.freqs_wavenumber
    assert sum(f < 0 for f in frequencies) == 1
