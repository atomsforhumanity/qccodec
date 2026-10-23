"""Sella and ProgramCalculator driving real xtb calculations through qccompute.

These test the interface (child calculations, Hessian subprograms, trajectories, the
ASE calculator), not the optimizer.
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
    pytest.mark.skipif(find_spec("sella") is None, reason="sella not installed"),
    requires("xtb"),
]

GFN2 = Model(method="gfn2")
GRADIENT = ProgramSpec(program="xtb", calctype=CalcType.gradient, model=GFN2)
HESSIAN = ProgramSpec(program="xtb", calctype=CalcType.hessian, model=GFN2)
# GFN2-xTB water minimum, as geomeTRIC and CREST find it (test_geometric.py).
WATER_MINIMUM = -5.0705443538
# Bent HCN, between HCN and HNC (bohr).
HCN = Structure(
    symbols=["C", "N", "H"],
    geometry=np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [1.1, 1.9, 0.0]]),
)


def test_minimization(run: Run, water: Structure) -> None:
    """Sella reaches the GFN2-xTB minimum; every step is an xtb gradient."""
    output = run(
        ProgramInput(
            program="sella",
            calctype=CalcType.optimization,
            structure=water,
            subprograms=[GRADIENT],
            keywords={"fmax": 0.001, "steps": 100},
        )
    )
    assert isinstance(output.results, OptimizationData)
    assert np.isclose(output.results.energies[-1], WATER_MINIMUM, atol=1e-5)
    assert all(f.input_data.program == "xtb" for f in output.results.trajectory)
    assert np.allclose(
        output.results.trajectory[0].input_data.structure.geometry, water.geometry
    )


def test_transition_state_from_an_xtb_hessian(run: Run) -> None:
    """Starting from xtb's Hessian, Sella finds a saddle with one imaginary mode."""
    output = run(
        ProgramInput(
            program="sella",
            calctype=CalcType.transition_state,
            structure=HCN,
            subprograms=[GRADIENT, HESSIAN],
            keywords={"fmax": 0.001, "steps": 200},
        )
    )
    results = output.results
    assert isinstance(results, OptimizationData)
    assert results.trajectory[0].results.hessian is not None
    assert results.final_structure is not None
    check = run(ProgramInput.from_spec(HESSIAN, results.final_structure))
    frequencies = check.results.freqs_wavenumber
    assert sum(f < -100 for f in frequencies) == 1


def test_program_calculator_drives_an_ase_optimizer(water: Structure) -> None:
    """ASE's BFGS with ProgramCalculator and qccompute reaches the same minimum."""
    from ase import Atoms
    from ase.optimize import BFGS
    from qccompute import compute
    from qcref import constants as c

    from qccodec.ase import ProgramCalculator

    atoms = Atoms(
        symbols=list(water.symbols), positions=water.geometry * c.BOHR_TO_ANGSTROM
    )
    atoms.calc = ProgramCalculator(GRADIENT, compute)
    assert BFGS(atoms, logfile=None).run(fmax=0.001, steps=100)
    energy = atoms.get_potential_energy() * c.EV_TO_HARTREE
    assert np.isclose(energy, WATER_MINIMUM, atol=1e-5)
    assert all(o.success for o in atoms.calc.outputs)
