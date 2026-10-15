"""geomeTRIC optimizations driving a real gradient program through qccompute."""

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


@requires("terachem")
def test_transition_state_with_terachem(run: Run, water: Structure) -> None:
    """A transition-state search on water completes with a full trajectory."""
    inp = _input(water, CalcType.transition_state, "terachem", HF_STO3G, purify="no")
    output = run(inp, propagate_wfn=True)
    assert isinstance(output.results, OptimizationData)
    assert len(output.results.trajectory) > 1
    logs = output.results.trajectory[-1].logs
    assert logs is not None
    assert "Initial guess will be loaded from c0" in logs
