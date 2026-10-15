"""TeraChem runs through qccompute. These need a CUDA GPU."""

import numpy as np
import pytest
from qcdata import CalcType, Model, OptimizationData, ProgramInput, Structure

from .conftest import Run, requires

pytestmark = [pytest.mark.integration, requires("terachem")]

HF_STO3G = Model(method="hf", basis="sto-3g")
H2_ENERGY = -1.1167143325
H2_GRADIENT = [[0.0, 0.0, -0.02845402], [0.0, 0.0, 0.02845402]]

# A water geometry whose B3LYP/6-31g SCF converges reliably in TeraChem.
WATER_NEAR_MINIMUM = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array(
        [
            [0.022531001997713594, 0.017650541231852297, -0.012249224534705398],
            [0.5009229288099285, 1.7183882034872506, 0.5180806028365682],
            [1.147442147965631, -0.49902392855765926, -1.3798296942644486],
        ]
    ),
)


def _input(
    structure: Structure,
    calctype: CalcType,
    model: Model = HF_STO3G,
    **keywords: str,
) -> ProgramInput:
    return ProgramInput(
        program="terachem",
        structure=structure,
        calctype=calctype,
        model=model,
        keywords={"purify": "no", **keywords},
    )


def test_energy(run: Run, hydrogen: Structure) -> None:
    """HF/STO-3G energy of H2."""
    inp = _input(hydrogen, CalcType.energy)
    output = run(inp, case="h2_energy")
    assert output.input_data == inp
    assert output.results.provenance.program == "terachem"
    assert np.isclose(output.results.energy, H2_ENERGY, atol=1e-6)


def test_gradient(run: Run, hydrogen: Structure) -> None:
    """HF/STO-3G gradient of H2."""
    output = run(_input(hydrogen, CalcType.gradient), case="h2_gradient")
    assert np.isclose(output.results.energy, H2_ENERGY, atol=1e-6)
    assert np.allclose(output.results.gradient, H2_GRADIENT, atol=1e-6)


def test_hessian(run: Run) -> None:
    """B3LYP/6-31g Hessian of water near its minimum."""
    inp = _input(
        WATER_NEAR_MINIMUM, CalcType.hessian, Model(method="b3lyp", basis="6-31g")
    )
    output = run(inp, case="water_b3lyp_hessian")
    hessian = np.asarray(output.results.hessian)
    assert np.isclose(output.results.energy, -76.3861099088, atol=1e-6)
    assert np.allclose(output.results.gradient, 0.0, atol=1e-4)
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-6)
    # Builds from 2023 and 2026 differ by up to 2e-4 in this row.
    assert np.allclose(hessian[0], REFERENCE_HESSIAN_ROW_0, atol=1e-3)


def test_optimization(run: Run, water: Structure) -> None:
    """TeraChem's own optimizer goes downhill."""
    inp = _input(water, CalcType.optimization, new_minimizer="yes")
    output = run(inp, case="water_optimization")
    assert isinstance(output.results, OptimizationData)
    assert len(output.results.trajectory) > 1
    assert output.results.energies[0] > output.results.energies[-1]


REFERENCE_HESSIAN_ROW_0 = [
    0.2319485298040269,
    0.04240308383589297,
    -0.195377613895862,
    -0.05131768231265779,
    -0.08413835236086714,
    -0.0002594768906270112,
    -0.1807328894945268,
    0.04171577173879733,
    0.1956319703780435,
]
