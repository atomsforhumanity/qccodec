"""ORCA runs through qccompute. ORCA is not redistributable; these run where installed."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from qcdata import (
    CalcType,
    Model,
    OptimizationData,
    ProgramError,
    ProgramInput,
    Structure,
)

from qccodec.testing import check_normal_modes

from .conftest import Run, required_programs


def _orca_installed() -> bool:
    """Whether ``orca`` is the ORCA program, not the GNOME screen reader of that name."""
    path = shutil.which("orca")
    return path is not None and (Path(path).resolve().parent / "orca_scfgrad").exists()


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _orca_installed() and "orca" not in required_programs(),
        reason="ORCA is not on PATH",
    ),
]

# HF/STO-3G on the shared distorted water, measured with ORCA 6.0.0.
ENERGY = -74.964413887145
GRADIENT = [
    [0.032074289, 0.023483301, -0.016945752],
    [-0.013215167, -0.021068347, 0.000366205],
    [-0.018859122, -0.002414954, 0.016579547],
]


def _input(structure: Structure, calctype: CalcType) -> ProgramInput:
    return ProgramInput(
        program="orca",
        structure=structure,
        calctype=calctype,
        model=Model(method="hf", basis="sto-3g"),
    )


def test_energy(run: Run, water: Structure) -> None:
    """HF/STO-3G energy."""
    output = run(_input(water, CalcType.energy), case="water_energy")
    assert output.results.provenance.program == "orca"
    assert np.isclose(output.results.energy, ENERGY, atol=1e-6)


def test_gradient(run: Run, water: Structure) -> None:
    """HF/STO-3G gradient."""
    output = run(_input(water, CalcType.gradient), case="water_gradient")
    assert np.isclose(output.results.energy, ENERGY, atol=1e-6)
    assert np.allclose(output.results.gradient, GRADIENT, atol=1e-6)


def test_hessian(run: Run, water: Structure) -> None:
    """The analytic HF Hessian, harmonic frequencies, and normal modes."""
    output = run(_input(water, CalcType.hessian), case="water_hessian")
    results = output.results
    hessian = np.asarray(results.hessian)
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-6)
    assert np.isclose(results.energy, ENERGY, atol=1e-6)
    # Translations and rotations are left out; the distorted water is not a
    # minimum, but HF/STO-3G still gives three real modes here.
    assert len(results.vibrational_analysis.frequencies) == 3
    check_normal_modes(results, list(water.symbols))


def test_hessian_modes_across_blocks(run: Run) -> None:
    """Ammonia's six modes span two column blocks of the .hess file."""
    ammonia = Structure(
        symbols=["N", "H", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.2],
                [1.77, 0.0, -0.45],
                [-0.885, 1.533, -0.45],
                [-0.885, -1.533, -0.45],
            ]
        ),
    )
    output = run(_input(ammonia, CalcType.hessian), case="ammonia_hessian")
    assert len(output.results.vibrational_analysis.frequencies) == 6
    check_normal_modes(output.results, list(ammonia.symbols))


def test_optimization(run: Run, water: Structure) -> None:
    """ORCA's optimizer reaches the HF/STO-3G minimum."""
    output = run(_input(water, CalcType.optimization), case="water_optimization")
    assert isinstance(output.results, OptimizationData)
    energies = output.results.energies
    assert len(energies) > 1
    assert np.isclose(energies[-1], -74.965901192106, atol=1e-6)
    assert all(np.diff(energies) <= 1e-8)


@pytest.mark.parametrize(
    ("case", "model", "keywords", "code"),
    [
        ("unknown_keyword", None, {"tightscff": True}, "unknown_key"),
        ("unknown_block", None, {"scff": {"maxiter": 100}}, "unknown_key"),
        ("unknown_block_setting", None, {"scf": {"maxiterr": 100}}, "unknown_key"),
        ("unknown_basis", Model(method="hf", basis="sto-3gg"), {}, None),
    ],
)
def test_orca_refuses_unknown_input(
    run: Run,
    water: Structure,
    case: str,
    model: Model | None,
    keywords: dict[str, object],
    code: str | None,
) -> None:
    """ORCA refuses what it does not know before calculating; kind input."""
    inp = _input(water, CalcType.energy).model_copy(
        update={"keywords": keywords, **({"model": model} if model else {})}
    )
    with pytest.raises(ProgramError) as e:
        run(inp, case=case)
    assert (e.value.error.kind, e.value.error.code) == ("input", code)


def test_scf_failure_is_convergence(run: Run, water: Structure) -> None:
    """ORCA exits 0 when the SCF fails; the codec reports convergence."""
    inp = _input(water, CalcType.energy).model_copy(
        update={"keywords": {"scf": {"maxiter": 2}}}
    )
    with pytest.raises(ProgramError) as e:
        run(inp, case="scf_not_converged")
    assert (e.value.error.kind, e.value.error.code) == (
        "convergence",
        "scf_not_converged",
    )


def test_optimization_out_of_cycles_fails(run: Run, water: Structure) -> None:
    """ORCA calls an optimization out of cycles a normal termination; it is not."""
    inp = _input(water, CalcType.optimization).model_copy(
        update={"keywords": {"geom": {"maxiter": 2}}}
    )
    with pytest.raises(ProgramError) as e:
        run(inp, case="optimization_not_converged")
    assert (e.value.error.kind, e.value.error.code) == ("convergence", "max_iter")
    assert e.value.output is not None
    assert len(e.value.output.results.frames) == 2


def test_impossible_multiplicity(run: Run, water: Structure) -> None:
    """A multiplicity the electron count cannot have is kind input."""
    doublet = water.model_copy(update={"multiplicity": 2})
    with pytest.raises(ProgramError) as e:
        run(_input(doublet, CalcType.energy), case="impossible_multiplicity")
    assert e.value.error.kind == "input"


@pytest.mark.parametrize(
    ("case", "method", "maxcore", "figures"),
    [
        # The SCF estimates its need before it starts.
        ("scf_out_of_memory", "RI-MP2 def2-TZVP def2-TZVP/C", 1, True),
        # DLPNO prints its minimum against %maxcore.
        ("dlpno_out_of_memory", "DLPNO-CCSD(T) def2-TZVP def2-TZVP/C", 5, True),
        # Batching runs out with no figure to report.
        ("mdci_out_of_memory", "CCSD(T) def2-TZVP", 5, False),
    ],
)
def test_out_of_memory_is_host_memory(
    run: Run, water: Structure, case: str, method: str, maxcore: int, figures: bool
) -> None:
    """With what it needed against %maxcore where ORCA prints both, in GiB."""
    inp = ProgramInput(
        program="orca",
        structure=water,
        calctype=CalcType.energy,
        model=Model(method=method),
        keywords={"maxcore": maxcore},
    )
    with pytest.raises(ProgramError) as e:
        run(inp, case=case)
    assert (e.value.error.kind, e.value.error.code) == ("resource", "host_memory")
    details = e.value.error.details
    if figures:
        assert details.keys() == {"required_gib", "available_gib"}
        assert details["available_gib"] == pytest.approx(maxcore / 1024, rel=1e-3)
        assert details["required_gib"] > details["available_gib"]
    else:
        assert details == {}
