"""The sella codec and qccodec.ase.ProgramCalculator, with an analytic stand-in child.

The child is a set of harmonic springs between every pair of atoms, so the minimum is
known exactly and no program has to be installed.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pytest
from qcdata import (
    CalcType,
    Data,
    Frame,
    Model,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    SinglePointData,
    Structure,
)

from qccodec import get_codec, sequential
from qccodec.exceptions import CalculationError, EncoderError
from qccodec.native import PythonInput, PythonOutput

# Spring rest lengths (bohr): O-H 1.81, H-H 2.86, a water-like triangle.
REST = {(0, 1): 1.81, (0, 2): 1.81, (1, 2): 2.86}
K = 0.5  # Hartree/bohr^2
WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array([[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]),
)
GRADIENT = ProgramSpec(
    program="springs", calctype=CalcType.gradient, model=Model(method="k")
)
HESSIAN = ProgramSpec(
    program="springs", calctype=CalcType.hessian, model=Model(method="k")
)


def springs(geometry: np.ndarray) -> tuple[float, np.ndarray]:
    """Energy (Hartree) and gradient (Hartree/bohr) of the springs."""
    energy, gradient = 0.0, np.zeros_like(geometry)
    for (i, j), rest in REST.items():
        d = geometry[i] - geometry[j]
        r = float(np.linalg.norm(d))
        energy += K * (r - rest) ** 2
        g = 2 * K * (r - rest) * d / r
        gradient[i] += g
        gradient[j] -= g
    return energy, gradient


def springs_hessian(geometry: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central-difference Hessian of the springs (Hartree/bohr^2)."""
    x = geometry.flatten()
    columns = []
    for k in range(x.size):
        step = np.zeros_like(x)
        step[k] = h
        plus = springs((x + step).reshape(-1, 3))[1].flatten()
        minus = springs((x - step).reshape(-1, 3))[1].flatten()
        columns.append((plus - minus) / (2 * h))
    return np.array(columns)


class Child:
    """A one-input compute stand-in that records its calls and can fail on one."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.inputs: list[ProgramInput] = []
        self.previous: list[ProgramOutput[ProgramInput, Data] | None] = []
        self.fail_at = fail_at

    def __call__(
        self,
        input_data: ProgramInput,
        *,
        previous: ProgramOutput[ProgramInput, Data] | None = None,
    ) -> ProgramOutput[ProgramInput, Data]:
        self.inputs.append(input_data)
        self.previous.append(previous)
        geometry = input_data.require_structure().geometry
        energy, gradient = springs(geometry)
        results: dict[str, Any] = {
            "provenance": {"program": "springs"},
            "energy": energy,
            "gradient": gradient,
        }
        if input_data.calctype == CalcType.hessian:
            results["hessian"] = springs_hessian(geometry)
        failed = self.fail_at is not None and len(self.inputs) - 1 == self.fail_at
        return ProgramOutput.model_validate(
            {
                "input_data": input_data,
                "success": not failed,
                "traceback": "springs failed" if failed else None,
                "results": SinglePointData.model_validate(results),
            }
        )


def _input(calctype: str = "optimization", **keywords: Any) -> ProgramInput:
    subprograms = keywords.pop("subprograms", [GRADIENT])
    return ProgramInput(
        program="sella",
        calctype=CalcType(calctype),
        structure=WATER,
        subprograms=subprograms,
        keywords=keywords,
    )


def _run(inp: ProgramInput, child: Child) -> OptimizationData:
    """Run the encoded invocation with the stand-in child and decode it."""
    native = get_codec("sella").encode(inp)
    assert isinstance(native, PythonInput)
    value = native.function(*native.args, **native.kwargs, compute=sequential(child))
    result = get_codec("sella").decode(inp, PythonOutput(value=value))
    assert isinstance(result, OptimizationData)
    return result


def _distances(structure: Structure | None) -> dict[tuple[int, int], float]:
    assert structure is not None
    g = structure.geometry
    return {
        (i, j): float(np.linalg.norm(g[i] - g[j])) for i, j in combinations(range(3), 2)
    }


def test_encode_splits_run_and_constructor_keywords() -> None:
    """fmax and steps go to Sella.run, the rest to Sella(); order follows the calctype."""
    native = get_codec("sella").encode(
        _input("transition_state", fmax=0.01, steps=5, eta=0.001, internal=True)
    )
    assert isinstance(native, PythonInput)
    assert native.kwargs["run_kwargs"] == {"fmax": 0.01, "steps": 5}
    assert native.kwargs["init_kwargs"] == {"eta": 0.001, "internal": True}
    assert native.kwargs["order"] == 1 and native.kwargs["hessian_spec"] is None
    minimization = get_codec("sella").encode(_input())
    assert isinstance(minimization, PythonInput)
    assert minimization.kwargs["order"] == 0
    assert native.compute_callback == "compute"
    assert native.child_callback == "compute_child"  # runners before qccompute 0.23


@pytest.mark.parametrize(
    ("inp", "match"),
    [
        (_input(order=1), "sets these keywords"),
        (_input(logfile="x"), "sets these keywords"),
        (_input(subprograms=[]), "gradient"),
        (
            _input(
                subprograms=[
                    GRADIENT,
                    GRADIENT.model_copy(update={"calctype": CalcType.energy}),
                ]
            ),
            "optionally one hessian",
        ),
        (_input(constraints={"fix_cartesian": [0]}), "Unknown Sella constraint"),
        (_input(constraints={"fix_bond": [[0, 3]]}), "0-based"),
        (_input(constraints={"fix_bond": [0, 1]}), "0-based"),
        (_input(constraints=[0]), "must map"),
    ],
)
def test_encode_rejects(inp: ProgramInput, match: str) -> None:
    """Owned keywords, bad subprograms, and malformed constraints are refused."""
    with pytest.raises(EncoderError, match=match):
        get_codec("sella").encode(inp)


def test_encode_rejects_command_line_arguments() -> None:
    """Sella's Python interface takes no command-line arguments."""
    inp = _input().model_copy(update={"cmdline_args": ["-x"]})
    with pytest.raises(EncoderError, match="command-line"):
        get_codec("sella").encode(inp)


def test_minimization_reaches_the_springs_minimum() -> None:
    """Each step is a recorded child; the end is the springs' rest triangle."""
    pytest.importorskip("sella")
    child = Child()
    result = _run(_input(fmax=1e-4, steps=100), child)
    assert result.provenance.program == "sella"
    assert result.provenance.program_version
    assert len(result.frames) > 2
    assert result.energies[0] > result.energies[-1]
    assert result.energies[-1] < 1e-8
    for pair, rest in REST.items():
        assert _distances(result.final_structure)[pair] == pytest.approx(rest, abs=1e-3)
    # Sella removes rigid motion, so the frame stays the input's.
    assert result.final_structure is not None
    assert np.allclose(
        result.final_structure.geometry.mean(axis=0), WATER.geometry.mean(axis=0)
    )
    assert all(frame.results.provenance.program == "springs" for frame in result.frames)
    assert np.allclose(result.frames[0].structure.geometry, WATER.geometry)


def test_running_out_of_steps_fails_with_the_trajectory() -> None:
    """An unconverged run is a failure, not a result."""
    pytest.importorskip("sella")
    with pytest.raises(CalculationError, match="did not converge") as raised:
        _run(_input(fmax=1e-6, steps=1), Child())
    assert isinstance(raised.value.data, OptimizationData)
    assert len(raised.value.data.frames) >= 1


def test_a_failed_child_ends_the_trajectory() -> None:
    """The failed calculation is kept as the last frame."""
    pytest.importorskip("sella")
    with pytest.raises(CalculationError) as raised:
        _run(_input(fmax=1e-6, steps=50), Child(fail_at=2))
    assert isinstance(raised.value.data, OptimizationData)
    trajectory = raised.value.data.frames
    assert not trajectory[-1].success
    assert all(frame.success for frame in trajectory[:-1])


def test_hessian_subprogram_starts_sella_and_is_attached() -> None:
    """The Hessian child becomes Sella's hessian_function; its frame keeps the Hessian."""
    pytest.importorskip("sella")
    child = Child()
    result = _run(
        _input(fmax=1e-4, steps=100, eig=True, subprograms=[GRADIENT, HESSIAN]), child
    )
    assert [i.calctype for i in child.inputs].count(CalcType.hessian) >= 1
    assert result.frames[0].results.hessian is not None
    assert result.energies[-1] < 1e-8


def test_constraints_hold() -> None:
    """A fixed bond keeps its target length while the rest relaxes."""
    pytest.importorskip("sella")
    result = _run(
        _input(
            fmax=1e-4,
            steps=100,
            constraints={"fix_bond": [{"indices": [0, 1], "target": 1.2}]},
        ),
        Child(),
    )
    target = 1.2 * 1.8897261246  # angstrom to bohr
    assert _distances(result.final_structure)[(0, 1)] == pytest.approx(target, abs=1e-4)


def test_children_receive_the_previous_output() -> None:
    """Each child after the first gets the last successful output to restart from."""
    pytest.importorskip("sella")
    child = Child()
    _run(_input(fmax=1e-3, steps=20), child)
    assert child.previous[0] is None
    assert all(p is not None for p in child.previous[1:])


def test_runners_before_qccompute_0_23_still_work() -> None:
    """Their one-input compute_child computes the children when compute is missing."""
    pytest.importorskip("sella")
    inp = _input(fmax=1e-4, steps=100)
    native = get_codec("sella").encode(inp)
    assert isinstance(native, PythonInput)
    value = native.function(**native.kwargs, compute_child=Child())
    result = get_codec("sella").decode(inp, PythonOutput(value=value))
    assert isinstance(result, OptimizationData)
    assert result.energies[-1] < 1e-8


def test_a_runner_without_a_compute_callback_fails_clearly() -> None:
    """The run fails, naming the callback the runner did not provide."""
    pytest.importorskip("sella")
    inp = _input()
    native = get_codec("sella").encode(inp)
    assert isinstance(native, PythonInput)
    with pytest.raises(RuntimeError, match="compute_callback") as raised:
        native.function(**native.kwargs)
    with pytest.raises(CalculationError, match="compute_callback"):
        get_codec("sella").decode(inp, PythonOutput(error=raised.value))


def test_calculator_converts_units_and_records_outputs() -> None:
    """Energies in eV and forces in eV/angstrom, from Hartree and Hartree/bohr."""
    ase = pytest.importorskip("ase")
    from qcref import constants as c

    from qccodec.ase import ProgramCalculator

    child = Child()
    atoms = ase.Atoms("OHH", positions=WATER.geometry * c.BOHR_TO_ANGSTROM)
    atoms.calc = ProgramCalculator(GRADIENT, child, charge=0, multiplicity=1)
    energy, gradient = springs(WATER.geometry)
    assert atoms.get_potential_energy() == pytest.approx(energy * c.HARTREE_TO_EV)
    assert np.allclose(
        atoms.get_forces(), -gradient * c.HARTREE_TO_EV * c.ANGSTROM_TO_BOHR
    )
    assert len(atoms.calc.outputs) == 1  # ASE caches by geometry
    assert np.allclose(child.inputs[0].require_structure().geometry, WATER.geometry)


def test_calculator_raises_with_the_failed_output() -> None:
    """A failed calculation raises ASE's CalculationFailed carrying the output."""
    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import CalculationFailed

    from qccodec.ase import ProgramCalculator

    atoms = ase.Atoms("OHH", positions=WATER.geometry * 0.529)
    atoms.calc = ProgramCalculator(GRADIENT, Child(fail_at=0))
    with pytest.raises(CalculationFailed) as raised:
        atoms.get_forces()
    assert not raised.value.output.success  # type: ignore[attr-defined]


def test_calculator_checks_its_spec_and_structure() -> None:
    """Only energy and gradient specs; a template's elements must match the atoms'."""
    ase = pytest.importorskip("ase")
    from qccodec.ase import ProgramCalculator

    with pytest.raises(ValueError, match="energy or gradient"):
        ProgramCalculator(HESSIAN, Child())
    atoms = ase.Atoms("OHH", positions=WATER.geometry)
    atoms.calc = ProgramCalculator(
        GRADIENT,
        Child(),
        structure=Structure(symbols=["N", "H", "H"], geometry=WATER.geometry),
    )
    with pytest.raises(ValueError, match="do not match"):
        atoms.get_potential_energy()


def test_declares_the_resources_of_its_gradient_child() -> None:
    """Lists of Sella runs share the machine by what the child declares."""
    child = ProgramSpec(
        program="orca",
        calctype=CalcType.gradient,
        model=Model(method="hf", basis="sto-3g"),
        keywords={"pal": 4},
    )
    native = get_codec("sella").encode(_input(subprograms=[child]))
    assert native.threads == 4


def test_declares_the_most_either_child_declares() -> None:
    """A Hessian child that needs more threads than the gradient child sets the total."""
    gradient = ProgramSpec(
        program="orca",
        calctype=CalcType.gradient,
        model=Model(method="hf", basis="sto-3g"),
        keywords={"pal": 2},
    )
    hessian = gradient.model_copy(
        update={"calctype": CalcType.hessian, "keywords": {"pal": 6}}
    )
    native = get_codec("sella").encode(_input(subprograms=[gradient, hessian]))
    assert native.threads == 6


def _irc(**keywords: Any) -> ProgramInput:
    subprograms = keywords.pop("subprograms", [GRADIENT])
    return ProgramInput(
        program="sella",
        calctype=CalcType.irc,
        structure=WATER,
        subprograms=subprograms,
        keywords=keywords,
    )


def test_irc_encoding_routes_its_run_keywords() -> None:
    """fmax_inner goes to IRC.run for an IRC; direction belongs to the codec."""
    native = get_codec("sella").encode(_irc(fmax=0.01, fmax_inner=0.005, dx=0.05))
    assert isinstance(native, PythonInput)
    assert native.kwargs["irc"] is True
    assert native.kwargs["run_kwargs"] == {"fmax": 0.01, "fmax_inner": 0.005}
    assert native.kwargs["init_kwargs"] == {"dx": 0.05}
    with pytest.raises(EncoderError, match="sets these keywords"):
        get_codec("sella").encode(_irc(direction="forward"))


def _frame(geometry: np.ndarray, energy: float, gradient: np.ndarray) -> Frame:
    return Frame(
        structure=WATER.model_copy(update={"geometry": geometry}),
        success=True,
        results=SinglePointData(
            provenance=Provenance(program="springs"), energy=energy, gradient=gradient
        ),
    )


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_irc_branches_join_through_the_transition_state() -> None:
    """Reverse branch reversed, the transition state once, then the forward branch."""
    from qccodec.programs.sella import _irc_path

    g = WATER.geometry
    grad = np.zeros_like(g)
    ts = _frame(g, 0.0, grad)
    reverse = [ts, _frame(g, -1.0, grad), _frame(g, -2.0, grad)]
    forward = [ts, _frame(g, -3.0, grad)]
    path, ts_index = _irc_path({"reverse": reverse, "forward": forward})
    assert [f.results.energy for f in path] == [-2.0, -1.0, 0.0, -3.0]
    assert ts_index == 2
    path, ts_index = _irc_path({"reverse": [ts]})  # failed before the forward branch
    assert len(path) == 1 and ts_index == 0
    assert _irc_path({}) == ([], None)


def test_irc_branches_lose_their_rigid_drift() -> None:
    """Frames that are rotated, shifted copies align back onto the transition state.

    Alignment works outward from the transition state on both sides, rotating each
    frame's gradient and Hessian with it.
    """
    from qccodec.programs._driver import align_path

    g = WATER.geometry
    grad = np.arange(9.0).reshape(3, 3)
    turn = _rotation(0.3)
    drifted = (g - g.mean(axis=0)) @ turn.T + np.array([1.0, -2.0, 0.5])
    block = np.kron(np.eye(3), turn)
    hessian = np.diag(np.arange(1.0, 10.0))
    copy = _frame(drifted, -1.0, grad @ turn.T)
    copy = copy.model_copy(
        update={
            "results": copy.results.model_copy(
                update={"hessian": block @ hessian @ block.T}
            )
        }
    )
    path = align_path([copy, _frame(g, 0.0, grad), copy], ts_index=1)
    for end in (path[0], path[2]):
        assert np.allclose(end.structure.geometry, g, atol=1e-10)
        assert end.results.gradient is not None and end.results.hessian is not None
        assert np.allclose(end.results.gradient, grad, atol=1e-10)
        assert np.allclose(end.results.hessian, hessian, atol=1e-10)
        assert end.results.energy == -1.0
    assert np.array_equal(path[1].structure.geometry, g)  # the transition state stays
