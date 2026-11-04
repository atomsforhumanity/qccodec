"""What driver codecs share (programs/_driver.py) and the Compute callback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qcdata import (
    CalcType,
    Data,
    Frame,
    Model,
    OptimizationData,
    PathData,
    PathOptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    SinglePointData,
    Structure,
)

from qccodec import PythonOutput, sequential
from qccodec.exceptions import CalculationError, EncoderError
from qccodec.programs._driver import (
    Recorder,
    check_keywords,
    child_resources,
    driver_decoder,
    outcome,
    subprograms,
)

H2 = Structure(symbols=["H", "H"], geometry=np.array([[0.0, 0, 0], [0, 0, 1.4]]))
GRADIENT = ProgramSpec(program="a", calctype=CalcType.gradient, model=Model(method="m"))
HESSIAN = GRADIENT.model_copy(update={"calctype": CalcType.hessian})
Output = ProgramOutput[ProgramInput, Data]


def _driver(*specs: ProgramSpec) -> ProgramInput:
    return ProgramInput(
        program="driver",
        calctype=CalcType.optimization,
        structure=H2,
        subprograms=list(specs),
    )


def _child(z: float = 1.4, program: str = "a") -> ProgramInput:
    structure = H2.model_copy(update={"geometry": np.array([[0.0, 0, 0], [0, 0, z]])})
    return ProgramInput.from_spec(
        GRADIENT.model_copy(update={"program": program}), structure
    )


def _output(child: ProgramInput, *, success: bool = True) -> Output:
    return ProgramOutput.model_validate(
        {
            "input_data": child,
            "success": success,
            "traceback": None if success else "failed",
            "results": SinglePointData.model_validate(
                {
                    "provenance": {"program": child.program},
                    "energy": -1.0,
                    "gradient": np.zeros((2, 3)),
                }
            ),
        }
    )


class OneAtATime:
    """A one-input compute recording each call's previous output; can fail some."""

    def __init__(self, fail: set[int] | None = None) -> None:
        self.calls: list[tuple[ProgramInput, Output | None]] = []
        self.fail = fail or set()

    def __call__(
        self, child: ProgramInput, *, previous: Output | None = None
    ) -> Output:
        self.calls.append((child, previous))
        return _output(child, success=len(self.calls) - 1 not in self.fail)


# ---------------------------------------------------------------- encoding


def test_subprograms_by_calculation_type() -> None:
    """One required subprogram and optional others; any other type is refused."""
    found = subprograms(
        _driver(GRADIENT, HESSIAN), CalcType.gradient, [CalcType.hessian]
    )
    assert found == {CalcType.gradient: GRADIENT, CalcType.hessian: HESSIAN}
    assert subprograms(_driver(GRADIENT), CalcType.gradient) == {
        CalcType.gradient: GRADIENT
    }
    for specs in [(), (HESSIAN,), (GRADIENT, HESSIAN)]:
        with pytest.raises(EncoderError, match="takes one gradient subprogram"):
            subprograms(_driver(*specs), CalcType.gradient)


def test_check_keywords_refuses_owned_and_unknown_names() -> None:
    """Owned names are refused first; unknown ones only when the program lists them."""
    check_keywords("p", {"a": 1}, owned={"b"})
    check_keywords("p", {"a": 1}, known={"a"})
    with pytest.raises(EncoderError, match="sets these keywords itself: \\['b'\\]"):
        check_keywords("p", {"b": 1}, owned={"b"}, known={"a"})
    with pytest.raises(EncoderError, match="Unknown p keywords \\['c'\\]"):
        check_keywords("p", {"a": 1, "c": 2}, known={"a"})


def test_child_resources_are_the_most_any_child_declares() -> None:
    """Threads and GPUs are maxima over the subprograms; unencodable ones count nothing."""
    orca = ProgramSpec(
        program="orca",
        calctype=CalcType.gradient,
        model=Model(method="hf", basis="sto-3g"),
        keywords={"pal": 3},
    )
    terachem = ProgramSpec(
        program="terachem",
        calctype=CalcType.hessian,
        model=Model(method="hf", basis="sto-3g"),
    )
    unknown = ProgramSpec(
        program="nope", calctype=CalcType.hessian, model=Model(method="m")
    )
    assert child_resources(_driver(orca, terachem), [orca, terachem]) == (3, 1)
    assert child_resources(_driver(orca, unknown), [orca, unknown]) == (3, 0)
    assert child_resources(_driver(unknown), [unknown]) == (1, 0)


# ----------------------------------------------------------------- running


def test_recorder_needs_a_callback() -> None:
    """Without compute or a legacy compute_child, the driver cannot run."""
    with pytest.raises(RuntimeError, match="compute_callback"):
        Recorder("driver", None)


def test_recorder_keeps_steps_and_failures_but_not_probes() -> None:
    """A probe becomes a frame only if it fails."""
    compute = OneAtATime(fail={2})
    recorder = Recorder("driver", sequential(compute))
    recorder.compute(_child(1.4))
    recorder.compute(_child(1.41), step=False)
    recorder.compute(_child(1.42), step=False)  # fails
    assert [f.structure.geometry[1, 2] for f in recorder.frames] == [1.4, 1.42]
    assert [f.success for f in recorder.frames] == [True, False]
    assert len(recorder.outputs) == 3


def test_each_child_starts_from_the_last_success_of_its_program() -> None:
    """Previous outputs are tracked per program; a failure does not replace one."""
    compute = OneAtATime(fail={1})
    recorder = Recorder("driver", None, compute)  # a runner before qccompute 0.23
    first = recorder.compute(_child(1.4))
    recorder.compute(_child(1.5))  # fails
    other = recorder.compute(_child(1.5, program="b"))
    recorder.compute(_child(1.6))
    recorder.compute(_child(1.7, program="b"))
    previous = [p for _, p in compute.calls]
    assert previous[:3] == [None, first, None]
    assert previous[3] is first
    assert previous[4] is other


def test_record_at_records_the_latest_child_at_a_geometry() -> None:
    """Drivers that learn which geometry was a step afterwards record it by geometry."""
    recorder = Recorder("driver", sequential(OneAtATime()))
    recorder.compute(_child(1.4), step=False)
    later = recorder.compute(_child(1.4), step=False)
    recorder.compute(_child(1.5), step=False)
    recorder.record_at(_child(1.4).require_structure().geometry)
    recorder.record_at(np.zeros(6))  # computed nowhere: nothing recorded
    assert len(recorder.frames) == 1
    assert recorder.frames[0] == Frame.from_output(later)


def test_images_start_from_their_own_previous_outputs() -> None:
    """Each image gets the same image's last successful output; none at first."""
    seen: list[Any] = []

    def compute(inputs: Any, /, *, previous: Any = None) -> Any:
        seen.append(list(previous))
        return [
            _output(c, success=i != 1 or len(seen) > 1) for i, c in enumerate(inputs)
        ]

    recorder = Recorder("driver", compute)  # type: ignore[arg-type]
    first = recorder.compute_images([_child(1.4), _child(1.5)])
    second = recorder.compute_images([_child(1.41), _child(1.51)])
    recorder.compute_images([_child(1.42), _child(1.52)])
    recorder.compute_images([_child(1.43)])  # a different chain starts afresh
    assert seen[0] == [None, None]
    assert seen[1] == [first[0], None]  # image 1 failed the first time
    assert seen[2] == second
    assert seen[3] == [None]
    assert recorder.frames == []


def test_hessians_attach_to_the_last_frame_at_their_geometry() -> None:
    """Each Hessian goes on the last frame computed at its geometry, if any."""
    recorder = Recorder("driver", sequential(OneAtATime()))
    for z in (1.4, 1.35, 1.3):
        recorder.compute(_child(z))
    first, last = np.eye(6), 2 * np.eye(6)
    recorder.add_hessian(np.array([0, 0, 0, 0, 0, 1.4]), first)
    recorder.add_hessian(np.array([[0, 0, 0], [0, 0, 1.3]]), last)
    recorder.add_hessian(np.array([0, 0, 0, 0, 0, 9.9]), 3 * np.eye(6))
    recorder.attach_hessians(recorder.frames)
    hessians = [frame.results.hessian for frame in recorder.frames]
    assert hessians[0] is not None and np.array_equal(hessians[0], first)
    assert hessians[1] is None
    assert hessians[2] is not None and np.array_equal(hessians[2], last)


def test_sequential_computes_lists_in_order() -> None:
    """One input returns one output; a list returns a list, with previous per input."""
    compute = OneAtATime()
    run = sequential(compute)
    single = run(_child(1.4))
    outputs = run([_child(1.5), _child(1.6)], previous=[single, None])
    assert [o.input_data.require_structure().geometry[1, 2] for o in outputs] == [
        1.5,
        1.6,
    ]
    assert [p for _, p in compute.calls] == [None, single, None]
    with pytest.raises(ValueError, match="2 inputs"):
        run([_child(1.5), _child(1.6)], previous=[None])


# ---------------------------------------------------------------- decoding


@pytest.mark.parametrize(
    ("calctype", "data_type"),
    [
        (CalcType.optimization, OptimizationData),
        (CalcType.transition_state, OptimizationData),
        (CalcType.irc, PathData),
        (CalcType.path_optimization, PathOptimizationData),
    ],
)
def test_driver_decoder_builds_the_calculation_s_data(
    calctype: CalcType, data_type: type
) -> None:
    """Frames, a path, or paths, by calculation type; an error raises with the data."""
    frame = Frame.from_output(_output(_child()))
    value = {
        **outcome("1.0"),
        "frames": [frame],
        "ts_index": 0,
        "paths": [([frame, frame], 1)],
    }
    if calctype == CalcType.path_optimization:
        inp = ProgramInput(
            program="driver",
            calctype=calctype,
            structures={"initial_chain": [H2, H2]},
            subprograms=[GRADIENT],
        )
    else:
        inp = _driver(GRADIENT).model_copy(update={"calctype": calctype})
    decode = driver_decoder("driver")
    data = decode(inp, PythonOutput(value=value))
    assert type(data) is data_type
    assert data.provenance.program == "driver"
    assert data.provenance.program_version == "1.0"
    with pytest.raises(CalculationError, match="stopped") as raised:
        decode(inp, PythonOutput(value={**value, "error": "stopped"}))
    assert type(raised.value.data) is data_type


def test_driver_decoder_reports_a_function_that_raised() -> None:
    """With no outcome at all, the function's exception is the failure."""
    decode = driver_decoder("driver")
    with pytest.raises(CalculationError, match="RuntimeError: no callback") as raised:
        decode(_driver(GRADIENT), PythonOutput(error=RuntimeError("no callback")))
    assert isinstance(raised.value.data, OptimizationData)
    assert raised.value.data.frames == []
