from __future__ import annotations

import json
import sys
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qcdata import (
    CalcType,
    Data,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
)

from qccodec import (
    Codec,
    CodecDefinition,
    PythonInput,
    PythonOutput,
    SubprocessInput,
    SubprocessOutput,
    get_codec,
)
from qccodec.exceptions import CalculationError, EncoderError, MatchNotFoundError


@pytest.mark.parametrize(
    "program,filename",
    [("terachem", "tc.in"), ("orca", "orca.inp"), ("crest", "input.toml")],
)
def test_complete_subprocess_invocation(
    prog_input_factory: Callable[..., ProgramInput], program: str, filename: str
) -> None:
    """Complete subprocess invocation."""
    inp = prog_input_factory("energy", program)
    inp = ProgramInput.model_validate(
        {**inp.model_dump(), "cmdline_args": ["--extra"], "files": {"restart": b"data"}}
    )
    native = get_codec(program).encode(inp)
    assert isinstance(native, SubprocessInput)
    assert native.command == (program, filename, "--extra")
    assert native.files[filename] and native.files["restart"] == b"data"
    conflict = ProgramInput.model_validate(
        {**inp.model_dump(), "files": {filename: "override"}}
    )
    with pytest.raises(EncoderError, match="conflict"):
        get_codec(program).encode(conflict)


def test_decorated_parsers_are_local_and_optional(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Decorated parsers are local and optional."""
    codecs = [
        CodecDefinition(
            "local", iter_files=lambda stdout, directory: [("stdout", stdout)]
        )
        for _ in range(2)
    ]

    @codecs[0].parser(filetype="stdout", calctypes=[CalcType.energy], target="energy")
    def energy(contents: str) -> float:
        """Parse a scalar directly from the supplied native contents."""
        return float(contents)

    @codecs[0].parser(filetype="stdout", target="gradient", required=False)
    def gradient(contents: str) -> list[list[float]]:
        """Simulate an optional scientific value missing from native output."""
        raise MatchNotFoundError("gradient", contents)

    assert energy("-1") == -1 and codecs[1].parsers == []
    data = codecs[0].decode_artifacts("energy", stdout="-1")
    assert isinstance(data, SinglePointData)
    assert data.energy == -1 and data.gradient is None


def test_dictionary_target_is_respected() -> None:
    """Dictionary target is respected."""
    codec = CodecDefinition(
        "local", iter_files=lambda stdout, directory: [("stdout", stdout)]
    )

    @codec.parser(filetype="stdout", target="wavefunction")
    def orbitals(contents: str) -> dict[str, list[float]]:
        """Return native orbital data to exercise explicit dictionary targets."""
        return {"scf_eigenvalues_a": [-1]}

    data = codec.decode_artifacts("energy", stdout="dummy")
    assert isinstance(data, SinglePointData)
    assert data.wavefunction is not None
    assert data.wavefunction.scf_eigenvalues_a is not None
    assert data.wavefunction.scf_eigenvalues_a.tolist() == [-1]


def test_crest_zero_exit_failure_preserves_energy(
    prog_input_factory: Callable[..., ProgramInput], tmp_path: Path
) -> None:
    """Crest zero exit failure preserves energy."""
    (tmp_path / "crest.engrad").write_text("# Energy ( Eh )\n#\n -1.0\n")
    with pytest.raises(CalculationError) as raised:
        get_codec("crest").decode(
            prog_input_factory("gradient", "crest"),
            SubprocessOutput(
                stdout="Version 3.0.2, FAILED", directory=tmp_path, returncode=0
            ),
        )
    assert isinstance(raised.value.data, SinglePointData)
    assert raised.value.data.energy == -1 and raised.value.data.gradient is None


def test_xtb_encode_and_decode_do_not_import_xtb(
    prog_input_factory: Callable[..., ProgramInput],
    monkeypatch: pytest.MonkeyPatch,
    test_data_dir: Path,
) -> None:
    """Xtb encode and decode do not import xtb."""
    monkeypatch.setitem(sys.modules, "xtb", None)
    codec = get_codec("xtb")
    inp = prog_input_factory("gradient", "xtb")
    native = codec.encode(inp)
    assert isinstance(native, PythonInput)
    assert native.kwargs["uhf"] == inp.structure.multiplicity - 1
    assert native.kwargs["numbers"] == list(inp.structure.atomic_numbers)
    snapshot = json.loads((test_data_dir / "xtb/native_snapshot.json").read_text())
    data = codec.decode(inp, PythonOutput(value=snapshot))
    assert isinstance(data, SinglePointData)
    assert data.gradient is not None
    assert data.energy == -1 and data.gradient.tolist() == [[0.1, 0.2, 0.3]]
    assert data.provenance.program_version == "fixture"


@pytest.mark.parametrize("fail", [False, True])
def test_xtb_native_sequence_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Xtb native sequence and cleanup."""
    from qccodec.codecs.xtb import invoke_xtb

    calls: list[tuple[object, ...]] = []

    class Param(Enum):
        GFN2xTB = 1

    class Calculator:
        def __init__(self, *args: object) -> None:
            """Initialize the fake native object used by the binding test."""
            calls.append(("constructor", args))

        def set_verbosity(self, value: float) -> None:
            """Record the requested native logging level."""
            calls.append(("verbosity", value))

        def set_output(self, filename: str) -> None:
            """Record the native output-unit binding."""
            calls.append(("output", filename))

        def set_accuracy(self, value: float) -> None:
            """Record a translated native calculator setting."""
            calls.append(("accuracy", value))

        def singlepoint(self) -> SimpleNamespace:
            """Return native getter functions or simulate SCF failure."""
            calls.append(("singlepoint",))
            if fail:
                raise RuntimeError("SCF failed")
            return SimpleNamespace(
                get_energy=lambda: -1,
                get_gradient=lambda: [[0, 0, 0]],
                get_dipole=lambda: [0, 0, 0],
                get_orbital_eigenvalues=lambda: [-0.5],
            )

        def release_output(self) -> None:
            """Record cleanup even when native invocation fails."""
            calls.append(("release",))

    monkeypatch.setitem(sys.modules, "xtb", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "xtb.interface",
        SimpleNamespace(Calculator=Calculator, Param=Param),
    )
    monkeypatch.setitem(sys.modules, "xtb.libxtb", SimpleNamespace(VERBOSITY_FULL=2))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0")
    kwargs: dict[str, Any] = dict(
        method="GFN2xTB",
        numbers=[1],
        positions=np.array([[0.0, 0.0, 0.0]]),
        charge=0,
        uhf=0,
        settings={"accuracy": 0.1},
    )
    if fail:
        with pytest.raises(RuntimeError):
            invoke_xtb(**kwargs)
    else:
        assert invoke_xtb(**kwargs)["energy"] == -1
    assert calls[-1] == ("release",) and ("accuracy", 0.1) in calls


def test_geometric_binding_and_failure_trajectory(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Geometric binding and failure trajectory."""
    from qccodec.codecs.geometric import make_engine

    inp = prog_input_factory("gradient", "child")

    class Engine:
        def __init__(self, molecule: object) -> None:
            """Initialize the fake native object used by the binding test."""
            pass

    seen: list[ProgramInput] = []
    trajectory: list[ProgramOutput[ProgramInput, SinglePointData]] = []

    def child(
        input_data: ProgramInput,
        *,
        previous: ProgramOutput[ProgramInput, Data] | None = None,
        codec: Codec | None = None,
    ) -> ProgramOutput[ProgramInput, Data]:
        """Return a failed child output for optimizer trajectory recovery."""
        seen.append(input_data)
        return ProgramOutput.model_validate(
            {
                "input_data": input_data,
                "success": False,
                "traceback": "child failed",
                "results": SinglePointData.model_validate(
                    {"provenance": {"program": "child"}, "energy": -1}
                ),
            }
        )

    engine = make_engine(
        cast(ModuleType, SimpleNamespace(engine=SimpleNamespace(Engine=Engine))),
        object(),
        inp.structure,
        {"reference": inp.structure},
        inp,
        child,
        trajectory,
    )
    coords = inp.structure.geometry + 0.1
    with pytest.raises(RuntimeError, match="child calculation"):
        engine.calc_new(coords)
    np.testing.assert_allclose(seen[0].structure.geometry, coords)
    assert seen[0].structures["reference"] == inp.structure
    assert seen[0].keywords == inp.keywords
    assert len(trajectory) == 1 and (not trajectory[0].success)
    outer = get_codec("geometric")
    with pytest.raises(CalculationError) as raised:
        outer.decode(
            inp,
            PythonOutput(
                value={"trajectory": trajectory, "version": "test", "error": "failed"}
            ),
        )
    assert isinstance(raised.value.data, OptimizationData)
    assert raised.value.data.trajectory == trajectory


def test_restart_translation_does_not_mutate_request(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Restart translation does not mutate request."""
    inp = prog_input_factory("energy")
    original = inp.model_copy(deep=True)
    previous = SinglePointData.model_validate(
        {"provenance": {"program": "terachem"}, "files": {"scrgeometry/c0": b"restart"}}
    )
    restart = get_codec("terachem").restart
    assert restart is not None
    updated = restart(inp, previous)
    assert updated.files["c0"] == b"restart" and updated.keywords["guess"] == "c0"
    assert inp == original
