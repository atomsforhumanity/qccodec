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
    ConformerSearchData,
    Data,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
)

from qccodec import (
    Codec,
    PythonInput,
    PythonOutput,
    SubprocessInput,
    SubprocessOutput,
    get_codec,
    supported_programs,
)
from qccodec.exceptions import CalculationError, EncoderError


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


@pytest.mark.parametrize("program", ["orca", "terachem", "crest"])
def test_builtin_parser_ownership(
    program: str, prog_input_factory: Callable[..., ProgramInput], tmp_path: Path
) -> None:
    """File codecs decode through their own parser lists, including on copies."""
    original = get_codec(program)
    assert original.parsers and any(spec.file == "stdout" for spec in original.parsers)
    if program == "terachem":
        (tmp_path / "scr.geometry").mkdir()
    copied = Codec(original.program, original.calctypes, options=original.options)
    data = copied.decode(
        prog_input_factory("energy", program),
        SubprocessOutput(
            stdout="No scientific fields", directory=tmp_path, returncode=0
        ),
    )
    assert isinstance(data, SinglePointData) and data.energy is None
    assert original.parsers


def test_builtin_discovery_is_independent_of_parsers() -> None:
    """Python integrations remain discoverable even without file parsers."""
    assert {"orca", "terachem", "crest", "xtb", "geometric"} <= set(
        supported_programs()
    )
    assert get_codec("xtb").parsers == []


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


def test_terachem_unfinished_job_is_failure(
    prog_input_factory: Callable[..., ProgramInput], tmp_path: Path
) -> None:
    """A TeraChem log without 'Job finished:' is a program failure with recoverable data."""
    (tmp_path / "scr.geometry").mkdir()
    with pytest.raises(CalculationError, match="Job finished") as raised:
        get_codec("terachem").decode(
            prog_input_factory("energy", "terachem"),
            SubprocessOutput(
                stdout="TeraChem v1.9 Git Version: abc\nFINAL ENERGY: -1.0 a.u.\n",
                directory=tmp_path,
                returncode=0,
            ),
        )
    partial = raised.value.data
    assert isinstance(partial, SinglePointData)
    assert partial.provenance.program_version == "v1.9 [abc]"
    assert partial.energy == -1.0


def test_crest_rotamers_are_opt_in(
    prog_input_factory: Callable[..., ProgramInput], test_data_dir: Path
) -> None:
    """Rotamers are parsed only when requested, for execution and standalone decode."""
    inp = prog_input_factory("conformer_search", "crest")
    native = SubprocessOutput(
        directory=test_data_dir / "crest" / "conformers_with_rotamers", returncode=0
    )
    codec = get_codec("crest")
    assert codec.options == {"collect_rotamers": False}
    for enabled in [False, True]:
        options = {"collect_rotamers": enabled}
        result = codec.decode(inp, native, options)
        assert isinstance(result, ConformerSearchData)
        assert bool(result.rotamers) is enabled and result.conformers
        standalone = codec.parse(
            inp.calctype, directory=native.directory, input_data=inp, options=options
        )
        assert isinstance(standalone, ConformerSearchData)
        assert bool(standalone.rotamers) is enabled


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
    from qccodec.programs.xtb import invoke_xtb

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
    from qccodec.programs.geometric import make_engine

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


def test_propagate_wfn_moves_orbitals_onto_next_input(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Orbital files from a previous output become the next input's guess."""
    inp = prog_input_factory("energy")
    original = inp.model_copy(deep=True)
    previous = ProgramOutput(
        input_data=inp,
        success=True,
        results=SinglePointData.model_validate(
            {
                "provenance": {"program": "terachem"},
                "energy": -1,
                "files": {"scrgeometry/c0": b"wfn"},
            }
        ),
    )
    propagate = get_codec("terachem").propagate_wfn
    assert propagate is not None
    updated = propagate(previous, inp)
    assert updated.files["c0"] == b"wfn" and updated.keywords["guess"] == "c0"
    assert inp == original
    empty = ProgramOutput(
        input_data=inp,
        success=True,
        results=SinglePointData.model_validate(
            {"provenance": {"program": "terachem"}, "energy": -1}
        ),
    )
    with pytest.raises(EncoderError, match="wavefunction files"):
        propagate(empty, inp)
