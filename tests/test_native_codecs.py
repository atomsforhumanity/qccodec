from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qcdata import (
    ConformerSearchData,
    Data,
    Frame,
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
    assert get_codec("geometric").parsers == []


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


def test_nonzero_exit_still_reports_the_programs_explanation(
    prog_input_factory: Callable[..., ProgramInput], tmp_path: Path
) -> None:
    """A run that exited with an error raises the failure hook's message and data."""
    (tmp_path / "crest.engrad").write_text("# Energy ( Eh )\n#\n -1.0\n")
    with pytest.raises(CalculationError, match="FAILED") as raised:
        get_codec("crest").decode(
            prog_input_factory("gradient", "crest"),
            SubprocessOutput(
                stdout="Version 3.0.2, FAILED", directory=tmp_path, returncode=1
            ),
        )
    assert isinstance(raised.value.data, SinglePointData)
    assert raised.value.data.energy == -1


def test_recovery_mode_returns_partial_data_quietly(tmp_path: Path) -> None:
    """parse(failed=True) still recovers without raising, unless asked to report."""
    (tmp_path / "crest.engrad").write_text("# Energy ( Eh )\n#\n -1.0\n")
    codec = get_codec("crest")
    stdout = "Version 3.0.2, FAILED"
    data = codec.parse("gradient", stdout=stdout, directory=tmp_path, failed=True)
    assert isinstance(data, SinglePointData) and data.energy == -1
    with pytest.raises(CalculationError, match="FAILED"):
        codec.parse(
            "gradient",
            stdout=stdout,
            directory=tmp_path,
            failed=True,
            report_failure=True,
        )


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
    trajectory: list[Frame] = []

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

    structure = inp.require_structure()
    engine = make_engine(
        cast(ModuleType, SimpleNamespace(engine=SimpleNamespace(Engine=Engine))),
        object(),
        structure,
        {"reference": structure},
        inp,
        child,
        trajectory,
    )
    coords = structure.geometry + 0.1
    with pytest.raises(RuntimeError, match="child calculation"):
        engine.calc_new(coords)
    np.testing.assert_allclose(seen[0].require_structure().geometry, coords)
    assert seen[0].structures["reference"] == structure
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
    assert raised.value.data.frames == trajectory


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


def test_geometric_coordinate_systems_match_geometric() -> None:
    """Each coordsys builds the same coordinates geomeTRIC's own JSON interface does."""
    from qccodec.programs.geometric import COORDINATE_SYSTEMS

    # geometric.run_json.CoordSysDict: (class, connect, addcart)
    assert COORDINATE_SYSTEMS == {
        "cart": ("CartesianCoordinates", False, False),
        "prim": ("PrimitiveInternalCoordinates", True, False),
        "dlc": ("DelocalizedInternalCoordinates", True, False),
        "hdlc": ("DelocalizedInternalCoordinates", False, True),
        "tric": ("DelocalizedInternalCoordinates", False, False),
    }
    geometric = pytest.importorskip("geometric")
    for class_name, _, _ in COORDINATE_SYSTEMS.values():
        assert hasattr(geometric.internal, class_name)


def test_geometric_hessians_attach_to_the_frame_at_their_geometry() -> None:
    """Each recorded Hessian goes on the last frame computed at the same geometry."""
    import numpy as np
    from qcdata import Frame, SinglePointData, Structure

    from qccodec.programs.geometric import _attach_hessians

    def frame(z: float) -> Frame:
        structure = Structure(
            symbols=["H", "H"], geometry=np.array([[0.0, 0, 0], [0, 0, z]])
        )
        return Frame(
            structure=structure,
            success=True,
            results=SinglePointData.model_validate(
                {
                    "provenance": {"program": "test"},
                    "energy": -1.0,
                    "gradient": np.zeros((2, 3)),
                }
            ),
        )

    trajectory = [frame(1.4), frame(1.35), frame(1.3)]
    first, last = np.eye(6), 2 * np.eye(6)
    _attach_hessians(
        trajectory,
        [
            (np.array([0, 0, 0, 0, 0, 1.4]), first),
            (np.array([0, 0, 0, 0, 0, 1.3]), last),
            (np.array([0, 0, 0, 0, 0, 9.9]), 3 * np.eye(6)),  # no such frame
        ],
    )
    hessians = [frame.results.hessian for frame in trajectory]
    assert hessians[0] is not None and np.array_equal(hessians[0], first)
    assert hessians[1] is None
    assert hessians[2] is not None and np.array_equal(hessians[2], last)


def test_geometric_irc_path_is_the_accepted_points_in_order() -> None:
    """geomeTRIC's progress (forward reversed, TS, backward) becomes the path.

    Rejected trial steps (frames not in progress) are left out; the path runs from the
    backward end to the forward end with the transition state's index.
    """
    from types import SimpleNamespace

    import numpy as np
    from qcdata import Frame, SinglePointData, Structure
    from qcref import constants as c

    from qccodec.programs.geometric import _irc_path

    def frame(x: float) -> Frame:
        structure = Structure(
            symbols=["H", "H"], geometry=np.array([[x, 0.0, 0.0], [x, 0.0, 1.4]])
        )
        return Frame(
            structure=structure,
            success=True,
            results=SinglePointData.model_validate(
                {"provenance": {"program": "test"}, "energy": -abs(x)}
            ),
        )

    ts, f1, f2, rejected, b1 = (
        frame(0.0),
        frame(0.1),
        frame(0.2),
        frame(9.0),
        frame(-0.1),
    )
    trajectory = [ts, f1, rejected, f2, b1]
    progress = [f2, f1, ts, b1]  # geomeTRIC's order after both branches
    optimizer = SimpleNamespace(
        progress=SimpleNamespace(
            xyzs=[f.structure.geometry * c.BOHR_TO_ANGSTROM for f in progress]
        )
    )
    path, ts_index = _irc_path(optimizer, trajectory, ts.structure)
    assert [f.results.energy for f in path] == [-0.1, 0.0, -0.1, -0.2]
    assert ts_index == 1


def _neb_input(**keywords: Any) -> ProgramInput:
    from qcdata import CalcType, Model, ProgramSpec, Structure

    chain = [
        Structure(
            symbols=["H", "H"], geometry=np.array([[0.0, 0, 0], [0, 0, 1.2 + 0.1 * i]])
        )
        for i in range(5)
    ]
    return ProgramInput(
        program="geometric",
        calctype=CalcType.path_optimization,
        structures={"initial_chain": chain},
        subprograms=[
            ProgramSpec(
                program="morse", calctype=CalcType.gradient, model=Model(method="m")
            )
        ],
        keywords=keywords,
    )


def _morse(child: ProgramInput) -> ProgramOutput[ProgramInput, Data]:
    """H2 on a Morse curve: energy and gradient (Hartree, bohr)."""
    g = child.require_structure().geometry
    d = g[1] - g[0]
    r = float(np.linalg.norm(d))
    x = np.exp(-1.0 * (r - 1.4))
    energy = 0.17 * (1 - x) ** 2
    dedr = 2 * 0.17 * (1 - x) * x
    grad = np.array([-dedr * d / r, dedr * d / r])
    return ProgramOutput.model_validate(
        {
            "input_data": child,
            "success": True,
            "results": SinglePointData.model_validate(
                {"provenance": {"program": "morse"}, "energy": energy, "gradient": grad}
            ),
        }
    )


def test_geometric_neb_encoding() -> None:
    """The chain and both callbacks are requested; bigchem is the runner's job."""
    native = get_codec("geometric").encode(_neb_input(images=5))
    assert isinstance(native, PythonInput)
    assert native.child_callback == "compute_child"
    assert native.children_callback == "compute_children"
    assert len(native.kwargs["chain"]) == 5
    assert native.kwargs["keywords"] == {"images": 5}
    with pytest.raises(EncoderError, match="bigchem"):
        get_codec("geometric").encode(_neb_input(bigchem=True))


def test_geometric_neb_computes_each_iteration_s_images_together() -> None:
    """Every evaluation hands all images to compute_children; paths are recorded."""
    pytest.importorskip("geometric")
    inp = _neb_input(images=5, neb_maxcyc=3)
    native = get_codec("geometric").encode(inp)
    assert isinstance(native, PythonInput)
    sizes: list[int] = []

    def children(inputs: Any) -> list[ProgramOutput[ProgramInput, Data]]:
        sizes.append(len(inputs))
        return [_morse(child) for child in inputs]

    value = native.function(
        **native.kwargs, compute_child=_morse, compute_children=children
    )
    assert sizes and set(sizes) == {5}
    assert value["paths"] and all(len(frames) == 5 for frames, _ in value["paths"])
    data = None
    try:
        data = get_codec("geometric").decode(inp, PythonOutput(value=value))
    except CalculationError as exc:  # three cycles may not converge
        data = exc.data
    from qcdata import PathOptimizationData

    assert isinstance(data, PathOptimizationData) and data.final_path is not None
    assert len(data.paths) == len(value["paths"])


def test_geometric_neb_falls_back_to_one_child_at_a_time() -> None:
    """Without compute_children (an older runner), images go to compute_child."""
    pytest.importorskip("geometric")
    native = get_codec("geometric").encode(_neb_input(images=5, neb_maxcyc=1))
    assert isinstance(native, PythonInput)
    seen: list[ProgramInput] = []

    def child(inp: ProgramInput, **_: Any) -> ProgramOutput[ProgramInput, Data]:
        seen.append(inp)
        return _morse(inp)

    native.function(**native.kwargs, compute_child=child)
    assert len(seen) >= 5 and len(seen) % 5 == 0


def test_geometric_neb_reports_failed_images() -> None:
    """A failed image stops the run with the frames computed so far."""
    pytest.importorskip("geometric")
    inp = _neb_input(images=5)
    native = get_codec("geometric").encode(inp)
    assert isinstance(native, PythonInput)

    def children(inputs: Any) -> list[ProgramOutput[ProgramInput, Data]]:
        outputs = [_morse(child) for child in inputs]
        outputs[2] = outputs[2].model_copy(update={"success": False, "traceback": "x"})
        return outputs

    value = native.function(
        **native.kwargs, compute_child=_morse, compute_children=children
    )
    assert "images [2]" in value["error"]
    with pytest.raises(CalculationError) as raised:
        get_codec("geometric").decode(inp, PythonOutput(value=value))
    from qcdata import PathOptimizationData

    assert isinstance(raised.value.data, PathOptimizationData)
    assert not raised.value.data.paths[-1].frames[2].success
