"""The qccodec.testing harness itself, exercised with a toy codec and fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qcdata import CalcType, Model, ProgramInput, SinglePointData
from qcdata.utils import water

from qccodec import Codec, Context, SubprocessInput
from qccodec.testing import (
    Case,
    assert_data_equal,
    cases,
    check_codec,
    load_case,
    run_case,
)


def toy_codec() -> Codec:
    """A file-based codec with stdout, file, and directory parsers plus an option."""
    codec = Codec("toy", {CalcType.energy, CalcType.gradient}, options={"natoms": True})
    codec.encoder(
        lambda inp: SubprocessInput(
            ("toy", "run.inp"), files={"run.inp": f"{inp.model.method}\n"}
        )
    )
    codec.parser(target="energy")(lambda stdout: float(stdout.split()[1]))
    codec.parser(file="run.grad", target="gradient", calctypes=[CalcType.gradient])(
        lambda text: [[float(v) for v in line.split()] for line in text.splitlines()]
    )

    @codec.parser(file="directory")
    def natoms(ctx: Context) -> dict[str, int]:
        """Fill calcinfo_natoms from the input when the option is on."""
        assert ctx.input_data is not None
        return (
            {"calcinfo_natoms": len(ctx.input_data.structure.symbols)}
            if ctx.options["natoms"]
            else {}
        )

    return codec


def write_case(
    root: Path,
    name: str,
    *,
    calctype: CalcType = CalcType.gradient,
    stdout: str | None = "energy -1.5",
    files: dict[str, str] | None = None,
    expected: SinglePointData | None = None,
    settings: dict[str, object] | None = None,
    encoded: dict[str, str] | None = None,
) -> Path:
    """Write a fixture case directory for the toy codec."""
    case = root / "toy" / name
    case.mkdir(parents=True)
    inp = ProgramInput(
        program="toy", calctype=calctype, structure=water, model=Model(method="hf")
    )
    (case / "input.json").write_text(inp.model_dump_json())
    if stdout is not None:
        (case / "stdout").write_text(stdout)
    for fname, text in (files or {}).items():
        (case / fname).write_text(text)
    if expected is not None:
        (case / "expected.json").write_text(expected.model_dump_json())
    if settings:
        (case / "case.json").write_text(json.dumps(settings))
    if encoded is not None:
        (case / "encoded").mkdir()
        for fname, text in encoded.items():
            (case / "encoded" / fname).write_text(text)
    return case


def expected_data(**fields: object) -> SinglePointData:
    """Build an expected result for the toy program."""
    return SinglePointData.model_validate({"provenance": {"program": "toy"}, **fields})


def test_case_discovery_and_loading(tmp_path: Path) -> None:
    """Cases are found by input.json; stdout-only cases get no directory."""
    write_case(tmp_path, "b", expected=expected_data(energy=-1.5))
    write_case(
        tmp_path,
        "a",
        files={"run.grad": "0 0 0\n"},
        expected=expected_data(energy=-1.5),
    )
    found = cases(tmp_path)
    assert [str(c) for c in found] == ["toy/a", "toy/b"]
    assert found[0].directory == found[0].path and found[1].directory is None
    assert found[0].input_data.calctype == CalcType.gradient
    assert isinstance(found[0].expected, SinglePointData)


def test_run_case_success_with_encoding_and_options(tmp_path: Path) -> None:
    """A full case checks encoding, the decode result, and each parser."""
    gradient = "0.1 0.2 0.3\n" * 3
    expected = expected_data(
        energy=-1.5, gradient=[[0.1, 0.2, 0.3]] * 3, calcinfo_natoms=3
    )
    case = write_case(
        tmp_path,
        "ok",
        files={"run.grad": gradient},
        expected=expected,
        encoded={"command.txt": "toy\nrun.inp\n", "run.inp": "hf\n"},
    )
    run_case(toy_codec(), load_case(case))

    (case / "encoded" / "run.inp").write_text("wrong\n")
    with pytest.raises(AssertionError, match="run.inp differs"):
        run_case(toy_codec(), load_case(case))


def test_run_case_reports_mismatches_and_options(tmp_path: Path) -> None:
    """Wrong expected values are reported with their path; options reach parsers."""
    case = write_case(
        tmp_path,
        "mismatch",
        files={"run.grad": "0 0 0\n"},
        expected=expected_data(energy=-2.0, gradient=[[0, 0, 0]], calcinfo_natoms=3),
    )
    with pytest.raises(AssertionError, match=r"toy/mismatch\.energy: -1\.5 != -2\.0"):
        run_case(toy_codec(), load_case(case))

    case = write_case(
        tmp_path,
        "no_natoms",
        files={"run.grad": "0 0 0\n"},
        expected=expected_data(energy=-1.5, gradient=[[0, 0, 0]]),
        settings={"options": {"natoms": False}},
    )
    run_case(toy_codec(), load_case(case))


def test_run_case_failure_and_recovery(tmp_path: Path) -> None:
    """raises/failed settings cover exceptions with partial data and recovery mode."""
    partial = expected_data(energy=-1.5, calcinfo_natoms=3)
    case = write_case(
        tmp_path,
        "missing_gradient",
        files={"other": ""},
        expected=partial,
        settings={"raises": "ParserError"},
    )
    run_case(toy_codec(), load_case(case))

    case = write_case(
        tmp_path,
        "recovered",
        files={"other": ""},
        expected=partial,
        settings={"failed": True},
    )
    run_case(toy_codec(), load_case(case))

    case = write_case(
        tmp_path,
        "did_not_raise",
        calctype=CalcType.energy,
        files={"run.grad": "0 0 0\n"},
        settings={"raises": "ParserError"},
    )
    with pytest.raises(AssertionError, match="expected ParserError"):
        run_case(toy_codec(), load_case(case))


def test_assert_data_equal_tolerance_and_ignored_keys() -> None:
    """Numbers compare within tolerance; execution and version metadata are ignored."""
    a = expected_data(energy=-1.0000000001)
    b = expected_data(energy=-1.0)
    assert_data_equal(a, b)
    with pytest.raises(AssertionError, match="energy"):
        assert_data_equal(a, b, rtol=1e-12)
    assert_data_equal(
        {
            "execution": {"scratch_dir": "/x"},
            "provenance": {"qcdata_version": "1", "program": "x"},
        },
        {"provenance": {"program": "x"}},
    )


def test_check_codec_catches_bad_targets_and_empty_parses(tmp_path: Path) -> None:
    """The contract check flags unknown targets and broken empty-parse behavior."""
    check_codec(
        toy_codec(),
        inputs=[
            ProgramInput(
                program="toy",
                calctype=CalcType.energy,
                structure=water,
                model=Model(method="hf"),
            )
        ],
    )

    bad = Codec("bad", {CalcType.energy})
    bad.parser(target="energ")(lambda s: 1.0)
    bad.parser(target=("provenance", "nope"), required=False)(lambda s: "x")
    with pytest.raises(AssertionError) as raised:
        check_codec(bad)
    message = str(raised.value)
    assert "target 'energ' is not a field of SinglePointData" in message
    assert "('provenance', 'nope')" in message

    escaping = Codec(
        "escape",
        {CalcType.energy},
        encode=lambda inp: SubprocessInput(("x",), files={"../out": ""}),
    )
    with pytest.raises(AssertionError, match="escapes directory"):
        check_codec(
            escaping,
            inputs=[
                ProgramInput(
                    program="escape", calctype=CalcType.energy, structure=water
                )
            ],
        )


def test_case_str_is_its_name(tmp_path: Path) -> None:
    """Cases print as program/name for pytest ids."""
    case = write_case(tmp_path, "named", expected=expected_data(energy=-1.5))
    loaded = load_case(case)
    assert isinstance(loaded, Case) and str(loaded) == "toy/named"


def test_encoded_files_ignore_the_qcdata_version_stamp(tmp_path: Path) -> None:
    """Generated files may differ only in qcdata's version stamp across releases."""
    codec = toy_codec()
    codec.encoder(
        lambda inp: SubprocessInput(
            ("toy", "run.inp"), files={"run.inp": "qcdata_version=0.99.0 hf\n"}
        )
    )
    case = write_case(
        tmp_path,
        "stamped",
        files={"run.grad": "0 0 0\n"},
        expected=expected_data(energy=-1.5, gradient=[[0, 0, 0]], calcinfo_natoms=3),
        encoded={
            "command.txt": "toy\nrun.inp\n",
            "run.inp": "qcdata_version=0.19.0 hf\n",
        },
    )
    run_case(codec, load_case(case))
