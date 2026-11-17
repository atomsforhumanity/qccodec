"""Shared keywords every codec reads alike: the accuracy target."""

from __future__ import annotations

from typing import Any

import pytest
from qcdata import CalcType, ErrorKind, Model, ProgramError, ProgramInput, ProgramSpec
from qcdata.utils import water

from qccodec import Codec, SubprocessInput, get_codec
from qccodec.exceptions import EncoderError
from qccodec.program import CodecProgram

TARGET = {"gradient": 1e-6}


def _input(
    program: str,
    method: str | None,
    calctype: CalcType = CalcType.energy,
    **keywords: Any,
) -> ProgramInput:
    subprograms = []
    if program in ("geometric", "sella"):
        subprograms = [
            ProgramSpec(
                program="xtb", calctype=CalcType.gradient, model=Model(method="gfn2")
            )
        ]
    return ProgramInput(
        program=program,
        calctype=calctype,
        structure=water,
        model=Model(method=method) if method else None,
        keywords=keywords,
        subprograms=subprograms,
    )


# Every codec refuses a target: none maps it onto its program's thresholds yet.
REFUSALS = [
    ("xtb", "gfn2", CalcType.energy, "unsupported"),
    ("xtb", "gfnff", CalcType.gradient, "no_effect"),
    ("xtb", "gfn-ff", CalcType.hessian, "no_effect"),
    ("xtb", "gfnff", CalcType.optimization, "unsupported"),  # the optimizer's
    ("orca", "hf", CalcType.energy, "unsupported"),
    ("psi4", "hf", CalcType.gradient, "unsupported"),
    ("terachem", "hf", CalcType.energy, "unsupported"),
    ("crest", "gfnff", CalcType.conformer_search, "unsupported"),
    ("tblite", "gfn2", CalcType.energy, "unsupported"),
    ("dftbplus", "GFN2-xTB", CalcType.energy, "unsupported"),
    ("dftbplus", "DFTB", CalcType.gradient, "no_effect"),
    ("dftbplus", "DFTB", CalcType.optimization, "unsupported"),
    ("geometric", None, CalcType.optimization, "unsupported"),
    ("sella", None, CalcType.optimization, "unsupported"),
    ("aimnet2", "aimnet2", CalcType.energy, "no_effect"),
    ("torchani", "ANI2x", CalcType.gradient, "no_effect"),
    ("mace", "mace-off:small", CalcType.energy, "no_effect"),
    ("orb", "orb-v3-conservative-inf-omat", CalcType.energy, "no_effect"),
    ("fairchem", "uma-s-1p1", CalcType.hessian, "no_effect"),
    ("terachem-fe", "hf", CalcType.energy, "unsupported"),
    ("terachem-pbs", "hf", CalcType.gradient, "unsupported"),
]


@pytest.mark.parametrize(
    ("program", "method", "calctype", "refusal"),
    REFUSALS,
    ids=[f"{p}-{m}-{c.value}" for p, m, c, _ in REFUSALS],
)
def test_every_codec_refuses_an_accuracy_target(
    program: str, method: str | None, calctype: CalcType, refusal: str
) -> None:
    """Unsupported where the calculation converges something, else no_effect."""
    inp = _input(program, method, calctype, accuracy=TARGET)
    with pytest.raises(EncoderError) as e:
        get_codec(program).encode(inp)
    if refusal == "no_effect":
        assert (e.value.kind, e.value.code) == ("input", "no_effect")
    else:
        assert (e.value.kind, e.value.code) == ("unsupported", None)


codec = Codec("fake", calctypes={CalcType.energy, CalcType.gradient})
seen: list[ProgramInput] = []


@codec.encoder
def _encode(inp: ProgramInput) -> SubprocessInput:
    seen.append(inp)
    return SubprocessInput(("fake",))


@pytest.mark.parametrize(
    ("value", "code", "match"),
    [
        (1e-6, None, "JSON object of targets"),  # no calctype to aim a number at
        ({"energie": 1e-8}, "unknown_key", "did you mean accuracy.energy"),
        ({}, "no_effect", "sets no target"),
        ({"energy": -1.0}, None, "positive, finite"),
    ],
)
def test_a_value_that_is_not_a_target_is_an_input_error(
    value: Any, code: str | None, match: str
) -> None:
    """Read with qcdata's Accuracy.from_keywords, like every program."""
    with pytest.raises(EncoderError, match=match) as e:
        codec.encode(_input("fake", "m", accuracy=value))
    assert (e.value.kind, e.value.code) == ("input", code)


def test_null_is_the_default_and_never_reaches_the_encoder() -> None:
    seen.clear()
    codec.encode(_input("fake", "m", accuracy=None, other=1))
    assert seen[0].keywords == {"other": 1}


def test_programs_refuse_it_in_their_plan() -> None:
    """The adapter reports the codec's refusal as a ProgramError with its kind."""
    program = CodecProgram(codec, executable="/bin/true")
    with pytest.raises(ProgramError) as e:
        program.plan(_input("fake", "m", accuracy=TARGET))
    assert e.value.error.kind is ErrorKind.unsupported
    with pytest.raises(ProgramError) as e:
        program.plan(_input("fake", "m", threads=2, accuracy=TARGET))
    assert e.value.error.kind is ErrorKind.unsupported
    without = CodecProgram(Codec("fake", calctypes={CalcType.energy}, converges=False))
    without.codec.encoder(_encode)
    with pytest.raises(ProgramError) as e:
        without.plan(_input("fake", "m", accuracy=TARGET))
    assert (e.value.error.kind, e.value.error.code) == ("input", "no_effect")


def test_inputs_with_a_target_are_not_batched() -> None:
    """Computed alone, so the codec's refusal names the input."""
    aimnet2 = get_codec("aimnet2")
    assert aimnet2.batch_key(_input("aimnet2", "aimnet2", CalcType.gradient))
    assert (
        aimnet2.batch_key(
            _input("aimnet2", "aimnet2", CalcType.gradient, accuracy=TARGET)
        )
        is None
    )


# ------------------------------------------------------------------ settings schema

LISTED = [
    ("xtb", "gfn2", CalcType.energy),
    ("xtb", "gfn2", CalcType.optimization),
    ("tblite", "gfn2", CalcType.gradient),
    ("geometric", None, CalcType.optimization),
    ("geometric", None, CalcType.irc),
    ("sella", None, CalcType.transition_state),
    ("sella", None, CalcType.irc),
    ("aimnet2", "aimnet2", CalcType.gradient),
    ("torchani", "ANI2x", CalcType.gradient),
    ("mace", "mace-off-small", CalcType.gradient),
    ("orb", "orb-v3-conservative-omol", CalcType.gradient),
    ("fairchem", "uma-s-1p1", CalcType.gradient),
]


def _program(name: str) -> CodecProgram:
    import importlib

    program = importlib.import_module(f"qccodec.programs.{name}").program
    assert isinstance(program, CodecProgram)
    return program


@pytest.mark.parametrize(
    ("program", "method", "calctype"),
    LISTED,
    ids=[f"{p}-{c.value}" for p, _, c in LISTED],
)
def test_a_listed_schema_is_what_the_encoder_takes(
    program: str, method: str | None, calctype: CalcType
) -> None:
    """Every listed keyword passes the encoder's name check; any other is refused."""
    schema = _program(program).settings_schema(calctype)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["threads"]["x-concern"] == "resource"
    assert "accuracy" not in schema["properties"]  # refused by every codec
    codec = get_codec(program)
    for name in set(schema["properties"]) - {"threads"}:
        try:
            codec.encode(_input(program, method, calctype, **{name: 1}))
        except EncoderError as refused:
            assert refused.code != "unknown_key", (name, str(refused))
        except Exception:  # noqa: BLE001, S110 - a bad value, not an unknown name
            pass
    with pytest.raises(EncoderError) as e:
        codec.encode(_input(program, method, calctype, not_a_setting=1))
    assert e.value.code == "unknown_key"


def test_per_calctype_and_open_schemas() -> None:
    xtb = _program("xtb")
    assert "opt" in xtb.settings_schema(CalcType.optimization)["properties"]
    assert "opt" not in xtb.settings_schema(CalcType.energy)["properties"]
    energy = xtb.settings_schema(CalcType.energy)["properties"]
    assert {"alpb", "etemp", "external_potentials"} <= set(energy)
    assert not {"chrg", "uhf", "gfn", "grad"} & set(energy)  # the codec sets them
    geometric = _program("geometric")
    assert "irc_direction" in geometric.settings_schema(CalcType.irc)["properties"]
    assert (
        "irc_direction"
        not in geometric.settings_schema(CalcType.optimization)["properties"]
    )
    for name in ("orca", "psi4", "terachem", "crest", "dftbplus"):
        schema = _program(name).settings_schema(CalcType.energy)
        assert schema["additionalProperties"] is True  # passed through unlisted
        assert list(schema["properties"]) == ["threads"]
