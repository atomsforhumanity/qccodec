"""Shared keywords every codec reads alike: the accuracy target."""

from __future__ import annotations

from typing import Any

import pytest
from qcdata import CalcType, ErrorKind, Model, ProgramError, ProgramInput, ProgramSpec
from qcdata.utils import water

from qccodec import Codec, PythonInput, SubprocessInput, get_codec
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
        # DFTB+'s non-SCC DFTB needs its Slater-Koster set.
        model=Model(method=method, basis="3ob-3-1" if method == "DFTB" else None)
        if method
        else None,
        keywords=keywords,
        subprograms=subprograms,
    )


# A calculation with nothing to converge meets a target as it is; one with convergence
# settings refuses it, since no codec maps a target onto its program's thresholds yet.
TARGETS = [
    ("xtb", "gfn2", CalcType.energy, "unsupported"),
    ("xtb", "gfnff", CalcType.gradient, "met"),
    ("xtb", "gfn-ff", CalcType.hessian, "met"),
    ("xtb", "gfnff", CalcType.optimization, "unsupported"),  # the optimizer's
    ("orca", "hf", CalcType.energy, "unsupported"),
    ("psi4", "hf", CalcType.gradient, "unsupported"),
    ("terachem", "hf", CalcType.energy, "unsupported"),
    ("crest", "gfnff", CalcType.conformer_search, "unsupported"),
    ("tblite", "gfn2", CalcType.energy, "unsupported"),
    ("dftbplus", "GFN2-xTB", CalcType.energy, "unsupported"),
    ("dftbplus", "DFTB", CalcType.gradient, "met"),
    ("dftbplus", "DFTB", CalcType.optimization, "unsupported"),
    ("geometric", None, CalcType.optimization, "unsupported"),
    ("sella", None, CalcType.optimization, "unsupported"),
    ("aimnet2", "aimnet2", CalcType.energy, "met"),
    ("torchani", "ANI2x", CalcType.gradient, "met"),
    ("mace", "mace-off-small", CalcType.energy, "met"),
    ("orb", "orb-v3-conservative-inf-omat", CalcType.energy, "met"),
    ("fairchem", "uma-s-1p1", CalcType.hessian, "met"),
    ("terachem-fe", "hf", CalcType.energy, "unsupported"),
    ("terachem-pbs", "hf", CalcType.gradient, "unsupported"),
]


@pytest.mark.parametrize(
    ("program", "method", "calctype", "outcome"),
    TARGETS,
    ids=[f"{p}-{m}-{c.value}" for p, m, c, _ in TARGETS],
)
def test_a_target_is_met_without_convergence_settings_else_refused(
    program: str, method: str | None, calctype: CalcType, outcome: str
) -> None:
    """Met (and never passed to the program) where nothing converges, else unsupported."""
    inp = _input(program, method, calctype, accuracy=TARGET)
    codec = get_codec(program)
    if outcome == "met":
        assert codec.accuracy(inp) is not None
        settings = {"slater_koster_dir": "/data/skf"} if program == "dftbplus" else {}
        assert codec.encode(
            _input(program, method, calctype, accuracy=TARGET, **settings)
        ) == codec.encode(_input(program, method, calctype, **settings))
        return
    with pytest.raises(EncoderError) as e:
        codec.encode(inp)
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


def _value(fail: bool = False) -> dict[str, Any]:
    if fail:
        raise RuntimeError("the model failed")
    return {"energy": -1.0, "version": "1.0"}


def _model(fail: bool = False) -> Codec:
    """A library with nothing to converge, which batches."""
    from qccodec.programs._python import single_point_decoder

    return Codec(
        "model",
        calctypes={CalcType.energy},
        encode=lambda inp: PythonInput(_value, args=(fail,)),
        decode=single_point_decoder("model"),
        encode_batch=lambda inputs: PythonInput(lambda: [_value()] * len(inputs)),
        batch_key=lambda inp: "all",
        converges=False,
    )


MET = {"accuracy": {"target": TARGET, "thresholds": {}}}


def test_a_met_target_is_recorded(tmp_path: Any) -> None:
    """In the record, as met with no thresholds; also on a failure, and in a batch."""
    program = CodecProgram(_model())
    out = program.compute(_input("model", "m", accuracy=TARGET))
    assert out.success and out.record == MET
    assert out.input_data.keywords == {"accuracy": TARGET}  # the input as given
    assert program.compute(_input("model", "m")).record == {}
    assert program.plan(_input("model", "m", accuracy=TARGET))["resources"]
    with pytest.raises(ProgramError) as e:
        program.compute(_input("model", "m", accuracy={"energy": 0.0}))
    assert e.value.error.kind is ErrorKind.input  # still read as a target
    failing = CodecProgram(_model(fail=True))
    with pytest.raises(ProgramError) as e:
        failing.compute(_input("model", "m", accuracy=TARGET))
    assert e.value.output is not None and e.value.output.record == MET
    batch = program.compute_batch(
        [_input("model", "m", accuracy=TARGET), _input("model", "m")]
    )
    assert [o.record for o in batch] == [MET, {}]
    assert all(o.execution.extras.get("batch_size") for o in batch)


def test_only_inputs_with_a_refused_target_are_computed_alone() -> None:
    """A target the codec meets shares a batch; one it refuses names its input."""
    aimnet2 = get_codec("aimnet2")
    accepted = _input("aimnet2", "aimnet2", CalcType.gradient, accuracy=TARGET)
    assert aimnet2.batch_key(accepted)
    assert (
        aimnet2.encode_batch([accepted]).args
        == aimnet2.encode_batch([_input("aimnet2", "aimnet2", CalcType.gradient)]).args
    )  # the library never sees it
    bare = _input("aimnet2", "aimnet2", CalcType.gradient, accuracy=1e-6)
    assert aimnet2.batch_key(bare) is None
    converging = Codec(
        "model",
        calctypes={CalcType.energy},
        encode_batch=lambda inputs: PythonInput(list),
        batch_key=lambda inp: "all",
    )
    assert converging.batch_key(_input("model", "m")) == "all"
    assert converging.batch_key(_input("model", "m", accuracy=TARGET)) is None


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
    codec = get_codec(program)
    # Listed where a calculation may have nothing to converge.
    assert ("accuracy" in schema["properties"]) is (codec.converges is not True)
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
        # Non-SCC DFTB has nothing to converge, so DFTB+ may meet a target.
        shared = ["threads", "accuracy"] if name == "dftbplus" else ["threads"]
        assert list(schema["properties"]) == shared
