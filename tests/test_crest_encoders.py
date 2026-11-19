import pytest
from qcdata import ProgramInput
from qcdata.utils import water

from qccodec.exceptions import EncoderError
from qccodec.programs.crest import _to_toml_dict, validate_input


def test_validate_input(prog_input_factory):
    inp_obj = prog_input_factory("conformer_search")
    validate_input(inp_obj)

    with pytest.raises(EncoderError):
        inp_obj.keywords["charge"] = -1
        validate_input(inp_obj)

    with pytest.raises(EncoderError):
        inp_obj.keywords["uhf"] = 0
        validate_input(inp_obj)

    with pytest.raises(EncoderError):
        inp_obj.keywords["runtype"] = "imtd-gc"
        validate_input(inp_obj)


def test_toml_dict():
    """Test converting a ProgramInput object to a TOML dictionary for CREST."""

    weird_water = water.model_copy(update={"charge": -1, "multiplicity": 2})
    inp_obj = ProgramInput(
        program="crest",
        structure=weird_water,
        calctype="conformer_search",
        model={"method": "gfn2"},
        keywords={"calculation": {"level": [{"alpb": "acetonitrile"}]}},
    )

    toml_dict = _to_toml_dict(inp_obj, "struct.xyz")

    assert toml_dict["input"] == "struct.xyz"
    assert toml_dict["runtype"] == "imtd-gc"
    assert "threads" not in toml_dict  # the runner's grant fills it in

    # Adds values correctly to existing "calculation" key
    assert toml_dict["calculation"]["level"][0]["method"] == "gfn2"
    assert toml_dict["calculation"]["level"][0]["charge"] == -1
    assert toml_dict["calculation"]["level"][0]["uhf"] == 1
    assert toml_dict["calculation"]["level"][0]["alpb"] == "acetonitrile"

    # Respects explicitly set threads and handles no "calculation" key
    inp_obj = ProgramInput(
        program="crest",
        structure=weird_water,
        calctype="conformer_search",
        model={"method": "gfn2"},
        keywords={"threads": 2},
    )

    toml_dict = _to_toml_dict(inp_obj, "struct.xyz")
    assert toml_dict["threads"] == 2
    assert toml_dict["calculation"]["level"][0]["method"] == "gfn2"
    assert toml_dict["calculation"]["level"][0]["charge"] == -1
    assert toml_dict["calculation"]["level"][0]["uhf"] == 1


def test_unknown_tables_are_refused(prog_input_factory):
    """CREST ignores tables it does not know, so the codec refuses them."""
    inp = prog_input_factory("conformer_search")
    for keywords, hint in [
        ({"cregn": {"ewin": 3.0}}, "cregen"),
        ({"calculation": {"levels": [{"alpb": "water"}]}}, "level"),
        ({"dynamics": {"mtea": [{"alpha": 0.1}]}}, "meta"),
    ]:
        bad = inp.model_copy(update={"keywords": keywords})
        with pytest.raises(EncoderError, match=f"did you mean '{hint}'") as e:
            validate_input(bad)
        assert e.value.code == "unknown_key"
    fine = {"calculation": {"level": [{"alpb": "water"}]}, "cregen": {"ewin": 3.0}}
    validate_input(inp.model_copy(update={"keywords": fine}))


# Lines CREST 3.0.2 prints for an input file it refuses (src/parsing/parse_maindata.f90,
# parse_calcdata.f90 and confparse2.f90), before stopping with a nonzero exit.
CREST_REFUSES_KEY = """\
unrecognized KEYWORD in main section : ewinn
unrecognized KEYWORD in [cregen]-block : ewinx
2 error(s) while reading input file
"""
CREST_REFUSES_VALUE = """\
unrecognized ARGUMENT : gfn3
1 error(s) while reading input file
"""


@pytest.mark.parametrize(
    ("stdout", "kind", "code", "match"),
    [
        (CREST_REFUSES_KEY, "input", "unknown_key", "'ewinx' \\(\\[cregen\\]-block\\)"),
        (CREST_REFUSES_VALUE, "input", None, "'gfn3'"),
        ("... FAILED ...", "internal", None, "FAILED"),
    ],
)
def test_crest_failures_have_kinds(prog_input_factory, stdout, kind, code, match):
    from qccodec import get_codec
    from qccodec.exceptions import CalculationError

    inp = prog_input_factory("energy", "crest")
    with pytest.raises(CalculationError, match=match) as e:
        get_codec("crest").parse(inp.calctype, stdout=stdout, input_data=inp)
    assert (e.value.kind, e.value.code) == (kind, code)
