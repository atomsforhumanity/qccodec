import pytest
from qcdata import ProgramInput
from qcdata.utils import water

from qccodec.encoders.crest import _to_toml_dict, validate_input
from qccodec.exceptions import EncoderError


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
    assert toml_dict.get("threads") is not None  # added implicitly if not set

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
