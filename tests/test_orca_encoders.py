import pytest
from qcdata import CalcType, Model, ProgramInput
from qcdata.utils import water

from qccodec.exceptions import EncoderError
from qccodec.programs.orca import _validate_keywords, encode


@pytest.mark.parametrize(
    "calctype, method, basis, keywords",
    [
        (CalcType.energy, "b3lyp", "def2-svp", {}),
        (CalcType.energy, "b3lyp", "def2-svp", {"maxcore": 500, "pal": 4}),
        (CalcType.energy, "b3lyp", "def2-svp", {"scf": {"convergence": "verytight"}}),
        (
            CalcType.energy,
            "revdsd-pbep86-d4/2021",
            "def2-svp",
            {"basis": {"auxc": "def2-svp/c"}},
        ),
        (CalcType.gradient, "b3lyp", "def2-svp", {}),
        (
            CalcType.gradient,
            "revdsd-pbep86-d4/2021",
            "def2-svp",
            {"basis": {"auxc": "def2-svp/c"}, "numgrad": True},
        ),
        (
            CalcType.gradient,
            "revdsd-pbep86-d4/2021",
            "def2-svp",
            {"basis": {"auxc": "def2-svp/c"}, "numgrad": {"accuracy": 6, "dx": 0.002}},
        ),
        (CalcType.hessian, "b3lyp", "def2-svp", {}),
        (
            CalcType.hessian,
            "revdsd-pbep86-d4/2021",
            "def2-svp",
            {"basis": {"auxc": "def2-svp/c"}, "freq": {"numfreq": True}},
        ),
        (CalcType.optimization, "b3lyp", "def2-svp", {"geom": {"maxiter": 30}}),
        (
            CalcType.optimization,
            "b3lyp",
            "def2-svp",
            {"geom": {"maxiter": 30}, "numgrad": True},
        ),
        (CalcType.transition_state, "b3lyp", "def2-svp", {"geom": {"calc_hess": True}}),
        (
            CalcType.transition_state,
            "revdsd-pbep86-d4/2021",
            "def2-svp",
            {
                "basis": {"auxc": "def2-svp/c"},
                "geom": {"calc_hess": True, "numhess": True},
            },
        ),
    ],
)
def test_write_input_files(
    calctype: CalcType, method: str, basis: str, keywords: dict[str, object]
):
    """Test write_input_files method."""
    program_input = ProgramInput(
        program="orca",
        calctype=calctype,
        model=Model(method=method, basis=basis),
        structure=water,
        keywords=keywords,
    )
    native_input = encode(program_input)
    input_file = native_input.files["orca.inp"]
    assert isinstance(input_file, str)

    # Check that all defined main / block keywords ended up in the file
    for keyword, value in keywords.items():
        assert keyword in input_file, f"{keyword} not in\n{input_file}"

        if isinstance(value, dict):
            for block_keyword in value:
                assert block_keyword in input_file, (
                    f"{block_keyword} not in\n{input_file}"
                )


def test_no_none_fields():
    """Tests 'None' not in input file when basis not provided."""
    program_input = ProgramInput(
        program="orca",
        calctype=CalcType.optimization,
        model=Model(method="xTB"),
        structure=water,
    )
    native_input = encode(program_input)
    input_file = native_input.files["orca.inp"]
    assert isinstance(input_file, str)

    assert "None" not in input_file, f"'None' found in\n{input_file}"


def test_validate_keywords_raises_error_if_coords_block_in_keywords():
    """The 'coords' block should not be set directly as a keyword."""

    keywords = {"coords": {"units": "angstrom"}}
    with pytest.raises(EncoderError):
        _validate_keywords(keywords)


def test_validate_keywords_raises_error_if_method_block_contains_method_keyword():
    """The 'method' keyword should not be set inside the 'method' block."""
    keywords = {"method": {"method": "b3lyp"}}
    with pytest.raises(EncoderError):
        _validate_keywords(keywords)


def test_validate_keywords_raises_error_if_method_block_contains_runtyp_keyword():
    """The 'runtyp' keyword should not be set inside the 'method' block."""
    keywords = {"method": {"runtyp": "sp"}}
    with pytest.raises(EncoderError):
        _validate_keywords(keywords)


def test_validate_keywords_raises_error_if_basis_block_contains_basis_keyword():
    """The 'basis' keyword should not be set inside the 'basis' block."""
    keywords = {"basis": {"basis": "def2-svp"}}
    with pytest.raises(EncoderError):
        _validate_keywords(keywords)


def _inp_lines(calctype: CalcType, keywords: dict) -> list[str]:
    native = encode(
        ProgramInput(
            program="orca",
            structure=water,
            calctype=calctype,
            model=Model(method="hf", basis="sto-3g"),
            keywords=keywords,
        )
    )
    return str(native.files["orca.inp"]).splitlines()


def test_true_keywords_are_simple_input_words() -> None:
    """True adds a simple-input word; False and None leave it out."""
    lines = _inp_lines(
        CalcType.gradient,
        {"TightSCF": True, "numgrad": False, "slowconv": None, "d3bj": True},
    )
    assert "! tightscf" in lines and "! d3bj" in lines
    assert not any("numgrad" in line or "slowconv" in line for line in lines)


def test_numfreq_flag_and_block() -> None:
    """numfreq as a flag or a block switches the Hessian to numerical; False doesn't."""
    assert "! numfreq" in _inp_lines(CalcType.hessian, {"numfreq": True})
    assert "! numfreq" in _inp_lines(CalcType.hessian, {"numfreq": {"dx": 0.005}})
    assert "! freq" in _inp_lines(CalcType.hessian, {"numfreq": False})


@pytest.mark.parametrize("value", ["yes", 1, 2.5])
def test_other_scalar_keywords_are_rejected(value: object) -> None:
    """Nothing is dropped silently: a scalar must be a flag, maxcore, or pal."""
    with pytest.raises(EncoderError, match="tightscf"):
        _inp_lines(CalcType.energy, {"tightscf": value})
