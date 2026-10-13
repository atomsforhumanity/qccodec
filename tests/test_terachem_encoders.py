import pytest

from qccodec.exceptions import EncoderError
from qccodec.programs.terachem import PADDING, XYZ_FILENAME, encode


def test_write_input_files(prog_input_factory):
    """Test write_input_files method."""
    prog_input_factory = prog_input_factory("energy")
    prog_input_factory.keywords.update({"purify": "no", "some-bool": False})

    native_input = encode(prog_input_factory)
    # Testing that we capture:
    # 1. Driver
    # 2. Structure
    # 3. Model
    # 4. Keywords (test booleans to lower case, ints, sts, floats)

    correct_tcin = (
        f"{'run':<{PADDING}} {prog_input_factory.calctype.value}\n"
        f"{'coordinates':<{PADDING}} {XYZ_FILENAME}\n"
        f"{'charge':<{PADDING}} {prog_input_factory.structure.charge}\n"
        f"{'spinmult':<{PADDING}} {prog_input_factory.structure.multiplicity}\n"
        f"{'method':<{PADDING}} {prog_input_factory.model.method}\n"
        f"{'basis':<{PADDING}} {prog_input_factory.model.basis}\n"
        f"{'purify':<{PADDING}} {prog_input_factory.keywords['purify']}\n"
        f"{'some-bool':<{PADDING}} "
        f"{str(prog_input_factory.keywords['some-bool']).lower()}\n"
    )
    assert native_input.files["tc.in"] == correct_tcin


def test_write_input_files_renames_hessian_to_frequencies(prog_input_factory):
    """Test write_input_files method for hessian."""
    # Modify input to be a hessian calculation
    prog_input_factory = prog_input_factory("hessian")
    prog_input_factory.keywords.update({"purify": "no", "some-bool": False})
    native_input = encode(prog_input_factory)

    assert native_input.files["tc.in"] == (
        f"{'run':<{PADDING}} frequencies\n"
        f"{'coordinates':<{PADDING}} {XYZ_FILENAME}\n"
        f"{'charge':<{PADDING}} {prog_input_factory.structure.charge}\n"
        f"{'spinmult':<{PADDING}} {prog_input_factory.structure.multiplicity}\n"
        f"{'method':<{PADDING}} {prog_input_factory.model.method}\n"
        f"{'basis':<{PADDING}} {prog_input_factory.model.basis}\n"
        f"{'purify':<{PADDING}} {prog_input_factory.keywords['purify']}\n"
        f"{'some-bool':<{PADDING}} "
        f"{str(prog_input_factory.keywords['some-bool']).lower()}\n"
    )


def test_encode_raises_error_qcdata_args_passed_as_keywords(prog_input_factory):
    """These keywords should not be in the .keywords dict. They belong on structured
    qcdata objects instead."""
    qcdata_keywords_from_terachem = ["charge", "spinmult", "method", "basis", "run"]
    prog_input_factory = prog_input_factory("energy")
    for keyword in qcdata_keywords_from_terachem:
        prog_input_factory.keywords[keyword] = "some value"
        with pytest.raises(EncoderError):
            encode(prog_input_factory)
