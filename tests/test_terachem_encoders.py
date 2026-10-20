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


def test_constraints_become_freeze_and_set_groups(prog_input_factory):
    """geomeTRIC-style constraints are written 1-based, distances in angstrom."""
    inp = prog_input_factory("optimization")
    inp = inp.model_copy(
        update={
            "keywords": {
                "new_minimizer": "yes",
                "constraints": {
                    "freeze": [
                        {"type": "xyz", "indices": [0, 2]},
                        {"type": "angle", "indices": [1, 0, 2]},
                    ],
                    "set": [
                        {"type": "distance", "indices": [0, 1], "value": 2.0},
                        {"type": "dihedral", "indices": [0, 1, 2, 0], "value": 120},
                    ],
                },
            }
        }
    )
    text = encode(inp).files["tc.in"]
    tail = text.split("new_minimizer")[1].splitlines()[1:]
    assert tail == [
        "end",
        "",
        "$constraint_freeze",
        "xyz 1,3",
        "angle 2_1_3",
        "$end",
        "$constraint_set",
        "bond 1.058354421 1_2",
        "dihedral 120 1_2_3_1",
        "$end",
    ]


@pytest.mark.parametrize(
    ("constraints", "match"),
    [
        ({"scan": []}, "scans are not supported"),
        ({"freeze": [{"type": "position", "indices": [0]}]}, "Unknown constraint"),
        ({"freeze": [{"type": "distance", "indices": [0]}]}, "takes 2"),
        ({"set": [{"type": "distance", "indices": [0, 1]}]}, "needs"),
    ],
)
def test_invalid_constraints(prog_input_factory, constraints, match):
    """Unsupported groups, types, atom counts, and missing values are rejected."""
    inp = prog_input_factory("optimization")
    inp = inp.model_copy(
        update={"keywords": {"new_minimizer": "yes", "constraints": constraints}}
    )
    with pytest.raises(EncoderError, match=match):
        encode(inp)


def test_constraints_only_for_optimizations(prog_input_factory):
    """Single points have nothing to constrain."""
    inp = prog_input_factory("gradient")
    inp = inp.model_copy(
        update={
            "keywords": {"constraints": {"freeze": [{"type": "xyz", "indices": [0]}]}}
        }
    )
    with pytest.raises(EncoderError, match="optimizations only"):
        encode(inp)
