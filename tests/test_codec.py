import pytest

from qccodec.codec import decode, encode
from qccodec.encoders import terachem
from qccodec.exceptions import EncoderError, ParserError

from .data.orca.answers import hessians


def test_main_terachem_energy(terachem_file):
    """Test the main terachem energy encoder."""
    contents = terachem_file("water.energy.out")
    computed_props = decode("terachem", "energy", stdout=contents)
    assert computed_props.energy == -76.3861099088


def test_decode_raises_for_missing_required_artifact(tmp_path):
    stdout = "Version 3.0.2, test stdout"

    with pytest.raises(ParserError) as exc_info:
        decode("crest", "gradient", stdout=stdout, directory=tmp_path)

    assert "crest.engrad" in str(exc_info.value)
    assert exc_info.value.data.provenance.program_version == "3.0.2"
    assert exc_info.value.data.energy is None
    assert exc_info.value.data.gradient is None


def test_decode_does_not_require_missing_global_artifacts(tmp_path):
    (tmp_path / "crest.engrad").write_text(
        "# Energy ( Eh )\n#\n -1.0\n# Gradient ( Eh/a0 )\n#\n 0.1\n 0.2\n 0.3\n"
    )

    computed_props = decode("crest", "gradient", directory=tmp_path)

    assert computed_props.energy == -1.0
    assert computed_props.gradient.tolist() == [[0.1, 0.2, 0.3]]


def test_decode_orca_hessian_succeeds_without_cartesian_gradient(test_data_dir):
    """Analytic Hessian jobs don't print a CARTESIAN GRADIENT block in Orca stdout,
    decode() must not treat the missing gradient as a fatal error for Hessian calctype."""
    orca_dir = test_data_dir / "orca"
    stdout = (orca_dir / "water.hess.out").read_text()

    computed_props = decode("orca", "hessian", stdout=stdout, directory=orca_dir)

    assert computed_props.gradient is None
    assert computed_props.hessian.tolist() == hessians.water_b3lyp


def test_encode_raises_error_with_invalid_calctype(prog_input_factory):
    prog_input_factory = prog_input_factory(
        "transition_state", program="crest"
    )  # Not currently supported by crest encoder
    with pytest.raises(EncoderError):
        encode(prog_input_factory)


def test_main_terachem_encoder(prog_input_factory):
    prog_input_factory = prog_input_factory("energy")
    prog_input_factory.keywords.update({"purify": "no", "some-bool": False})
    native_input = encode(prog_input_factory)
    correct_tcin = (
        f"{'run':<{terachem.PADDING}} {prog_input_factory.calctype.value}\n"
        f"{'coordinates':<{terachem.PADDING}} {terachem.XYZ_FILENAME}\n"
        f"{'charge':<{terachem.PADDING}} {prog_input_factory.structure.charge}\n"
        f"{'spinmult':<{terachem.PADDING}} {prog_input_factory.structure.multiplicity}\n"
        f"{'method':<{terachem.PADDING}} {prog_input_factory.model.method}\n"
        f"{'basis':<{terachem.PADDING}} {prog_input_factory.model.basis}\n"
        f"{'purify':<{terachem.PADDING}} {prog_input_factory.keywords['purify']}\n"
        f"{'some-bool':<{terachem.PADDING}} "
        f"{str(prog_input_factory.keywords['some-bool']).lower()}\n"
    )
    assert native_input.input_file == correct_tcin


def test_parsed_provenance_survives_standalone_save(terachem_file, tmp_path):
    from qcdata import SinglePointData

    data = decode("terachem", "energy", stdout=terachem_file("water.energy.out"))
    assert data.provenance.program == "terachem"
    assert data.provenance.program_version
    path = tmp_path / "parsed.json"
    data.save(path)
    restored = SinglePointData.open(path)
    assert restored == data
    assert restored.provenance.program_version == data.provenance.program_version


@pytest.mark.parametrize("program", ["terachem", "orca", "crest"])
def test_encode_requires_model(prog_input_factory, program):
    from qcdata import ProgramInput

    input_data = ProgramInput.model_validate(
        {**prog_input_factory("energy", program=program).model_dump(), "model": None}
    )
    with pytest.raises(EncoderError, match="model"):
        encode(input_data)


@pytest.mark.parametrize("program", ["terachem", "orca", "crest"])
def test_failed_decode_without_artifacts(program, tmp_path, prog_input_factory):
    from qcdata import OptimizationData, ProgramOutput

    input_data = prog_input_factory("optimization", program=program)
    data = decode(
        program, "optimization", failed=True, directory=tmp_path, input_data=input_data
    )
    assert isinstance(data, OptimizationData)
    assert data.trajectory == []
    assert data.provenance.program == program
    assert data.provenance.program_version is None
    output = ProgramOutput(
        input_data=input_data, results=data, success=False, traceback="execution failed"
    )
    assert ProgramOutput.model_validate_json(output.model_dump_json()) == output


def test_failed_decode_recovers_energy_without_gradient(tmp_path):
    from qcdata import SinglePointData

    (tmp_path / "crest.engrad").write_text("# Energy ( Eh )\n#\n -1.0\n")
    data = decode(
        "crest",
        "gradient",
        directory=tmp_path,
        stdout="Version 3.0.2, FAILED",
        failed=True,
    )
    assert isinstance(data, SinglePointData)
    assert data.energy == -1.0
    assert data.gradient is None
    assert data.provenance.program_version == "3.0.2"
    with pytest.raises(ParserError):
        decode("crest", "gradient", directory=tmp_path)


def test_failed_decode_validates_recovered_values(tmp_path):
    from pydantic import ValidationError

    (tmp_path / "crest.engrad").write_text(
        "# Energy ( Eh )\n#\n -1.0\n# Gradient ( Eh/a0 )\n#\n 0.1\n 0.2\n"
    )
    with pytest.raises(ValidationError):
        decode("crest", "gradient", directory=tmp_path, failed=True)


@pytest.mark.parametrize("partial_artifact", [False, True])
def test_crest_trajectory_without_gradient_records_available_energy(
    tmp_path, prog_input_factory, partial_artifact
):
    from qcdata import OptimizationData

    input_data = prog_input_factory("optimization", program="crest")
    xyz_lines = input_data.structure.to_xyz().splitlines()
    xyz_lines[1] = "step -1.0"
    (tmp_path / "crestopt.log").write_text("\n".join(xyz_lines) + "\n")
    if partial_artifact:
        (tmp_path / "crest.engrad").write_text("# Energy ( Eh )\n#\n -1.0\n")
    data = decode(
        "crest",
        "optimization",
        directory=tmp_path,
        input_data=input_data,
        failed=partial_artifact,
    )
    assert isinstance(data, OptimizationData)
    entry = data.trajectory[-1]
    assert entry.success
    assert entry.input_data.calctype == "energy"
    assert entry.results.energy == -1.0
    assert entry.results.gradient is None


def test_failed_decode_continues_after_missing_required_value(tmp_path):
    (tmp_path / "crest.engrad").write_text(
        "# Gradient ( Eh/a0 )\n#\n 0.1\n 0.2\n 0.3\n"
    )
    data = decode("crest", "gradient", directory=tmp_path, failed=True)
    assert data.energy is None
    assert data.gradient.tolist() == [[0.1, 0.2, 0.3]]
