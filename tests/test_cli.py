import subprocess
import sys

from qccodec.codec import decode


def test_cli(test_data_dir):
    # Call CLI script as a subprocess
    filepath = test_data_dir / "terachem" / "water.energy.out"
    sp_proc = subprocess.run(
        [sys.executable, "-m", "qccodec.cli", "terachem", "energy", filepath],
        capture_output=True,
        text=True,
    )
    # Check the return code
    assert sp_proc.returncode == 0

    # Check the output
    parse_data = decode("terachem", "energy", stdout=filepath.read_text())
    expected_output = parse_data.model_dump_json(
        indent=4, exclude_unset=True, exclude_defaults=True
    )
    assert sp_proc.stdout.strip() == expected_output


def test_cli_failed_without_artifacts():
    from qcdata import OptimizationData

    process = subprocess.run(
        [sys.executable, "-m", "qccodec.cli", "crest", "optimization", "--failed"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    data = OptimizationData.model_validate_json(process.stdout)
    assert data.provenance.program == "crest"
    assert data.trajectory == []
