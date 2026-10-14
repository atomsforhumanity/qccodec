from pathlib import Path

import pytest
from qcdata import ProgramInput
from qcdata.utils import water


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Root of the fixture cases: tests/data/<program>/<case>/."""
    return Path(__file__).parent / "data"


@pytest.fixture
def terachem_file(test_data_dir):
    """Return a function that reads the stdout of a TeraChem fixture case."""

    def _read(case: str) -> str:
        return (test_data_dir / "terachem" / case / "stdout").read_text()

    return _read


@pytest.fixture(scope="session")
def prog_input_factory():
    """Create a function that returns a ProgramInput object with a specified calculation type."""

    def create_input(calctype, program="terachem"):
        return ProgramInput(
            program=program,
            structure=water,
            calctype=calctype,
            # Tests depend up this model; do not change
            model={"method": "hf", "basis": "sto-3g"},
        )

    return create_input
