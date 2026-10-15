"""Run every example script exactly as a user would: ``python examples/<program>/<x>.py``."""

import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import requires

EXAMPLES = Path(__file__).parent.parent.parent / "examples"

# Programs each example directory needs on PATH; geomeTRIC examples use CREST gradients.
NEEDS = {"geometric": ("crest",)}
# The tcpb clients need a running TeraChem server and a tcpb release built against
# current qcdata; neither exists in CI yet.
NEEDS_SERVER = {"terachem-fe", "terachem-pbs"}

SCRIPTS = sorted(EXAMPLES.rglob("*.py"))


def _marks(script: Path) -> list[pytest.MarkDecorator]:
    program = script.parent.name
    if program in NEEDS_SERVER:
        return [pytest.mark.skip(reason="needs a running TeraChem server")]
    return [requires(*NEEDS.get(program, (program,)))]


@pytest.mark.integration
@pytest.mark.parametrize(
    "script",
    [
        pytest.param(s, marks=_marks(s), id=s.relative_to(EXAMPLES).as_posix())
        for s in SCRIPTS
    ],
)
def test_example_runs(script: Path) -> None:
    """The script runs to completion with its default settings."""
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
