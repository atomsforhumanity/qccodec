"""Run every example script exactly as a user would: ``python examples/<program>/<x>.py``."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import requires, requires_hugging_face, requires_modules

EXAMPLES = Path(__file__).parent.parent.parent / "examples"

# Programs each example directory needs on PATH; geomeTRIC examples use CREST gradients,
# Sella examples xtb.
NEEDS = {"geometric": ("crest",), "sella": ("xtb",), "dftbplus": ("dftb+",)}
# The tcpb clients need a running TeraChem server and a tcpb release built against
# current qcdata; neither exists in CI yet.
NEEDS_SERVER = {"terachem-fe", "terachem-pbs"}
# Library codecs need their Python modules rather than a program on PATH.
NEEDS_MODULES = {
    "tblite": ("tblite",),
    "aimnet2": ("aimnet",),
    "torchani": ("torchani",),
    "mace": ("mace",),
    "orb": ("orb_models",),
    "fairchem": ("fairchem",),
}

SCRIPTS = sorted(EXAMPLES.rglob("*.py"))


def _marks(script: Path) -> list[pytest.MarkDecorator]:
    program = script.parent.name
    if program in NEEDS_SERVER:
        return [pytest.mark.skip(reason="needs a running TeraChem server")]
    if program in NEEDS_MODULES:
        marks = [requires_modules(program, *NEEDS_MODULES[program])]
        if program == "fairchem":
            marks.append(requires_hugging_face(program))
        return marks
    marks = [requires(*NEEDS.get(program, (program,)))]
    if program == "dftbplus":
        marks.append(
            pytest.mark.skipif(
                not os.environ.get("QCCODEC_DFTB_PARAMETERS"),
                reason="QCCODEC_DFTB_PARAMETERS must hold the 3ob-3-1 parameter set",
            )
        )
    return marks


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
