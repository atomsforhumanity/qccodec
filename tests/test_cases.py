"""Every fixture case under tests/data runs through the shared harness."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from qcdata import CalcType, ProgramInput

from qccodec import get_codec, supported_programs
from qccodec.testing import Case, cases, check_codec, run_case

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("case", cases(DATA), ids=str)
def test_case(case: Case) -> None:
    """Encode, decode, and per-parser checks for one recorded program run."""
    run_case(get_codec(case.input_data.program), case)


@pytest.mark.parametrize("program", supported_programs())
def test_builtin_codecs_satisfy_the_contract(
    program: str, prog_input_factory: Callable[..., ProgramInput]
) -> None:
    """Every built-in codec passes the structural checks."""
    codec = get_codec(program)
    inputs = (
        [prog_input_factory(CalcType.energy, program)]
        if CalcType.energy in codec.calctypes
        else []
    )
    check_codec(codec, inputs=inputs)
