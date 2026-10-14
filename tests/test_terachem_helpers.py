"""TeraChem helper functions that are not registered parsers."""

import pytest
from qcdata import CalcType

from qccodec.exceptions import MatchNotFoundError
from qccodec.programs.terachem import (
    calculation_succeeded,
    parse_calctype,
    parse_excited_states,
    parse_version_control_details,
)


def test_parse_git_commit(terachem_file):
    contents = terachem_file("water_energy")
    assert (
        parse_version_control_details(contents)
        == "4daa16dd21e78d64be5415f7663c3d7c2785203c"  # pragma: allowlist secret
    )


def test_parse_excited_states_raises_exception_no_excited_states(terachem_file):
    with pytest.raises(MatchNotFoundError):
        parse_excited_states(terachem_file("water_energy"))


@pytest.mark.parametrize(
    "case,calctype",
    (
        ("water_energy", CalcType.energy),
        ("water_gradient", CalcType.gradient),
        ("water_hessian", CalcType.hessian),
    ),
)
def test_parse_calctype(terachem_file, case, calctype):
    assert parse_calctype(terachem_file(case)) == calctype


def test_parse_calctype_raises_exception():
    with pytest.raises(MatchNotFoundError):
        parse_calctype("No driver here")


def test_calculation_succeeded(terachem_file):
    assert calculation_succeeded(terachem_file("water_energy")) is True
    assert (
        calculation_succeeded(
            """
        Incorrect purify value
        DIE called at line number 3572 in file terachem/params.cpp
         Job terminated: Thu Mar 23 03:47:12 2023
        """
        )
        is False
    )


@pytest.mark.parametrize("case", ["failure_nocuda", "failure_basis"])
def test_calculation_succeeded_cuda_failure(terachem_file, case):
    assert calculation_succeeded(terachem_file(case)) is False
