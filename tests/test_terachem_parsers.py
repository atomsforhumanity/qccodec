from pathlib import Path

import pytest
from qcdata import CalcType, ProgramOutput

from qccodec.exceptions import MatchNotFoundError
from qccodec.parsers.terachem import (
    calculation_succeeded,
    parse_calctype,
    parse_energy,
    parse_excited_states,
    parse_gradient,
    parse_hessian,
    parse_natoms,
    parse_nmo,
    parse_trajectory,
    parse_version,
    parse_version_control_details,
)

from .conftest import ParserTestCase, run_test_harness
from .data.terachem.answers import excited_states, gradients, hessians, trajectories

######################################################
##### Top level tests for all registered parsers #####
######################################################


test_cases = [
    ParserTestCase(
        name="Parse energy",
        parser=parse_energy,
        stdout=Path("water.energy.out"),
        calctype=CalcType.energy,
        success=True,
        answer=-76.3861099088,
    ),
    ParserTestCase(
        name="Parse energy no energy",
        parser=parse_energy,
        stdout="No energy here",
        calctype=CalcType.energy,
        success=False,
    ),
    ParserTestCase(
        name="Parse energy gradient",
        parser=parse_energy,
        stdout=Path("water.gradient.out"),
        calctype=CalcType.gradient,
        success=True,
        answer=-76.3861099088,
    ),
    ParserTestCase(
        name="Parse energy hessian",
        parser=parse_energy,
        stdout=Path("water.frequencies.out"),
        calctype=CalcType.hessian,
        success=True,
        answer=-76.3861099088,
    ),
    ParserTestCase(
        name="Parse energy excited states energy",
        parser=parse_energy,
        stdout=Path("water.tddft.out"),
        calctype=CalcType.energy,
        success=True,
        answer=-76.4002287974,
    ),
    ParserTestCase(
        name="Parse energy excited states positive",
        parser=parse_energy,
        stdout="FINAL ENERGY: 124.38543379982 a.u",
        calctype=CalcType.energy,
        success=True,
        answer=124.38543379982,
    ),
    ParserTestCase(
        name="Parse energy excited states negative integer",
        parser=parse_energy,
        stdout="FINAL ENERGY: -7634 a.u",
        calctype=CalcType.energy,
        success=True,
        answer=-7634,
    ),
    ParserTestCase(
        name="Parse energy excited states positive integer",
        parser=parse_energy,
        stdout="FINAL ENERGY: 7123 a.u",
        calctype=CalcType.energy,
        success=True,
        answer=7123,
    ),
    ParserTestCase(
        name="Parse version git",
        parser=parse_version,
        stdout=Path("water.energy.out"),
        calctype=CalcType.energy,
        success=True,
        answer="v1.9-2022.03-dev [4daa16dd21e78d64be5415f7663c3d7c2785203c]",
    ),
    ParserTestCase(
        name="Parse version hg",
        parser=parse_version,
        stdout=Path("hg.out"),
        calctype=CalcType.energy,
        success=True,
        answer="v1.5K [ccdev]",
    ),
    ParserTestCase(
        name="Parse water gradient",
        parser=parse_gradient,
        stdout=Path("water.gradient.out"),
        calctype=CalcType.gradient,
        success=True,
        answer=gradients.water,
    ),
    ParserTestCase(
        name="Parse caffeine gradient",
        parser=parse_gradient,
        stdout=Path("caffeine.gradient.out"),
        calctype=CalcType.gradient,
        success=True,
        answer=gradients.caffeine,
    ),
    ParserTestCase(
        name="Parse caffeine gradient from frequencies output",
        parser=parse_gradient,
        stdout=Path("caffeine.frequencies.out"),
        calctype=CalcType.hessian,
        success=True,
        answer=gradients.caffeine_frequencies,
    ),
    ParserTestCase(
        name="Parse water hessian",
        parser=parse_hessian,
        stdout=Path("water.frequencies.out"),
        calctype=CalcType.hessian,
        success=True,
        answer=hessians.water,
    ),
    ParserTestCase(
        name="Parse caffeine hessian",
        parser=parse_hessian,
        stdout=Path("caffeine.frequencies.out"),
        calctype=CalcType.hessian,
        success=True,
        answer=hessians.caffeine,
    ),
    ParserTestCase(
        name="Parse number of atoms water",
        parser=parse_natoms,
        stdout=Path("water.energy.out"),
        calctype=CalcType.energy,
        success=True,
        answer=3,
    ),
    ParserTestCase(
        name="Parse number of atoms caffeine",
        parser=parse_natoms,
        stdout=Path("caffeine.gradient.out"),
        calctype=CalcType.gradient,
        success=True,
        answer=24,
    ),
    ParserTestCase(
        name="Parse number of MOs water",
        parser=parse_nmo,
        stdout=Path("water.energy.out"),
        calctype=CalcType.energy,
        success=True,
        answer=13,
    ),
    ParserTestCase(
        name="Parse number of MOs caffeine",
        parser=parse_nmo,
        stdout=Path("caffeine.gradient.out"),
        calctype=CalcType.gradient,
        success=True,
        answer=146,
    ),
    ParserTestCase(
        name="Parse excited states water",
        parser=parse_excited_states,
        stdout=Path("water.tddft.out"),
        calctype=CalcType.energy,
        success=True,
        answer=excited_states.water,
    ),
    ParserTestCase(
        name="Parse excited states caffeine",
        parser=parse_excited_states,
        stdout=Path("caffeine.tddft.out"),
        calctype=CalcType.energy,
        success=True,
        answer=excited_states.caffeine,
    ),
    ParserTestCase(
        name="Parse excited states not found",
        parser=parse_excited_states,
        stdout=Path("water.energy.out"),
        calctype=CalcType.energy,
        decode_exc=False,
        success=False,
    ),
    ParserTestCase(
        name="Parse trajectory",
        parser=parse_trajectory,
        stdout=Path("water.opt.out"),
        calctype=CalcType.optimization,
        success=True,
        answer=trajectories.trajectory,
        clear_registry=False,
        extra_files=["optim.xyz"],
        program_input=trajectories.trajectory_spec,
    ),
    ParserTestCase(
        name="Parse trajectory charge and multiplicity",
        parser=parse_trajectory,
        stdout=Path("ch3.opt.out"),
        calctype=CalcType.optimization,
        success=True,
        answer=trajectories.ch3_trajectory,
        clear_registry=False,
        extra_files=["ch3optim.xyz"],
        extra_files_names=["optim.xyz"],
        program_input=trajectories.ch3_trajectory_spec,
    ),
    ParserTestCase(
        name="Parse trajectory MatchNotFound",
        parser=parse_trajectory,
        stdout=Path("water.energy.out"),
        calctype=CalcType.optimization,
        success=False,
        clear_registry=False,
        extra_files=["optim.xyz"],
        program_input=trajectories.trajectory_spec,
    ),
    ParserTestCase(
        name="Parse trajectory excited state",
        parser=parse_trajectory,
        stdout=Path("excited-state-opt.out"),
        calctype=CalcType.optimization,
        success=True,
        answer=trajectories.es_trajectory,
        clear_registry=False,
        extra_files=["es_optim.xyz"],
        extra_files_names=["optim.xyz"],
    ),
]


@pytest.mark.parametrize("test_case", test_cases, ids=lambda tc: tc.name)
def test_terachem_parsers(test_data_dir, prog_input_factory, tmp_path, test_case):
    """
    Tests the terachem parsers to ensure that they correctly parse the output files and
    behave correctly within the decode function.
    """
    # Water and excited state successful trajectories
    if test_case.success and "Parse trajectory" in test_case.name:
        # Update scratch_dir to the tmp_path for the trajectory test case
        for i in range(len(test_case.answer)):
            po_dict = test_case.answer[i].model_dump()
            po_dict["execution"]["scratch_dir"] = tmp_path
            test_case.answer[i] = ProgramOutput(**po_dict)

    run_test_harness(test_data_dir, prog_input_factory, tmp_path, test_case)


####################################################
##### Test for non registered parser functions #####
####################################################


def test_parse_git_commit(terachem_file):
    contents = terachem_file("water.energy.out")
    assert (
        parse_version_control_details(contents)
        == "4daa16dd21e78d64be5415f7663c3d7c2785203c"  # pragma: allowlist secret
    )


def test_parse_excited_states_raises_exception_no_excited_states(terachem_file):
    """
    Tests the parse_excited_states function to ensure that it correctly raises
    an exception when no excited states are found in the output file.
    """
    contents = terachem_file("water.energy.out")

    with pytest.raises(MatchNotFoundError):
        parse_excited_states(contents)


###############################################
##### Unused parsers, but keeping for now #####
###############################################


@pytest.mark.parametrize(
    "filename,calctype",
    (
        ("water.energy.out", CalcType.energy),
        ("water.gradient.out", CalcType.gradient),
        ("water.frequencies.out", CalcType.hessian),
    ),
)
def test_parse_calctype(terachem_file, filename, calctype):
    contents = terachem_file(filename)
    assert parse_calctype(contents) == calctype


def test_parse_calctype_raises_exception():
    with pytest.raises(MatchNotFoundError):
        parse_calctype("No driver here")


def test_calculation_succeeded(terachem_file):
    contents = terachem_file("water.energy.out")
    assert calculation_succeeded(contents) is True
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


@pytest.mark.parametrize(
    "filename,result",
    (
        ("failure.nocuda.out", False),
        ("failure.basis.out", False),
    ),
)
def test_calculation_succeeded_cuda_failure(terachem_file, filename, result):
    contents = terachem_file(filename)
    assert calculation_succeeded(contents) is result
