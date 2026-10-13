from pathlib import Path

import pytest
from qcdata import CalcType, ProgramInput

from qccodec import Context, get_codec
from qccodec.programs.crest import (
    parse_conformers,
    parse_energy,
    parse_energy_numhess,
    parse_g98_freqs,
    parse_g98_normal_modes,
    parse_gradient,
    parse_numhess1,
    parse_rotamers,
    parse_trajectory,
    parse_version,
)

from .conftest import ParserTestCase, run_test_harness
from .data.crest.answers import (
    conformers,
    frequencies,
    gradients,
    hessians,
    normal_modes,
    optimizations,
)

######################################################
##### Top level tests for all registered parsers #####
######################################################


test_cases = [
    ParserTestCase(
        name="Parse version",
        parser=parse_version,
        stdout=Path("crest_stdout.txt"),
        calctype=CalcType.energy,
        success=True,
        answer="3.0.1",
    ),
    ParserTestCase(
        name="Parse version MatchNotFoundError",
        parser=parse_version,
        stdout="No version to parse",
        calctype=CalcType.energy,
        success=False,
    ),
    ParserTestCase(
        name="Parse energy CalcType.energy",
        parser=parse_energy,
        calctype=CalcType.energy,
        success=True,
        answer=-0.335557824179335,
        extra_files=["crest.engrad"],
    ),
    ParserTestCase(
        name="Parse energy CalcType.gradient",
        parser=parse_energy,
        calctype=CalcType.gradient,
        success=True,
        answer=-0.335557824179335,
        extra_files=["crest.engrad"],
    ),
    ParserTestCase(
        name="Parse gradient CalcType.gradient",
        parser=parse_gradient,
        calctype=CalcType.gradient,
        success=True,
        answer=gradients.water,
        extra_files=["crest.engrad"],
    ),
    ParserTestCase(
        name="Parse gradient CalcType.energy",
        parser=parse_gradient,
        calctype=CalcType.energy,
        success=True,
        answer=gradients.water,
        extra_files=["crest.engrad"],
    ),
    ParserTestCase(
        name="Parse energy CalcType.hessian",
        parser=parse_energy_numhess,
        stdout=Path("hessian_stdout.txt"),
        calctype=CalcType.hessian,
        success=True,
        answer=-110.490788960984773,
    ),
    ParserTestCase(
        name="Parse numhess1",
        parser=parse_numhess1,
        calctype=CalcType.hessian,
        success=True,
        answer=hessians.water,
        extra_files=["numhess1"],
    ),
    ParserTestCase(
        name="Parse frequencies HF",
        parser=parse_g98_freqs,
        calctype=CalcType.hessian,
        success=True,
        answer=frequencies.HF,
        extra_files=["g98HF.out"],
        extra_files_names=["g98.out"],
    ),
    ParserTestCase(
        name="Parse frequencies water",
        parser=parse_g98_freqs,
        calctype=CalcType.hessian,
        success=True,
        answer=frequencies.water,
        extra_files=["g98.out"],
    ),
    ParserTestCase(
        name="Parse frequencies caffeine",
        parser=parse_g98_freqs,
        calctype=CalcType.hessian,
        success=True,
        answer=frequencies.caffeine,
        extra_files=["g98big.out"],
        extra_files_names=["g98.out"],
    ),
    ParserTestCase(
        name="Parse normal_modes HF",
        parser=parse_g98_normal_modes,
        calctype=CalcType.hessian,
        success=True,
        answer=normal_modes.HF,
        extra_files=["g98HF.out"],
        extra_files_names=["g98.out"],
    ),
    ParserTestCase(
        name="Parse normal_modes water",
        parser=parse_g98_normal_modes,
        calctype=CalcType.hessian,
        success=True,
        answer=normal_modes.water,
        extra_files=["g98.out"],
    ),
    ParserTestCase(
        name="Parse normal_modes caffeine",
        parser=parse_g98_normal_modes,
        calctype=CalcType.hessian,
        success=True,
        answer=normal_modes.caffeine,
        extra_files=["g98big.out"],
        extra_files_names=["g98.out"],
    ),
    ParserTestCase(
        name="Parse conformers",
        parser=parse_conformers,
        calctype=CalcType.conformer_search,
        success=True,
        answer=conformers.conformers,
        extra_files=["crest_conformers.xyz"],
    ),
    ParserTestCase(
        name="Parse rotamers",
        parser=parse_rotamers,
        calctype=CalcType.conformer_search,
        success=True,
        answer=conformers.rotamers,
        extra_files=["crest_rotamers.xyz"],
        options={"collect_rotamers": True},
    ),
    ParserTestCase(
        name="Parse optimization",
        parser=parse_trajectory,
        stdout=Path("optstdout.txt"),
        calctype=CalcType.optimization,
        success=True,
        answer=optimizations.optimization,
        extra_files=["crest.engrad", "crestopt.log"],
    ),
]


@pytest.mark.parametrize("test_case", test_cases, ids=lambda tc: tc.name)
def test_crest_parsers(test_data_dir, prog_input_factory, tmp_path, test_case):
    """
    Tests the crest parsers to ensure that they correctly parse the output files and
    behave correctly within the decode function.
    """
    run_test_harness(test_data_dir, prog_input_factory, tmp_path, test_case)


####################################################
################ Additional Tests ##################
####################################################


def test_parse_conformers_charge_multiplicity_updates(
    test_data_dir, prog_input_factory
):
    # Change charge and multiplicity in prog_input
    prog_input = prog_input_factory("conformer_search")
    prog_input_dict = prog_input.model_dump()
    prog_input_dict["structure"]["charge"] = -2
    prog_input_dict["structure"]["multiplicity"] = 3
    # Using fake prog_input for water; need .structure.charge and .multiplicity
    ctx = Context(
        get_codec("crest"),
        CalcType.conformer_search,
        directory=test_data_dir / "crest",
        input_data=ProgramInput(**prog_input_dict),
    )
    data = parse_conformers(ctx)
    # Check conformer energies
    for struct in data["conformers"]:
        assert struct.charge == -2
        assert struct.multiplicity == 3


def test_parse_rotamers_charge_multiplicity_updates(test_data_dir, prog_input_factory):
    # Change charge and multiplicity in prog_input
    prog_input = prog_input_factory("conformer_search")
    prog_input_dict = prog_input.model_dump()
    prog_input_dict["structure"]["charge"] = 3
    prog_input_dict["structure"]["multiplicity"] = 5
    # Using fake prog_input for water; need .structure.charge and .multiplicity
    ctx = Context(
        get_codec("crest"),
        CalcType.conformer_search,
        directory=test_data_dir / "crest",
        input_data=ProgramInput(**prog_input_dict),
        options={"collect_rotamers": True},
    )
    data = parse_rotamers(ctx)
    # Check conformer energies
    for struct in data["rotamers"]:
        assert struct.charge == 3
        assert struct.multiplicity == 5
