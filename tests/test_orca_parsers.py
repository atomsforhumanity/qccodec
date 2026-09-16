from pathlib import Path

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from qccodec.parsers.orca import (
    parse_energy,
    parse_gradient,
    parse_gradient_hessian,
    parse_hessian,
    parse_natoms,
    parse_trajectory,
    parse_version,
)

from .conftest import ParserTestCase, run_test_harness
from .data.orca.answers import gradients, hessians, trajectories

######################################################
##### Top level tests for all registered parsers #####
######################################################


test_cases = [
    ParserTestCase(
        name="Parse version",
        parser=parse_version,
        stdout=Path("water.energy.out"),
        calctype=CalcType.energy,
        success=True,
        answer="6.1.0",
    ),
    ParserTestCase(
        name="Parse energy from energy log",
        parser=parse_energy,
        stdout=Path("water.energy.out"),
        calctype=CalcType.energy,
        success=True,
        answer=-76.320421659333,
    ),
    ParserTestCase(
        name="Parse energy from gradient log",
        parser=parse_energy,
        stdout=Path("water.grad.out"),
        calctype=CalcType.gradient,
        success=True,
        answer=-76.320385565717,
    ),
    ParserTestCase(
        name="Parse energy from hessian log",
        parser=parse_energy,
        stdout=Path("water.hess.out"),
        calctype=CalcType.hessian,
        success=True,
        answer=-76.320421659333,
    ),
    ParserTestCase(
        name="Parse analytic gradient",
        parser=parse_gradient,
        stdout=Path("water.grad.out"),
        calctype=CalcType.gradient,
        success=True,
        answer=gradients.water_b3lyp,
    ),
    ParserTestCase(
        name="Parse numerical gradient",
        parser=parse_gradient,
        stdout=Path("water.numgrad.out"),
        calctype=CalcType.gradient,
        success=True,
        answer=gradients.water_revdsd,
    ),
    ParserTestCase(
        name="Parse (absent) analytical gradient from hessian log",
        parser=parse_gradient_hessian,
        stdout=Path("water.hess.out"),
        calctype=CalcType.hessian,
        success=False,
        decode_exc=False,
        answer=None,
    ),
    ParserTestCase(
        name="Parse analytic hessian",
        parser=parse_hessian,
        stdout=Path("water.hess.out"),
        calctype=CalcType.hessian,
        success=True,
        answer=hessians.water_b3lyp,
        extra_files=["water.hess.hess"],
    ),
    ParserTestCase(
        name="Parse numerical hessian",
        parser=parse_hessian,
        stdout=Path("water.numhess.out"),
        calctype=CalcType.hessian,
        success=True,
        answer=hessians.water_revdsd,
        extra_files=["water.numhess.hess"],
    ),
    ParserTestCase(
        name="Parse hessian with single-column final block",
        parser=parse_hessian,
        stdout=Path("single_column.out"),
        calctype=CalcType.hessian,
        success=True,
        answer=hessians.single_column,
        extra_files=["single_column.hess"],
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
        name="Parse trajectory",
        parser=parse_trajectory,
        stdout=Path("water.opt.out"),
        calctype=CalcType.optimization,
        success=True,
        answer=trajectories.trajectory,
        clear_registry=False,
        extra_files=["water.opt_trj.xyz"],
    ),
    ParserTestCase(
        name="Parse trajectory (m=2)",
        parser=parse_trajectory,
        stdout=Path("ch3.opt.out"),
        calctype=CalcType.optimization,
        success=True,
        answer=trajectories.trajectory_ch3,
        clear_registry=False,
        program_input=ProgramInput(
            program="orca",
            structure=Structure(
                symbols=["C", "H", "H", "H"],
                geometry=np.array(
                    [
                        [
                            2.2960172429784643e-07,
                            -5.47453658675606e-07,
                            -0.13371279750931814,
                        ],
                        [1.20598062158438, 1.4745349158139784, 0.5311114980010905],
                        [-1.879974623959173, 0.3071463406992686, 0.5311113957669071],
                        [0.6739912652954723, -1.7816747322337971, 0.5311139136379973],
                    ]
                ),
                charge=0,
                multiplicity=2,
            ),
            model=Model(method="xtb"),
            calctype=CalcType.optimization,
        ),
        extra_files=["ch3.opt_trj.xyz"],
        extra_files_names=["orca_trj.xyz"],
    ),
]


@pytest.mark.parametrize("test_case", test_cases, ids=lambda tc: tc.name)
def test_orca_parsers(test_data_dir, prog_input_factory, tmp_path, test_case):
    """
    Tests the orca parsers to ensure that they correctly parse the output files and
    behave correctly within the decode function.
    """
    run_test_harness(test_data_dir, prog_input_factory, tmp_path, test_case)
