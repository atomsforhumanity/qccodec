"""Normal modes follow qcdata's definition in every codec.

Each mode is a unit-length Cartesian displacement in the input's frame and atom
order; with a Hessian, it is the mass-weighted Hessian's eigenvector at the reported
frequency.
"""

from __future__ import annotations

import numpy as np
import pytest

from qccodec import get_codec
from qccodec.exceptions import ParserError
from qccodec.programs._common import rotation_onto
from qccodec.testing import Case, cases, check_normal_modes

from .test_cases import DATA

WITH_MODES = [
    case
    for case in cases(DATA)
    if getattr(case.expected, "normal_modes_cartesian", None) is not None
]


def test_some_cases_have_modes() -> None:
    """Every codec that reports modes has a recorded case for this test."""
    programs = {case.input_data.program for case in WITH_MODES}
    assert {"orca", "psi4", "xtb", "crest", "terachem"} <= programs, programs


@pytest.mark.parametrize("case", WITH_MODES, ids=str)
def test_recorded_modes_follow_qcdata(case: Case) -> None:
    """Unit length, one per frequency, and eigenvectors of the mass-weighted Hessian."""
    data = get_codec(case.input_data.program).parse(
        case.input_data.calctype,
        stdout=case.stdout,
        directory=case.directory,
        input_data=case.input_data,
        failed=case.failed,
    )
    check_normal_modes(data, list(case.input_data.require_structure().symbols))


def test_rotation_onto_recovers_a_rotation() -> None:
    """A rotated, shifted copy aligns back exactly; a different structure is refused."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(5, 3))
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = q * np.sign(np.diag(r))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    source = (target - target.mean(axis=0)) @ rotation.T + [1.0, -2.0, 0.5]
    found = rotation_onto(source, target)
    assert np.allclose(found @ rotation, np.eye(3), atol=1e-10)
    with pytest.raises(ParserError, match="differ from the input structure"):
        rotation_onto(source + rng.normal(scale=0.1, size=source.shape), target)
