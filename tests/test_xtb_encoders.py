"""xtb command-line encoding: methods, keyword flags, and rejected inputs."""

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from qccodec import get_codec
from qccodec.exceptions import EncoderError
from qccodec.native import SubprocessInput

WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array([[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]),
)


def _command(
    calctype: CalcType = CalcType.gradient,
    method: str = "gfn2",
    structure: Structure = WATER,
    **kwargs: object,
) -> list[str]:
    inp = ProgramInput(
        program="xtb",
        structure=structure,
        calctype=calctype,
        model=Model(method=method),
        **kwargs,  # type: ignore[arg-type]
    )
    native = get_codec("xtb").encode(inp)
    assert isinstance(native, SubprocessInput)
    return list(native.command)


@pytest.mark.parametrize(
    ("method", "flags"),
    [
        ("gfn2", ["--gfn", "2"]),
        ("GFN1", ["--gfn", "1"]),
        ("gfn0", ["--gfn", "0"]),
        ("gfnff", ["--gfnff"]),
        ("g-xTB", ["--gxtb"]),
    ],
)
def test_methods(method: str, flags: list[str]) -> None:
    """Each method maps to its xtb flags, case- and hyphen-insensitively."""
    assert _command(method=method)[-len(flags) :] == flags


def test_charge_and_unpaired_electrons() -> None:
    """Charge and multiplicity become --chrg and --uhf."""
    cation = Structure(
        symbols=WATER.symbols, geometry=WATER.geometry, charge=1, multiplicity=2
    )
    command = _command(structure=cation)
    assert command[:6] == ["xtb", "structure.xyz", "--grad", "--chrg", "1", "--uhf"]
    assert command[6] == "1"


def test_keywords_become_flags_in_order() -> None:
    """Scalars, lists, and booleans follow xtb's own flag spelling."""
    command = _command(
        keywords={
            "alpb": ["water", "bar1M"],
            "etemp": 1000,
            "parallel": None,
            "verbose": True,
            "silent": False,
        },
        cmdline_args=["--acc", "0.1"],
    )
    tail = ["--alpb", "water", "bar1M", "--etemp", "1000", "--verbose", "--acc", "0.1"]
    assert command[-len(tail) :] == tail


def test_optimization_level() -> None:
    """The opt keyword sets the optimization level right after --opt."""
    command = _command(CalcType.optimization, keywords={"opt": "tight"})
    assert command[2:4] == ["--opt", "tight"]
    with pytest.raises(EncoderError, match="optimization level"):
        _command(CalcType.gradient, keywords={"opt": "tight"})


def test_xcontrol_file_passes_through() -> None:
    """Detailed input travels as a supplied file named by the input flag."""
    xcontrol = "$fix\n  atoms: 1\n$end\n"
    inp = ProgramInput(
        program="xtb",
        structure=WATER,
        calctype=CalcType.optimization,
        model=Model(method="gfn2"),
        keywords={"input": "xcontrol"},
        files={"xcontrol": xcontrol},
    )
    native = get_codec("xtb").encode(inp)
    assert isinstance(native, SubprocessInput)
    assert native.files["xcontrol"] == xcontrol
    assert list(native.command[-2:]) == ["--input", "xcontrol"]


@pytest.mark.parametrize("keyword", ["chrg", "uhf", "gfn", "gxtb", "hess", "md"])
def test_reserved_keywords_are_rejected(keyword: str) -> None:
    """Flags derived from the input, or that change the run type, are refused."""
    with pytest.raises(EncoderError, match=keyword):
        _command(keywords={keyword: 1})


def test_unknown_method() -> None:
    """Unknown methods list the supported ones."""
    with pytest.raises(EncoderError, match="gfn2"):
        _command(method="b3lyp")
