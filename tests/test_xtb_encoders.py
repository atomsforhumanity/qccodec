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


POINT_CHARGES = [[-0.8, [0.0, 0.0, 5.0], 8], [0.4, [1.0, 1.0, 6.0]]]


def _native(method: str = "gfn2", **kwargs: object) -> SubprocessInput:
    inp = ProgramInput(
        program="xtb",
        structure=WATER,
        calctype=CalcType.gradient,
        model=Model(method=method),
        **kwargs,  # type: ignore[arg-type]
    )
    native = get_codec("xtb").encode(inp)
    assert isinstance(native, SubprocessInput)
    return native


def test_point_charges_write_pcharge_and_embedding() -> None:
    """external_potentials become a pcharge file and an $embedding group."""
    native = _native(keywords={"external_potentials": POINT_CHARGES})
    assert native.files["pcharge"] == (
        "2\n"
        "-0.8000000000 0.0000000000 0.0000000000 5.0000000000 8\n"
        "0.4000000000 1.0000000000 1.0000000000 6.0000000000\n"
    )
    assert "input=pcharge" in str(native.files["xcontrol"])
    assert "gradient=pcgrad" in str(native.files["xcontrol"])
    assert native.command[-2:] == ("--input", "xcontrol")


def test_point_charges_extend_a_supplied_xcontrol() -> None:
    """A user's xcontrol keeps its groups and gains the $embedding group."""
    constrain = "$constrain\n  distance: 1, 2, auto\n$end\n"
    native = _native(
        keywords={"input": "detailed.inp", "external_potentials": POINT_CHARGES},
        files={"detailed.inp": constrain},
    )
    text = str(native.files["detailed.inp"])
    assert text.startswith(constrain) and "$embedding" in text
    assert "xcontrol" not in native.files


@pytest.mark.parametrize("method", ["gfn0", "gfnff", "gxtb"])
def test_point_charges_rejected_where_xtb_ignores_them(method: str) -> None:
    """GFN0, GFN-FF, and g-xTB silently ignore $embedding, so the codec refuses."""
    with pytest.raises(EncoderError, match="ignores them"):
        _native(method=method, keywords={"external_potentials": POINT_CHARGES})


def test_malformed_point_charges() -> None:
    """Entries must be [charge, [x, y, z]] with an optional element."""
    with pytest.raises(EncoderError, match="external_potentials"):
        _native(keywords={"external_potentials": [[1.0, [0.0, 0.0]]]})
