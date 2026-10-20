"""DFTB+ encoding: Hamiltonians from qcref parameter sets, HSD keywords, and errors."""

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from qccodec import get_codec
from qccodec.exceptions import CalculationError, EncoderError
from qccodec.native import SubprocessInput
from qccodec.programs.dftbplus import Typed, parse_results_tag, render

WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array([[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]),
)
SKF = "/opt/dftb-params"


def _encode(
    method: str = "DFTB3",
    basis: str | None = "3ob-3-1",
    calctype: CalcType = CalcType.gradient,
    structure: Structure = WATER,
    **kwargs: object,
) -> SubprocessInput:
    keywords = dict(kwargs.pop("keywords", {}))  # type: ignore[call-overload]
    if basis and "slater_koster_dir" not in keywords:
        keywords["slater_koster_dir"] = SKF
    inp = ProgramInput(
        program="dftbplus",
        calctype=calctype,
        structure=structure,
        model=Model(method=method, basis=basis),
        keywords=keywords,
        **kwargs,  # type: ignore[arg-type]
    )
    native = get_codec("dftbplus").encode(inp)
    assert isinstance(native, SubprocessInput)
    return native


def _hsd(**kwargs: object) -> str:
    return str(_encode(**kwargs).files["dftb_in.hsd"])  # type: ignore[arg-type]


def test_dftb3_uses_the_3ob_settings_from_qcref() -> None:
    """Hubbard derivatives, damping, and angular momenta come from the parameter set."""
    hsd = _hsd()
    assert "Hamiltonian = DFTB {" in hsd
    assert "ThirdOrderFull = Yes" in hsd
    assert "O = -0.1575" in hsd and "H = -0.1857" in hsd
    assert "HCorrection = Damping {\n    Exponent = 4.0\n  }" in hsd
    assert 'O = "p"' in hsd and 'H = "s"' in hsd
    assert f'Prefix = "{SKF}/3ob-3-1/"' in hsd
    assert "PrintForces = Yes" in hsd
    assert "ParserVersion = 14" in hsd


def test_scc_dftb_with_mio_has_no_third_order() -> None:
    """mio-1-1 is an SCC-DFTB set: no Hubbard derivatives or damping."""
    hsd = _hsd(method="SCC-DFTB", basis="mio-1-1", calctype=CalcType.energy)
    assert "Scc = Yes" in hsd and "ThirdOrder" not in hsd and "HCorrection" not in hsd
    assert "PrintForces" not in hsd


def test_xtb_needs_no_parameter_files() -> None:
    """DFTB+'s xTB Hamiltonian takes only the method and charge."""
    hsd = _hsd(method="gfn2-xtb", basis=None)
    assert 'Hamiltonian = xTB {\n  Method = "GFN2-xTB"\n  Charge = 0\n}' in hsd


def test_command_and_cmdline_args() -> None:
    """dftb+ reads dftb_in.hsd from the calculation directory."""
    assert _encode(cmdline_args=["--extra"]).command == ("dftb+", "--extra")


def test_hessian_and_optimization_drivers() -> None:
    """Hessians use SecondDerivatives with a tight SCC; optimizations LBFGS."""
    hessian = _hsd(calctype=CalcType.hessian)
    assert (
        "Driver = SecondDerivatives {" in hessian and "SCCTolerance = 1e-08" in hessian
    )
    optimization = _hsd(
        calctype=CalcType.optimization,
        keywords={"Driver": {"MaxSteps": 50, "Convergence": {"GradElem": 1e-5}}},
    )
    assert "Driver = GeometryOptimisation {" in optimization
    assert "Optimiser = Lbfgs {}" in optimization
    assert "MaxSteps = 50" in optimization and "GradElem = 1e-05" in optimization
    assert 'OutputPrefix = "geo_end"' in optimization


def test_keywords_merge_as_hsd() -> None:
    """Dicts merge into blocks; strings are verbatim HSD; lists are space-separated."""
    hsd = _hsd(
        keywords={
            "Hamiltonian": {
                "MaxSCCIterations": 500,
                "Dispersion": "DftD3 {\n    s6 = 1.0\n    s8 = 3.209\n  }",
                "ElectricField": {"External": {"Direction": [0, 0, 1]}},
            },
            "Options": {"WriteDetailedOut": False},
        }
    )
    assert "MaxSCCIterations = 500" in hsd
    assert "Dispersion = DftD3 {\n    s6 = 1.0\n    s8 = 3.209\n  }" in hsd
    assert "Direction = 0 0 1" in hsd
    assert "WriteResultsTag = Yes\n  WriteDetailedOut = No" in hsd


def test_open_shells_add_spin_polarisation() -> None:
    """Multiplicity becomes the number of unpaired electrons."""
    radical = Structure(
        symbols=["O", "H"],
        geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]]),
        multiplicity=2,
    )
    hsd = _hsd(structure=radical)
    assert "SpinPolarisation = Colinear {\n    UnpairedElectrons = 1\n  }" in hsd


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "PM7"}, "Unknown DFTB\\+ method"),
        ({"basis": None}, "needs a Slater-Koster parameter set"),
        ({"basis": "pbc-0-3"}, "Unknown DFTB parameter set"),
        ({"method": "DFTB3", "basis": "mio-1-1"}, "no DFTB3 Hubbard derivatives"),
        ({"method": "GFN2-xTB", "basis": "3ob-3-1"}, "needs no parameter set"),
        ({"keywords": {"Geometry": "GenFormat {}"}}, "Geometry comes from structure"),
        ({"keywords": {"Hamiltonian": {"Charge": 1}}}, "structure.charge"),
    ],
)
def test_rejected_inputs(kwargs: dict, match: str) -> None:
    """Unknown methods and sets, wrong pairings, and structured data in keywords."""
    with pytest.raises(EncoderError, match=match):
        _encode(**kwargs)


def test_elements_outside_the_set() -> None:
    """mio-1-1 has no Zn."""
    zinc = Structure(symbols=["Zn"], geometry=np.zeros((1, 3)))
    with pytest.raises(EncoderError, match="no parameters for \\['Zn'\\]"):
        _encode(method="SCC-DFTB", basis="mio-1-1", structure=zinc)


def test_parameter_directory_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QCCODEC_DFTB_PARAMETERS stands in for the keyword; one of them is required."""
    monkeypatch.setenv("QCCODEC_DFTB_PARAMETERS", "/data/skf")
    hsd = _hsd(keywords={"slater_koster_dir": ""})
    assert 'Prefix = "/data/skf/3ob-3-1/"' in hsd
    monkeypatch.delenv("QCCODEC_DFTB_PARAMETERS")
    with pytest.raises(EncoderError, match="QCCODEC_DFTB_PARAMETERS"):
        _hsd(keywords={"slater_koster_dir": ""})


def test_render_typed_and_empty_blocks() -> None:
    """Typed nodes render as ``name = type { ... }``."""
    text = render({"A": Typed("B", {"C": True}), "D": Typed("E"), "F": {}})
    assert text == "A = B {\n  C = Yes\n}\nD = E {}\nF {\n}\n"


def test_results_tag_reader() -> None:
    """Quantities and their values; non-numeric entries are skipped."""
    tag = (
        "mermin_energy       :real:0:\n -0.407067291016514E+001\n"
        "forces              :real:2:3,2\n  0.1E+000  0.2E+000 -0.3E+000\n"
        "  0.4E+000 -0.5E+000  0.6E+000\n"
        "flag                :logical:0:\n T\n"
    )
    values = parse_results_tag(tag)
    assert values["mermin_energy"] == [-4.07067291016514]
    assert values["forces"] == [0.1, 0.2, -0.3, 0.4, -0.5, 0.6]
    assert "flag" not in values


def test_failure_names_unrecognized_input() -> None:
    """DFTB+ lists ignored input in a warning before its generic error."""
    stdout = (
        "WARNING!\n-> The following 1 node(s) have been ignored by the parser:\n(1)\n"
        "Path: dftbplusinput/Hamiltonian/DFTB/NotAnOption\nLine: 7-7\n\n"
        "ERROR!\n-> Code halting due to the presence of errors in dftb_in file.\n"
    )
    with pytest.raises(CalculationError) as error:
        get_codec("dftbplus").parse(
            CalcType.energy, stdout=stdout, failed=True, report_failure=True
        )
    assert "Code halting" in str(error.value)
    assert "Hamiltonian/DFTB/NotAnOption" in str(error.value)
