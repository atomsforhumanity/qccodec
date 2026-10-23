"""Codecs that call a Python library in-process: encoding, decoding, unit conversion.

The libraries themselves (tblite, aimnet, torchani, mace-torch, orb-models,
fairchem-core) are exercised by the integration tests; here they are absent or faked.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, SinglePointData, Structure
from qcref import constants as c

from qccodec import PythonInput, PythonOutput, clear_model_cache, get_codec, model_cache
from qccodec.exceptions import CalculationError, EncoderError
from qccodec.programs._python import from_ev

WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array([[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]),
)
HYDROXIDE = Structure(
    symbols=["O", "H"], geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]]), charge=-1
)
MODELS = {
    "tblite": "gfn2",
    "aimnet2": "aimnet2",
    "torchani": "ANI2x",
    "mace": "mace-off-small",
    "orb": "orb-v3-conservative-omol",
    "fairchem": "uma-s-1p2p1",
}


def _input(
    program: str,
    calctype: CalcType = CalcType.gradient,
    method: str | None = None,
    structure: Structure = WATER,
    **kwargs: Any,
) -> ProgramInput:
    return ProgramInput(
        program=program,
        calctype=calctype,
        structure=structure,
        model=Model(method=method or MODELS[program]),
        **kwargs,
    )


def _encode(inp: ProgramInput) -> PythonInput:
    native = get_codec(inp.program).encode(inp)
    assert isinstance(native, PythonInput)
    return native


# ------------------------------------------------------------------ shared rules


@pytest.mark.parametrize("program", sorted(MODELS))
def test_encoding_needs_no_library(program: str) -> None:
    """Encoding builds plain arguments; the library is imported only when invoked."""
    native = _encode(_input(program))
    assert native.function.__module__ == f"qccodec.programs.{program}"


@pytest.mark.parametrize("program", sorted(MODELS))
def test_unknown_keywords_are_rejected(program: str) -> None:
    """A misspelled keyword fails instead of being ignored."""
    with pytest.raises(EncoderError, match="Unknown"):
        _encode(_input(program, keywords={"not_a_setting": 1}))


@pytest.mark.parametrize("program", sorted(MODELS))
def test_basis_sets_are_rejected(program: str) -> None:
    """These models have no basis set."""
    inp = _input(program).model_copy(
        update={"model": Model(method=MODELS[program], basis="sto-3g")}
    )
    with pytest.raises(EncoderError, match="no basis"):
        _encode(inp)


@pytest.mark.parametrize("program", sorted(MODELS))
def test_files_and_cmdline_args_are_rejected(program: str) -> None:
    """There is no command line or calculation directory input to pass them to."""
    with pytest.raises(EncoderError, match="cmdline_args"):
        _encode(_input(program, cmdline_args=["-v"]))


@pytest.mark.parametrize("program", ["aimnet2", "torchani", "mace", "orb", "fairchem"])
def test_device_keyword(program: str) -> None:
    """``device`` goes to the library; it is not a model setting."""
    assert _encode(_input(program, keywords={"device": "cuda:1"})).kwargs["device"] == (
        "cuda:1"
    )
    with pytest.raises(EncoderError, match="device"):
        _encode(_input(program, keywords={"device": 1}))


def test_from_ev_converts_to_atomic_units() -> None:
    """eV, eV/angstrom forces, and eV/angstrom^2 Hessians become Hartree and bohr."""
    forces = np.arange(6.0).reshape(2, 3)
    hessian = np.arange(36.0).reshape(2, 3, 2, 3)
    result = from_ev(2.0, forces, hessian)
    assert result["energy"] == pytest.approx(2.0 * c.EV_TO_HARTREE)
    assert np.allclose(
        result["gradient"], -forces * c.EV_TO_HARTREE / c.ANGSTROM_TO_BOHR
    )
    assert np.allclose(
        result["hessian"],
        hessian.reshape(6, 6) * c.EV_TO_HARTREE / c.ANGSTROM_TO_BOHR**2,
    )


def test_decoder_keeps_what_the_calctype_asks_for() -> None:
    """An energy drops a gradient the library happened to compute."""
    snapshot = {
        "energy": -1.0,
        "gradient": [[0.0, 0.0, 0.1]] * 3,
        "hessian": np.eye(9).tolist(),
        "atomic_charges": [-0.6, 0.3, 0.3],
        "version": "1.2.3",
    }
    codec = get_codec("aimnet2")
    out = PythonOutput(value=snapshot)
    energy = codec.decode(_input("aimnet2", CalcType.energy), out)
    assert isinstance(energy, SinglePointData)
    assert energy.gradient is None and energy.hessian is None
    assert energy.provenance.program == "aimnet2"
    assert energy.provenance.program_version == "1.2.3"
    assert np.allclose(np.asarray(energy.atomic_charges), [-0.6, 0.3, 0.3])
    gradient = codec.decode(_input("aimnet2", CalcType.gradient), out)
    assert isinstance(gradient, SinglePointData)
    assert gradient.gradient is not None and gradient.hessian is None
    hessian = codec.decode(_input("aimnet2", CalcType.hessian), out)
    assert isinstance(hessian, SinglePointData)
    assert hessian.hessian is not None


def test_decoder_reports_library_errors() -> None:
    """A library exception becomes CalculationError naming it, with empty data."""
    out = PythonOutput(error=RuntimeError("SCF did not converge"))
    with pytest.raises(
        CalculationError, match="RuntimeError: SCF did not converge"
    ) as e:
        get_codec("tblite").decode(_input("tblite"), out)
    assert isinstance(e.value.data, SinglePointData)
    assert e.value.data.provenance.program == "tblite"


def test_cached_model_loads_once() -> None:
    """A model is built on first use and shared afterwards."""
    calls: list[int] = []

    def load() -> object:
        calls.append(1)
        return object()

    key = ("test", "once")
    first = model_cache.cached_model(key, load)
    assert model_cache.cached_model(key, load) is first
    assert calls == [1]
    clear_model_cache()
    assert model_cache.cached_models() == []
    assert model_cache.cached_model(key, object) is not first


def test_model_cache_unloads_least_recently_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QCCODEC_MAX_MODELS bounds the cache; the least recently used model goes first."""
    clear_model_cache()
    monkeypatch.setenv("QCCODEC_MAX_MODELS", "2")
    for name in ("a", "b"):
        model_cache.cached_model((name,), object)
    model_cache.cached_model(("a",), object)  # a is now the most recent
    model_cache.cached_model(("c",), object)
    assert model_cache.cached_models() == [("a",), ("c",)]
    monkeypatch.setenv("QCCODEC_MAX_MODELS", "0")
    model_cache.cached_model(("d",), object)
    assert model_cache.cached_models() == []
    monkeypatch.setenv("QCCODEC_MAX_MODELS", "many")
    with pytest.raises(ValueError, match="QCCODEC_MAX_MODELS"):
        model_cache.cached_model(("e",), object)
    clear_model_cache()


# ------------------------------------------------------------------ tblite


@pytest.mark.parametrize(
    ("method", "name"),
    [("gfn2", "GFN2-xTB"), ("GFN1-xTB", "GFN1-xTB"), ("ipea1", "IPEA1-xTB")],
)
def test_tblite_methods(method: str, name: str) -> None:
    """Short and tblite-style method names both work."""
    assert _encode(_input("tblite", method=method)).kwargs["method"] == name


def test_tblite_unknown_method() -> None:
    """gfnff is xtb's force field, not a tblite method."""
    with pytest.raises(EncoderError, match="Unknown tblite method"):
        _encode(_input("tblite", method="gfnff"))


def test_tblite_settings_and_interactions() -> None:
    """Keywords split into Calculator.set settings and Calculator.add interactions."""
    native = _encode(
        _input(
            "tblite",
            structure=HYDROXIDE,
            keywords={
                "accuracy": 0.1,
                "max-iter": 500,
                "alpb-solvation": ["water", "bar1mol"],
                "electric-field": [0.0, 0.0, 0.01],
                "spin-polarization": 1.0,
            },
        )
    )
    kwargs = native.kwargs
    assert kwargs["charge"] == -1 and kwargs["uhf"] == 0
    assert kwargs["settings"] == {"accuracy": 0.1, "max-iter": 500}
    assert kwargs["interactions"] == [
        ("alpb-solvation", ["water", "bar1mol"]),
        ("electric-field", [[0.0, 0.0, 0.01]]),
        ("spin-polarization", [1.0]),
    ]
    assert np.allclose(kwargs["positions"], HYDROXIDE.geometry)  # bohr, not angstrom


def test_tblite_invoke_with_a_fake_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """The invoke function configures the calculator and returns atomic units as is."""
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class Calculator:
        def __init__(self, method: str, numbers: Any, positions: Any, **kw: Any):
            calls.append(("init", (method, list(numbers), kw["charge"], kw["uhf"])))

        def set(self, name: str, value: Any) -> None:
            calls.append(("set", (name, value)))

        def add(self, name: str, *args: Any) -> None:
            calls.append(("add", (name, *args)))

        def singlepoint(self) -> Any:
            values = {
                "energy": -5.07,
                "gradient": np.full((3, 3), 0.01),
                "charges": np.array([-0.56, 0.28, 0.28]),
                "dipole": np.array([0.1, 0.2, 0.3]),
            }
            return SimpleNamespace(get=values.__getitem__)

    module = ModuleType("tblite.interface")
    module.Calculator = Calculator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tblite", ModuleType("tblite"))
    monkeypatch.setitem(sys.modules, "tblite.interface", module)
    native = _encode(
        _input("tblite", keywords={"accuracy": 0.1, "alpb-solvation": "water"})
    )
    result = native.function(**native.kwargs)
    assert calls == [
        ("init", ("GFN2-xTB", [8, 1, 1], 0.0, 0)),
        ("set", ("accuracy", 0.1)),
        ("add", ("alpb-solvation", "water")),
    ]
    data = get_codec("tblite").decode(_input("tblite"), PythonOutput(value=result))
    assert isinstance(data, SinglePointData)
    assert data.energy == -5.07
    assert np.allclose(np.asarray(data.gradient), 0.01)
    assert np.allclose(np.asarray(data.scf_dipole_moment), [0.1, 0.2, 0.3])


# ------------------------------------------------------------------ MLIPs


def test_aimnet2_arguments() -> None:
    """AIMNet2 takes angstrom, charge, and multiplicity; Hessians switch on autograd."""
    kwargs = _encode(_input("aimnet2", CalcType.hessian, structure=HYDROXIDE)).kwargs
    assert np.allclose(kwargs["positions"], HYDROXIDE.geometry / c.ANGSTROM_TO_BOHR)
    assert kwargs["charge"] == -1 and kwargs["multiplicity"] == 1
    assert kwargs["forces"] and kwargs["hessian"] and kwargs["validate_species"]
    energy = _encode(_input("aimnet2", CalcType.energy)).kwargs
    assert not energy["forces"] and not energy["hessian"]


@pytest.mark.parametrize(
    ("method", "name"),
    [("ANI2x", "ANI2x"), ("ani-2x", "ANI2x"), ("ani1ccx", "ANI1ccx")],
)
def test_torchani_models(method: str, name: str) -> None:
    """Model names are matched without regard to case or dashes."""
    assert _encode(_input("torchani", method=method)).kwargs["model"] == name


def test_torchani_rejects_what_ani_cannot_describe() -> None:
    """ANI models are neutral closed-shell only; dtype and model names are checked."""
    with pytest.raises(EncoderError, match="neutral closed-shell"):
        _encode(_input("torchani", structure=HYDROXIDE))
    with pytest.raises(EncoderError, match="Unknown TorchANI model"):
        _encode(_input("torchani", method="ANI3"))
    with pytest.raises(EncoderError, match="dtype"):
        _encode(_input("torchani", keywords={"dtype": "float16"}))


@pytest.mark.parametrize(
    ("method", "loader", "model"),
    [
        ("mace-off", "mace_off", None),
        ("mace-off-small", "mace_off", "small"),
        ("MACE-MP-medium-mpa-0", "mace_mp", "medium-mpa-0"),
        ("mace-omol-extra_large", "mace_omol", "extra_large"),
        ("/models/mine.model", None, "/models/mine.model"),
    ],
)
def test_mace_models(method: str, loader: str | None, model: str | None) -> None:
    """A family names MACE's loader, the rest its model argument; files load directly."""
    kwargs = _encode(_input("mace", method=method)).kwargs
    assert (kwargs["loader"], kwargs["model"]) == (loader, model)


def test_mace_charge_and_spin() -> None:
    """Only MACE-OMOL reads charge and spin; the other families refuse ions."""
    with pytest.raises(EncoderError, match="neutral closed-shell"):
        _encode(_input("mace", structure=HYDROXIDE))
    kwargs = _encode(_input("mace", method="mace-omol", structure=HYDROXIDE)).kwargs
    assert kwargs["charge_and_spin"] == (-1, 1)
    with pytest.raises(EncoderError, match="Unknown MACE model"):
        _encode(_input("mace", method="mace-xyz"))


def test_orb_has_no_hessian_and_checks_precision() -> None:
    """Orb gives energies and gradients only."""
    with pytest.raises(EncoderError):
        _encode(_input("orb", CalcType.hessian))
    with pytest.raises(EncoderError, match="precision"):
        _encode(_input("orb", keywords={"precision": "float16"}))


def test_fairchem_defaults_and_settings() -> None:
    """The omol head and uncompiled inference by default; presets pass through."""
    kwargs = _encode(_input("fairchem", structure=HYDROXIDE)).kwargs
    assert kwargs["task"] is None  # the model's only head, else omol
    assert kwargs["inference_settings"] == {
        "compile": False,
        "merge_mole": False,
        "activation_checkpointing": False,
    }
    assert (kwargs["charge"], kwargs["multiplicity"]) == (-1, 1)
    turbo = _input("fairchem", keywords={"inference_settings": "turbo", "task": "omat"})
    assert _encode(turbo).kwargs["inference_settings"] == "turbo"
    with pytest.raises(EncoderError, match="inference_settings"):
        _encode(_input("fairchem", keywords={"inference_settings": 3}))


def test_fairchem_default_task() -> None:
    """Single-dataset checkpoints use their own head; UMA defaults to omol."""
    from qccodec.programs.fairchem import default_task

    assert default_task(["oc25"]) == "oc25"
    assert default_task(["omat", "omol", "oc20"]) == "omol"
    with pytest.raises(ValueError, match="task"):
        default_task(["omat", "oc20"])


# ------------------------------------------------------------------ batches


def test_codec_batch_hook_validation() -> None:
    """A codec batches only with both functions and a positive max_batch."""
    from qccodec import Codec
    from qccodec.exceptions import RegistryError

    with pytest.raises(RegistryError, match="both"):
        Codec("x", calctypes=["energy"], encode_batch=lambda inputs: None)  # type: ignore[arg-type, return-value]
    with pytest.raises(RegistryError, match="max_batch"):
        Codec("x", calctypes=["energy"], max_batch=0)
    plain = get_codec("tblite")
    assert plain.batch_key(_input("tblite")) is None
    with pytest.raises(EncoderError, match="cannot batch"):
        plain.encode_batch([_input("tblite")])


@pytest.mark.parametrize("program", ["aimnet2", "torchani", "mace", "orb", "fairchem"])
def test_ml_batch_keys(program: str) -> None:
    """Structures differ within a batch; models, keywords, and calctypes do not."""
    codec = get_codec(program)
    moved = WATER.model_copy(update={"geometry": WATER.geometry + 0.1})
    key = codec.batch_key(_input(program))
    assert key is not None
    assert codec.batch_key(_input(program, structure=moved)) == key
    assert codec.batch_key(_input(program, CalcType.energy)) != key
    assert codec.batch_key(_input(program, keywords={"device": "cpu"})) != key
    assert codec.batch_key(_input(program, CalcType.hessian)) is None
    assert codec.batch_key(_input("tblite")) is None  # another program's input


@pytest.mark.parametrize("program", ["aimnet2", "torchani", "mace", "orb", "fairchem"])
def test_ml_encode_batch(program: str) -> None:
    """A batch encodes every molecule, in order, for one call."""
    codec = get_codec(program)
    moved = WATER.model_copy(update={"geometry": WATER.geometry + 0.1})
    native = codec.encode_batch([_input(program), _input(program, structure=moved)])
    assert isinstance(native, PythonInput)
    molecules = native.kwargs["molecules"]
    assert len(molecules) == 2
    assert np.allclose(molecules[1][1], moved.geometry / c.ANGSTROM_TO_BOHR)
    assert native.gpus == 1  # the device is chosen at run time, so a GPU is declared
    with pytest.raises(EncoderError, match="share a batch key"):
        codec.encode_batch([_input(program), _input(program, CalcType.energy)])
    with pytest.raises(EncoderError, match="1 to"):
        codec.encode_batch([])
    with pytest.raises(EncoderError, match="1 to"):
        codec.encode_batch([_input(program)] * (codec.max_batch + 1))
