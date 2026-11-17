"""Codecs declare the CPU threads and GPUs a calculation uses."""

from __future__ import annotations

import os

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, Structure

from qccodec import get_codec
from qccodec.native import PythonInput, SubprocessInput

WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array([[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]),
)


def _native(program: str, method: str = "gfn2", basis: str | None = None, **kw: object):
    inp = ProgramInput(
        program=program,
        calctype=CalcType.gradient,
        structure=WATER,
        model=Model(method=method, basis=basis),
        **kw,  # type: ignore[arg-type]
    )
    return get_codec(program).encode(inp)


@pytest.mark.parametrize(
    ("program", "method", "basis", "kwargs", "threads", "gpus"),
    [
        ("xtb", "gfn2", None, {}, None, 0),  # no request: the runner's grant
        ("xtb", "gfn2", None, {"keywords": {"parallel": 4}}, 4, 0),
        ("orca", "hf", "sto-3g", {}, 1, 0),
        ("orca", "hf", "sto-3g", {"keywords": {"pal": 8}}, 8, 0),
        ("psi4", "hf", "sto-3g", {}, 1, 0),
        ("psi4", "hf", "sto-3g", {"cmdline_args": ["-n", "6"]}, 6, 0),
        ("psi4", "hf", "sto-3g", {"cmdline_args": ["--nthread=3"]}, 3, 0),
        ("crest", "gfn2", None, {"keywords": {"threads": 2}}, 2, 0),
        ("terachem", "hf", "sto-3g", {}, None, 1),
        ("terachem", "hf", "sto-3g", {"keywords": {"gpus": "2 0 1"}}, None, 2),
    ],
)
def test_declared_resources(
    program: str,
    method: str,
    basis: str | None,
    kwargs: dict,
    threads: int | None,
    gpus: int,
) -> None:
    """Each codec reports the threads and GPUs it asked the program to use."""
    native = _native(program, method, basis, **kwargs)
    assert isinstance(native, SubprocessInput)
    assert (native.threads, native.gpus) == (threads, gpus)


def test_crest_declares_its_default_threads() -> None:
    """CREST's default thread count (all cores, at most 16) is declared too."""
    native = _native("crest")
    assert native.threads == min(os.cpu_count() or 16, 16)


def test_defaults() -> None:
    """Without a declaration, a calculation requests no threads (it takes the grant)
    and no GPU."""
    assert (SubprocessInput(("x",)).threads, SubprocessInput(("x",)).gpus) == (None, 0)
    assert (PythonInput(print).threads, PythonInput(print).gpus) == (None, 0)


@pytest.mark.parametrize("program", ["aimnet2", "torchani", "mace", "orb", "fairchem"])
@pytest.mark.parametrize(
    ("device", "gpus"), [(None, 1), ("cuda", 1), ("cpu", 0), ("cuda:1", 0)]
)
def test_ml_codecs_declare_a_gpu_unless_on_cpu(
    program: str, device: str | None, gpus: int
) -> None:
    """A model on a GPU declares one, so runners can pin it; explicit indexes opt out."""
    methods = {
        "aimnet2": "aimnet2",
        "torchani": "ANI2x",
        "mace": "mace-off-small",
        "orb": "orb-v3-conservative-omol",
        "fairchem": "uma-s-1p2p1",
    }
    keywords = {} if device is None else {"device": device}
    native = _native(program, methods[program], keywords=keywords)
    assert isinstance(native, PythonInput) and native.gpus == gpus


def test_geometric_declares_the_resources_of_its_child() -> None:
    """An optimization occupies what its gradient child declares."""
    from qcdata import ProgramSpec

    def optimization(child: ProgramSpec) -> PythonInput:
        inp = ProgramInput(
            program="geometric",
            calctype=CalcType.optimization,
            structure=WATER,
            subprograms=[child],
        )
        native = get_codec("geometric").encode(inp)
        assert isinstance(native, PythonInput)
        return native

    orca = ProgramSpec(
        program="orca",
        calctype=CalcType.gradient,
        model=Model(method="hf", basis="sto-3g"),
        keywords={"pal": 4},
    )
    terachem = ProgramSpec(
        program="terachem",
        calctype=CalcType.gradient,
        model=Model(method="hf", basis="sto-3g"),
    )
    assert (optimization(orca).threads, optimization(orca).gpus) == (4, 0)
    assert (optimization(terachem).threads, optimization(terachem).gpus) == (None, 1)


@pytest.mark.parametrize(
    ("program", "method", "basis", "kwargs"),
    [
        ("xtb", "gfn2", None, {"keywords": {"parallel": 8}}),
        ("orca", "hf", "sto-3g", {"keywords": {"pal": 8}}),
        ("psi4", "hf", "sto-3g", {"cmdline_args": ["--nthread=8"]}),
        ("psi4", "hf", "sto-3g", {"cmdline_args": ["-n", "8", "--memory", "2GB"]}),
        ("crest", "gfn2", None, {"keywords": {"threads": 8}}),
    ],
)
def test_thread_requests_can_be_capped(
    program: str, method: str, basis: str | None, kwargs: dict
) -> None:
    """A runner granting fewer threads than asked for gets them (decision: min)."""
    inp = ProgramInput(
        program=program,
        calctype=CalcType.gradient,
        structure=WATER,
        model=Model(method=method, basis=basis),
        **kwargs,
    )
    codec = get_codec(program)
    assert codec.cap_threads is not None
    capped = codec.cap_threads(inp, 3)
    assert codec.encode(capped).threads == 3
    if program == "psi4":
        assert "--memory" in capped.cmdline_args or "--memory" not in inp.cmdline_args
