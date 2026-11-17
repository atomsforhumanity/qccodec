"""Codecs declare the CPU threads and GPUs a calculation uses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        ("orca", "hf", "sto-3g", {}, None, 0),
        ("orca", "hf", "sto-3g", {"keywords": {"pal": 8}}, 8, 0),
        ("psi4", "hf", "sto-3g", {}, None, 0),
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


@pytest.mark.parametrize(
    ("program", "method", "basis", "kwargs", "threads"),
    [
        ("xtb", "gfn2", None, {}, 1),
        ("xtb", "gfn2", None, {"keywords": {"parallel": 4}}, 4),
        ("orca", "hf", "sto-3g", {}, 1),  # parallel ORCA needs MPI: asked for only
        ("orca", "hf", "sto-3g", {"keywords": {"pal": 8}}, 8),
        ("psi4", "hf", "sto-3g", {}, None),  # scales: all the runner has
        ("psi4", "hf", "sto-3g", {"cmdline_args": ["-n", "6"]}, 6),
        ("crest", "gfn2", None, {}, None),
        ("crest", "gfn2", None, {"keywords": {"threads": 2}}, 2),
        ("terachem", "hf", "sto-3g", {}, 1),  # its work is on its GPUs
        ("tblite", "gfn2", None, {}, 1),
    ],
)
def test_plans_state_their_threads(
    program: str, method: str, basis: str | None, kwargs: dict, threads: int | None
) -> None:
    """A plan asks for the input's request, else 1, else all (programs that scale)."""
    from qccompute.program import find_program, plan_of

    inp = ProgramInput(
        program=program,
        calctype=CalcType.gradient,
        structure=WATER,
        model=Model(method=method, basis=basis),
        **kwargs,  # type: ignore[arg-type]
    )
    found = find_program(program)
    assert found is not None
    assert plan_of(found, inp).resources.threads == threads


def test_the_grant_reaches_a_program_that_scales(tmp_path: Path) -> None:
    """Without a request, Psi4 runs with what it was granted: -n <grant>."""
    from types import SimpleNamespace

    from qccodec.programs.psi4 import program

    seen: list[tuple[str, ...]] = []

    class Stop(Exception):
        pass

    def run(inp, native, ctx, directory, start, warnings):  # type: ignore[no-untyped-def]
        seen.append(native.command)
        raise Stop

    inp = ProgramInput(
        program="psi4",
        calctype=CalcType.gradient,
        structure=WATER,
        model=Model(method="hf", basis="sto-3g"),
    )
    grant = SimpleNamespace(threads=6, gpus=(), timeout=None, scratch=tmp_path)
    program._run = run  # type: ignore[method-assign]
    try:
        with pytest.raises(Stop):
            program.compute(inp, grant)
    finally:
        del program._run
    assert seen[0][-2:] == ("-n", "6")


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
    psi4 = ProgramSpec(
        program="psi4",
        calctype=CalcType.gradient,
        model=Model(method="hf", basis="sto-3g"),
    )
    asked = psi4.model_copy(update={"keywords": {"threads": 3}})
    unknown = ProgramSpec(program="no-such-program", calctype=CalcType.gradient)
    assert (optimization(orca).threads, optimization(orca).gpus) == (4, 0)
    assert (optimization(terachem).threads, optimization(terachem).gpus) == (1, 1)
    assert optimization(psi4).threads is None  # Psi4 asks for all the runner has
    assert optimization(asked).threads == 3  # the subprogram's threads keyword
    assert (optimization(unknown).threads, optimization(unknown).gpus) == (1, 0)


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


@pytest.mark.parametrize(
    ("program", "method", "basis", "flag"),
    [
        ("xtb", "gfn2", None, ("--parallel", "5")),
        ("orca", "hf", "sto-3g", "%pal nprocs 5 end"),
        ("psi4", "hf", "sto-3g", ("-n", "5")),
        ("crest", "gfn2", None, "threads = 5"),
        ("tblite", "gfn2", None, None),
    ],
)
def test_the_shared_threads_keyword_becomes_the_programs_setting(
    tmp_path: Path, program: str, method: str, basis: str | None, flag: object
) -> None:
    """keywords={"threads": n} is every program's request, in its own spelling."""
    from types import SimpleNamespace

    from qccompute.program import find_program, plan_of
    from qcdata import ProgramError

    inp = ProgramInput(
        program=program,
        calctype=CalcType.gradient,
        structure=WATER,
        model=Model(method=method, basis=basis),
        keywords={"threads": 5},
    )
    adapter = find_program(program)
    assert adapter is not None
    assert plan_of(adapter, inp).resources.threads == 5
    seen: list[tuple[Any, list[Any]]] = []

    class Stop(Exception):
        pass

    def run(inp, native, ctx, directory, start, warnings):  # type: ignore[no-untyped-def]
        seen.append((native, warnings))
        raise Stop

    grant = SimpleNamespace(threads=8, gpus=(), timeout=None, scratch=tmp_path)
    adapter._run = run  # type: ignore[attr-defined]
    try:
        with pytest.raises(Stop):
            adapter.compute(inp, grant)  # type: ignore[arg-type]
    finally:
        del adapter._run  # type: ignore[attr-defined]
    native, warnings = seen[0]
    assert native.threads == 5 and not warnings
    if isinstance(flag, tuple):
        assert native.command[-2:] == flag
    elif isinstance(flag, str):
        assert any(flag in str(text) for text in native.files.values())
    if program in ("xtb", "orca"):
        own = {"xtb": {"parallel": 2}, "orca": {"pal": 2}}[program]
        both = inp.model_copy(update={"keywords": {"threads": 5, **own}})
        with pytest.raises(ProgramError) as e:
            plan_of(adapter, both)
        assert e.value.error.code == "no_effect"
