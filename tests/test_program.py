"""CodecProgram: a codec run as a program, without qccompute (a stand-in ctx)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qcdata import (
    CalcType,
    ErrorKind,
    FileInput,
    ProgramError,
    ProgramInput,
    Structure,
)

from qccodec import Codec, CodecProgram, PythonInput, SubprocessInput, re_search
from qccodec.exceptions import CalculationError, EncoderError
from qccodec.program import executable_digest

H2 = Structure(
    symbols=["H", "H"], geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
)

# A stand-in program: prints its threads and an energy, or fails, or sleeps.
SCRIPT = """
import os, sys, time
mode = open("run.inp").read().strip()
print("threads", os.environ.get("OMP_NUM_THREADS"), "gpus", os.environ.get("CUDA_VISIBLE_DEVICES"))
open("extra.txt", "w").write("made")
if mode == "fail":
    print("SCF failed to converge"); sys.exit(3)
if mode == "sleep":
    time.sleep(30)
print("Total energy: -1.5")
"""

codec = Codec(
    "fake",
    calctypes={CalcType.energy},
    errors=[(r"SCF failed", "convergence", "scf_not_converged")],
)


@codec.encoder
def _encode(inp: ProgramInput) -> SubprocessInput:
    if "typo" in inp.keywords:
        raise EncoderError("Unknown fake keyword 'typo'.", code="unknown_key")
    return SubprocessInput(
        (sys.executable, "-c", SCRIPT),
        files={"run.inp": str(inp.keywords.get("mode", "ok"))},
        threads=inp.keywords.get("threads"),  # type: ignore[arg-type]
        gpus=1,
    )


@codec.parser(target="energy")
def _energy(stdout: str) -> float:
    return float(re_search(r"Total energy:\s+(-?\d+\.\d+)", stdout).group(1))


@codec.failure
def _failure(ctx: Any) -> str | None:
    return "SCF failed" if "failed to converge" in (ctx.stdout or "") else None


codec.cap_threads = lambda inp, n: inp.model_copy(
    update={"keywords": {**inp.keywords, "threads": n}}
)
program = CodecProgram(codec, executable=sys.executable)


def inp(**keywords: Any) -> ProgramInput:
    return ProgramInput(
        program="fake", calctype=CalcType.energy, structure=H2, keywords=keywords
    )


def ctx(tmp_path: Path, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(
        threads=fields.get("threads", 4),
        gpus=fields.get("gpus", ("1", "0")),
        timeout=fields.get("timeout"),
        scratch=tmp_path,
    )


def test_compute_runs_the_command_and_keeps_its_logs(tmp_path: Path) -> None:
    out: Any = program.compute(inp(), ctx(tmp_path))
    assert out.success and out.results.energy == -1.5
    assert "threads 4 gpus 1" in str(out.logs)  # no request: the grant; one GPU of two
    assert out.execution.threads == 4
    assert out.execution.extras["executable"] == sys.executable
    asked = program.compute(inp(threads=2), ctx(tmp_path))
    assert "threads 2" in str(asked.logs) and asked.execution.threads == 2
    direct = program.compute(inp())  # no ctx: a temporary directory, no grant
    assert direct.success and "threads None" in str(direct.logs)


def test_a_request_larger_than_the_grant_is_capped(tmp_path: Path) -> None:
    out: Any = program.compute(inp(threads=8), ctx(tmp_path, threads=4))
    assert "threads 4" in str(out.logs) and out.execution.threads == 4
    assert [w.code for w in out.warnings] == ["resource_capped"]
    assert out.input_data.keywords["threads"] == 8  # the caller's input, unchanged
    within: Any = program.compute(inp(threads=2), ctx(tmp_path, threads=4))
    assert within.execution.threads == 2 and not within.warnings


def test_failures_raise_with_the_output_and_logs(tmp_path: Path) -> None:
    with pytest.raises(ProgramError) as e:
        program.compute(inp(mode="fail"), ctx(tmp_path))
    assert e.value.error.kind is ErrorKind.convergence  # the codec's errors rule
    assert e.value.error.code == "scf_not_converged"
    assert e.value.error.message == "SCF failed"  # the codec's explanation leads
    assert e.value.output is not None and "failed to converge" in str(
        e.value.output.logs
    )
    assert "exit status 3" in str(e.value.output.traceback)
    with pytest.raises(ProgramError) as e:
        program.compute(inp(mode="sleep"), ctx(tmp_path, timeout=1.0))
    assert e.value.error.kind is ErrorKind.timeout
    with pytest.raises(ProgramError) as e:
        program.compute(inp().model_copy(update={"calctype": CalcType.gradient}))
    assert e.value.error.kind is ErrorKind.unsupported
    missing = CodecProgram(codec, executable="/no/such/program")
    with pytest.raises(ProgramError) as e:
        missing.compute(inp())
    assert e.value.error.kind is ErrorKind.environment
    with pytest.raises(ProgramError) as e:
        program.compute(inp(typo=1))
    assert (e.value.error.kind, e.value.error.code) == ("input", "unknown_key")


def _raise(message: str) -> None:
    raise RuntimeError(message)


def _decode_library_failure(inp: Any, out: Any) -> Any:
    raise CalculationError(f"library failed: {out.error}")


library = Codec(
    "library",
    calctypes={CalcType.energy},
    encode=lambda inp: PythonInput(_raise, args=("SCF not converged in 1 cycles",)),
    decode=_decode_library_failure,
    errors=[(r"SCF not converged", "convergence", "scf_not_converged")],
)


def test_a_libraries_exception_is_classified_by_the_codec(tmp_path: Path) -> None:
    """The codec's explanation of a library's exception leads, with its kind."""
    with pytest.raises(ProgramError) as e:
        CodecProgram(library).compute(
            ProgramInput(program="library", calctype=CalcType.energy, structure=H2),
            ctx(tmp_path),
        )
    assert (e.value.error.kind, e.value.error.code) == (
        "convergence",
        "scf_not_converged",
    )
    assert e.value.output is not None
    assert "RuntimeError: SCF not converged" in str(e.value.output.traceback)


def test_plans_declare_the_requested_resources() -> None:
    assert program.plan(inp(threads=3))["resources"] == {"threads": 3, "gpus": 1}
    assert program.plan(inp())["resources"]["threads"] == 1  # packs as one thread
    caps = program.capabilities()
    assert (
        caps["in_process"] and caps["thread_safe"] and caps["calctypes"] == ["energy"]
    )


def test_file_inputs_return_every_file_written(tmp_path: Path) -> None:
    from qccodec.programs.file import program as files

    script = "open('out.txt', 'w').write('hello'); print('done')"
    out = files.compute(
        FileInput(
            program=sys.executable,
            files={"run.py": script},
            cmdline_args=["run.py"],
        ),
        ctx(tmp_path),
    )
    assert out.success and out.results.files == {"out.txt": "hello"}
    assert "done" in str(out.logs)


def test_numerics_is_the_version_and_the_executables_digest(tmp_path: Path) -> None:
    import hashlib

    script = tmp_path / "prog"
    script.write_text("#!/bin/sh\necho 'prog version 6.7.1'\n")
    script.chmod(0o755)
    expected = "sha256:" + hashlib.sha256(script.read_bytes()).hexdigest()
    assert executable_digest(str(script)) == expected
    program = CodecProgram(codec, executable=str(script))
    assert program.numerics(inp()) == f"prog version 6.7.1 {expected}"
    import numpy

    library = CodecProgram(codec, library="numpy")
    assert library.numerics(inp()) == library.version() == numpy.__version__
    assert CodecProgram(codec, executable="/no/such").numerics(None) == "not installed"


def test_version_is_the_programs(tmp_path: Path) -> None:
    """An executable is asked with --version and read with the codec's parser."""
    from qccodec.program import executable_version

    script = tmp_path / "prog"
    script.write_text("#!/bin/sh\necho 'banner'\necho 'Total energy: -1.5 prog'\n")
    script.chmod(0o755)
    versioned = Codec("versioned", calctypes={CalcType.energy})
    versioned.parser(target=("provenance", "program_version"))(
        lambda text: re_search(r"energy: (\S+)", text).group(1)
    )
    assert executable_version(str(script), versioned) == "-1.5"
    plain = tmp_path / "plain"
    plain.write_text("#!/bin/sh\necho '1.11'\n")
    plain.chmod(0o755)
    assert CodecProgram(codec, executable=str(plain)).version() == "1.11"
    assert CodecProgram(codec, executable="/no/such").version() == "not installed"
    silent = tmp_path / "silent"
    silent.write_text("#!/bin/sh\n")
    silent.chmod(0o755)
    assert executable_version(str(silent), versioned) == "unknown"


def _logs_and_writes(name: str, *, directory: Path) -> str:
    import logging

    logging.getLogger("qccodec.test.library").info(f"working on {name}\n")
    (directory / "out.txt").write_text(name)
    return name


def test_call_python_changes_no_process_state(tmp_path: Path) -> None:
    """A function taking its directory runs as is; each thread keeps its own log."""
    import logging
    import os
    from concurrent.futures import ThreadPoolExecutor

    from qccodec.run import call_python

    library = logging.getLogger("qccodec.test.library")
    library.setLevel(logging.INFO)
    seen: list[str] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    library.addHandler(Keep())
    cwd = os.getcwd()

    def one(i: int) -> Any:
        directory = tmp_path / str(i)
        directory.mkdir()
        request = PythonInput(
            _logs_and_writes,
            args=(f"job {i}",),
            directory_keyword="directory",
            loggers=("qccodec.test.library",),
        )
        return call_python(request, directory)

    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(one, range(8)))
    for i, out in enumerate(outputs):
        assert out.value == f"job {i}" and out.stdout == f"working on job {i}\n"
        assert (tmp_path / str(i) / "out.txt").read_text() == f"job {i}"
    assert os.getcwd() == cwd and not seen  # the caller's handlers saw nothing
    library.info("after\n")
    assert seen == ["after\n"]  # outside a call, records pass through


def _on(device: str) -> dict[str, Any]:
    return {"energy": -1.0, "version": "1.0", "device": device}


def _gpu_library(device: str | None, ran_on: str) -> Codec:
    """A library that plans a GPU unless told the CPU, and reports where it ran."""
    from qccodec.programs._python import single_point_decoder

    def encode(inp: ProgramInput) -> PythonInput:
        return PythonInput(_on, args=(ran_on,), gpus=0 if device == "cpu" else 1)

    return Codec(
        "gpulib",
        calctypes={CalcType.energy},
        encode=encode,
        decode=single_point_decoder("gpulib"),
        encode_batch=lambda inputs: PythonInput(
            lambda: [_on(ran_on)] * len(inputs), gpus=1
        ),
        batch_key=lambda inp: "all",
    )


def _gpu_input(device: str | None = None) -> ProgramInput:
    keywords = {} if device is None else {"device": device}
    return ProgramInput(
        program="gpulib", calctype=CalcType.energy, structure=H2, keywords=keywords
    )


def test_a_gpu_library_on_the_cpu_warns_backend_fallback(tmp_path: Path) -> None:
    """No GPU visible (none granted): the automatic device falls back, with a warning."""
    fell_back: Any = CodecProgram(_gpu_library(None, "cpu")).compute(
        _gpu_input(), ctx(tmp_path, gpus=())
    )
    assert [w.code for w in fell_back.warnings] == ["backend_fallback"]
    assert fell_back.warnings[0].path == "device"
    assert fell_back.resolved.settings == {"device": "cpu"}
    assert fell_back.resolved.reasons == {"device": "auto: no GPU was visible"}
    on_gpu: Any = CodecProgram(_gpu_library(None, "cuda")).compute(
        _gpu_input(), ctx(tmp_path)
    )
    assert not on_gpu.warnings and on_gpu.resolved.settings == {"device": "cuda"}
    assert on_gpu.resolved.reasons == {"device": "auto: a GPU was visible"}
    asked: Any = CodecProgram(_gpu_library("cpu", "cpu")).compute(
        _gpu_input("cpu"), ctx(tmp_path)
    )
    assert not asked.warnings and asked.resolved.settings == {"device": "cpu"}
    assert asked.resolved.reasons == {}
    batched = CodecProgram(_gpu_library(None, "cpu")).compute_batch(
        [_gpu_input(), _gpu_input()], ctx(tmp_path, gpus=())
    )
    assert [[w.code for w in out.warnings] for out in batched] == [
        ["backend_fallback"]
    ] * 2
    assert all(out.resolved is not None for out in batched)
