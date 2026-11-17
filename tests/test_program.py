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

from qccodec import Codec, CodecProgram, SubprocessInput, re_search
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

codec = Codec("fake", calctypes={CalcType.energy})


@codec.encoder
def _encode(inp: ProgramInput) -> SubprocessInput:
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


def test_failures_raise_with_the_output_and_logs(tmp_path: Path) -> None:
    with pytest.raises(ProgramError) as e:
        program.compute(inp(mode="fail"), ctx(tmp_path))
    assert e.value.error.kind is ErrorKind.internal
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


def test_numerics_is_the_executables_digest(tmp_path: Path) -> None:
    import hashlib

    script = tmp_path / "prog"
    script.write_text("#!/bin/sh\necho 'prog version 6.7.1'\n")
    script.chmod(0o755)
    expected = "sha256:" + hashlib.sha256(script.read_bytes()).hexdigest()
    assert executable_digest(str(script)) == expected
    assert CodecProgram(codec, executable=str(script)).numerics(inp()) == expected
    assert CodecProgram(codec, library="numpy").numerics(inp()).startswith("numpy ")
    assert CodecProgram(codec, executable="/no/such").numerics(None) == "not installed"
