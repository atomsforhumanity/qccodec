"""CodecProgram: a codec run as a program, without qccompute (a stand-in ctx)."""

from __future__ import annotations

import os
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
    Model,
    ProgramError,
    ProgramInput,
    Structure,
)

from qccodec import Codec, CodecProgram, PythonInput, SubprocessInput, re_search
from qccodec.exceptions import CalculationError, EncoderError
from qccodec.program import file_digest, image_digest
from qccodec.run import run_python

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
    gpus = fields.get("gpus", ("1", "0"))
    return SimpleNamespace(
        threads=fields.get("threads", 4),
        gpus=gpus,
        timeout=fields.get("timeout"),
        scratch=tmp_path,
        # A worker's view: its CUDA_VISIBLE_DEVICES is the grant.
        device_ordinals=fields.get("device_ordinals", lambda: list(range(len(gpus)))),
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
    assert caps == {
        "calctypes": ["energy"],
        "batches": False,
        "in_process": True,
        "thread_safe": True,
    }


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
    assert file_digest(str(script)) == expected
    program = CodecProgram(codec, executable=str(script))
    assert program.numerics(inp()) == f"prog version 6.7.1 {expected}"
    import numpy

    library = CodecProgram(codec, library="numpy")
    assert library.numerics(inp()) == library.version() == numpy.__version__
    assert CodecProgram(codec, executable="/no/such").numerics(None) == "not installed"


def test_a_file_replaced_in_place_is_hashed_again(tmp_path: Path) -> None:
    import os

    path = tmp_path / "model.pt"
    path.write_bytes(b"weights 1")
    first = file_digest(path)
    path.write_bytes(b"weights 22")  # retrained: another size and time
    os.utime(path, ns=(1, 1))
    assert file_digest(path) != first
    assert file_digest(tmp_path / "missing.pt") == "not installed"


def test_an_ml_potential_is_identified_by_its_weights(tmp_path: Path) -> None:
    """The library's version and its model file's digest, when the file is on disk."""
    import hashlib

    import numpy

    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    digest = "sha256:" + hashlib.sha256(b"weights").hexdigest()
    program = CodecProgram(
        codec,
        library="numpy",
        weights=lambda spec: model if spec.model.method == "local" else None,
    )
    local = inp().model_copy(update={"model": Model(method="local")})
    named = inp().model_copy(update={"model": Model(method="not-downloaded")})
    assert program.numerics(local) == f"{numpy.__version__} {digest}"
    assert program.numerics(named) == numpy.__version__  # the version stands in
    assert program.numerics(None) == numpy.__version__


def test_a_launchers_image_identifies_the_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Docker launcher's numerics are its image's digest, not the launcher's."""
    from qccodec import program as adapter

    image_id = "sha256:" + "ab" * 32
    docker = tmp_path / "bin" / "docker"
    docker.parent.mkdir()
    docker.write_text(
        "#!/bin/sh\n"
        f'[ "$5" = "tc:1.9" ] && echo {image_id} && exit 0\n'
        "echo 'Error: No such image' >&2; exit 1\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{docker.parent}:{os.environ['PATH']}")
    monkeypatch.setattr(adapter, "_IMAGES", {})
    launcher = tmp_path / "terachem"
    launcher.write_text("#!/bin/sh\necho 'prog version 1.9'\n")
    launcher.chmod(0o755)
    assert image_digest("tc:1.9") == image_id
    assert image_digest("tc:missing") == "not installed"
    configured = CodecProgram(codec, executable=str(launcher), image="tc:1.9")
    assert configured.numerics(inp()) == f"prog version 1.9 image {image_id}"
    monkeypatch.setenv("QCCODEC_FAKE_IMAGE", "tc:1.9")
    by_variable = CodecProgram(codec, executable=str(launcher))
    assert by_variable.numerics(inp()) == f"prog version 1.9 image {image_id}"
    missing = CodecProgram(codec, executable=str(launcher), image="tc:missing")
    assert missing.numerics(inp()) == "not installed"


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


def test_a_gpu_library_on_the_cpu_warns_only_when_granted_a_gpu(tmp_path: Path) -> None:
    """Granted none, the CPU is the plan; granted one it could not use, a fallback."""
    library = CodecProgram(_gpu_library(None, "cpu"))
    fell_back: Any = library.compute(_gpu_input(), ctx(tmp_path, gpus=("1",)))
    assert [w.code for w in fell_back.warnings] == ["backend_fallback"]
    assert fell_back.warnings[0].path == "device"
    assert "granted GPU 1" in fell_back.warnings[0].message
    assert fell_back.resolved.settings == {"device": "cpu"}
    assert fell_back.resolved.reasons == {
        "device": "auto: the granted GPU was not usable"
    }
    quiets: list[Any] = [
        library.compute(_gpu_input(), ctx(tmp_path, gpus=())),
        library.compute(_gpu_input()),
    ]  # no grant: no GPU, or a direct call
    for quiet in quiets:
        assert not quiet.warnings and quiet.resolved.settings == {"device": "cpu"}
        assert quiet.resolved.reasons == {"device": "auto: no GPU was granted"}
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
    for gpus, warned in (((), []), (("0",), ["backend_fallback"])):
        batched = library.compute_batch(
            [_gpu_input(), _gpu_input()], ctx(tmp_path, gpus=gpus)
        )
        assert [[w.code for w in out.warnings] for out in batched] == [warned] * 2
        assert all(out.resolved is not None for out in batched)


def _seen_device(granted: str | None = None) -> dict[str, Any]:
    """Report the device handed in and the CUDA_VISIBLE_DEVICES the library saw."""
    return {
        "energy": -1.0,
        "version": "1.0",
        "device": granted or "auto",
        "extras": {"visible": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }


def _device_library(gpus: int = 1) -> Codec:
    from qccodec.programs._python import single_point_decoder

    return Codec(
        "gpulib",
        calctypes={CalcType.energy},
        encode=lambda inp: PythonInput(
            _seen_device, gpus=gpus, device_keyword="granted"
        ),
        decode=single_point_decoder("gpulib"),
        encode_batch=lambda inputs: PythonInput(
            lambda granted=None: [_seen_device(granted)] * len(inputs),
            gpus=gpus,
            device_keyword="granted",
        ),
        batch_key=lambda inp: "all",
    )


def test_a_library_is_handed_its_granted_gpu_as_a_cuda_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ctx.device_ordinals maps the grant; CUDA_VISIBLE_DEVICES stays as it is."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    library = CodecProgram(_device_library())
    granted: Any = library.compute(
        _gpu_input(), ctx(tmp_path, gpus=("3",), device_ordinals=lambda: [3])
    )
    assert granted.results.extras == {"visible": None}  # not narrowed in process
    assert granted.resolved.settings == {"device": "cuda"}  # the ordinal is local
    none: Any = library.compute(_gpu_input(), ctx(tmp_path, gpus=()))
    assert none.resolved.settings == {"device": "cpu"}  # the runner gave none
    direct: Any = library.compute(_gpu_input())
    assert direct.resolved.settings == {"device": "auto"}  # left to the library
    batched: Any = library.compute_batch(
        [_gpu_input(), _gpu_input()], ctx(tmp_path, gpus=("0",))
    )
    assert [o.resolved.settings for o in batched] == [{"device": "cuda"}] * 2
    on_cpu: Any = CodecProgram(_device_library(gpus=0)).compute(
        _gpu_input("cpu"), ctx(tmp_path)
    )
    assert on_cpu.resolved.settings == {"device": "auto"}  # no GPU: nothing handed
    assert on_cpu.results.extras == {"visible": None}


def test_a_granted_gpu_this_process_cannot_see_is_an_environment_failure(
    tmp_path: Path,
) -> None:
    def unseen() -> list[int]:
        raise ProgramError.of("environment", "The granted GPU 'GPU-x' is not here.")

    with pytest.raises(ProgramError) as e:
        CodecProgram(_device_library()).compute(
            _gpu_input(), ctx(tmp_path, gpus=("GPU-x",), device_ordinals=unseen)
        )
    assert e.value.error.kind is ErrorKind.environment
    assert "GPU-x" in e.value.error.message


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given, granted, or the library's choice; a granted GPU PyTorch can't use: CPU."""
    import sys

    from qccodec.programs._python import resolve_device

    cuda = SimpleNamespace(available=True)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda.available))
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert resolve_device("cuda:1", "cuda:0") == "cuda:1"  # an index is the caller's
    assert resolve_device("cuda", "cuda:2") == "cuda:2"  # a GPU: the granted one
    assert resolve_device("cuda", "cpu") == "cuda"
    assert resolve_device("cpu", "cuda:0") == "cpu"
    assert resolve_device(None, "cuda:2") == "cuda:2"
    assert resolve_device(None, "cpu") == "cpu"  # granted none, though one is visible
    assert resolve_device(None) == "cuda"  # called directly
    cuda.available = False
    assert resolve_device(None, "cuda:2") == "cpu"  # the adapter warns
    assert resolve_device(None) == "cpu"


def test_each_python_call_runs_with_its_own_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Libraries loaded before the call (BLAS, PyTorch) take each call's threads."""
    from threadpoolctl import threadpool_info

    from tests.worker_programs import report_threads

    if not any(i["user_api"] == "blas" for i in threadpool_info()):
        pytest.skip("numpy's BLAS is not one threadpoolctl controls")
    set_to: list[int] = []
    torch = SimpleNamespace(set_num_threads=set_to.append)
    monkeypatch.setitem(sys.modules, "torch", torch)
    seen = [
        run_python(PythonInput(report_threads), tmp_path, threads=n).value
        for n in (2, 3)
    ]
    assert [v["extras"]["blas_threads"][0] for v in seen] == [2, 3]
    assert set_to == [2, 3]
    unlimited = run_python(PythonInput(report_threads), tmp_path).value
    assert set_to == [2, 3]  # no threads: left alone
    assert unlimited["extras"]["blas_threads"][0] not in (2, 3) or os.cpu_count() in (
        2,
        3,
    )


def test_one_worker_serves_calls_with_different_threads() -> None:
    """A worker is pooled by program, so each call applies its own threads."""
    import qccompute
    from qccompute.program import register_program, unregister_program

    from tests.worker_programs import blas_threads

    register_program(blas_threads)
    try:
        with qccompute.Runner(threads=4) as runner:
            outs = [
                runner.compute(
                    ProgramInput(
                        program="blasthreads",
                        calctype=CalcType.energy,
                        structure=H2,
                        keywords={"threads": n},
                    )
                )
                for n in (1, 3, 2)
            ]
    finally:
        unregister_program("blasthreads")
    extras = [o.results.extras for o in outs]
    assert len({e["pid"] for e in extras}) == 1  # one worker served them all
    assert [e["blas_threads"][0] for e in extras] == [1, 3, 2]


def test_inputs_of_a_batch_run_alone_get_their_own_threads(tmp_path: Path) -> None:
    """The runner sizes a batch's grant as the sum of its inputs' threads."""
    outs: Any = program.compute_batch(
        [inp(), inp(threads=2), inp(threads=8)], ctx(tmp_path, threads=4)
    )
    assert [o.execution.threads for o in outs] == [1, 2, 4]  # planned; capped
    assert "threads 1 " in str(outs[0].logs)
    assert [w.code for w in outs[2].warnings] == ["resource_capped"]


def test_a_file_inputs_command_is_identified_by_its_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hashed as resolved on PATH; no version, since any command may run."""
    import hashlib

    from qccodec.programs.file import program as files

    with open(os.path.realpath(sys.executable), "rb") as f:
        expected = "sha256:" + hashlib.file_digest(f, "sha256").hexdigest()
    command = FileInput(program=os.path.basename(sys.executable), cmdline_args=["-V"])
    monkeypatch.setenv(
        "PATH", f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    )
    assert files.numerics(command) == expected
    missing = FileInput(program="no-such-command-anywhere")
    assert files.numerics(missing) == "not installed"


def test_numerics_inputs_are_read_once_per_process() -> None:
    from qccodec.programs._python import package_version

    package_version.cache_clear()
    library = CodecProgram(codec, library="numpy")
    for _ in range(3):
        library.numerics(inp())
    assert package_version.cache_info().misses == 1
