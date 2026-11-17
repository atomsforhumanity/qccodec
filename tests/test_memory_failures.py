"""Failures for want of memory: kind resource, with what was needed where it is known."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qcdata import CalcType, ErrorKind, ProgramError, ProgramInput
from qcdata.utils import water

from qccodec import Codec, CodecProgram, SubprocessInput
from qccodec.codec import Context
from qccodec.programs import orca, psi4, terachem


def test_a_failure_hooks_details_reach_the_error(tmp_path: Path) -> None:
    """A hook's (message, details) becomes ErrorInfo.details, classified as usual."""
    codec = Codec(
        "short",
        calctypes={CalcType.energy},
        encode=lambda inp: SubprocessInput(("true",)),
        errors=[("ran out of memory", "resource", "host_memory")],
    )

    @codec.failure
    def failure(ctx: Context) -> tuple[str, dict[str, Any]]:
        return "short ran out of memory", {"required_gib": 2.0, "available_gib": 1.0}

    @codec.parser(target="energy", required=False)
    def energy(stdout: str) -> float:
        return -1.0

    inp = ProgramInput(program="short", calctype=CalcType.energy, structure=water)
    with pytest.raises(ProgramError) as e:
        CodecProgram(codec, executable="/bin/true").compute(inp)
    error = e.value.error
    assert (error.kind, error.code) == (ErrorKind.resource, "host_memory")
    assert error.details == {"required_gib": 2.0, "available_gib": 1.0}


def test_orca_scales_its_per_process_figures_by_the_processes() -> None:
    stdout = (
        "           *        Program running with 4 parallel MPI-processes     *\n"
        "Current MaxCore: 100 MB -> Minimum MaxCore: 300 MB\n"
    )
    message, details = orca.out_of_memory(stdout, "MP2")  # type: ignore[misc]
    assert "300 MB per process" in message and "(4 processes)" in message
    assert details == {"required_gib": 1200 / 1024, "available_gib": 400 / 1024}
    assert orca.out_of_memory("Error (ORCA_SCF): SCF failed\n", "SCF") is None


def test_psi4_reads_a_density_fitting_need_against_its_memory() -> None:
    stdout = (
        "    Memory:     2.0 GiB\n"
        "  DFTensor Memory: Qso construction needs 3.500 GiB; user supplied 1.500 GiB.\n"
    )
    assert psi4.memory_details(stdout) == {"required_gib": 4.0, "available_gib": 2.0}
    assert psi4.memory_details("    Memory:     2.0 GiB\n") == {}


@pytest.mark.parametrize(
    ("stdout", "code"),
    [
        ("CUDA error: out of memory, file x.cu, line 3\n", "device_memory"),
        ("GPUBox: Failed to allocate GPU memory on device 0\n", "device_memory"),
        ("GPUbox: Allocation oversubscribes host buffer.\n", "host_memory"),
    ],
)
def test_terachem_out_of_memory_without_figures(stdout: str, code: str) -> None:
    message, details = terachem.out_of_memory(stdout)  # type: ignore[misc]
    assert terachem.codec.classify(message) == ("resource", code)
    assert details == {}
