"""Programs the worker tests run in a worker process, which imports them from here."""

from __future__ import annotations

import os
from typing import Any

from qcdata import CalcType

from qccodec import Codec, CodecProgram, PythonInput
from qccodec.programs._python import single_point_decoder


def report_threads() -> dict[str, Any]:
    """An energy, with the threads the loaded BLAS runs with and the process's id."""
    from threadpoolctl import threadpool_info

    blas = [i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"]
    return {
        "energy": -1.0,
        "version": "1.0",
        "extras": {"blas_threads": blas, "pid": os.getpid()},
    }


class _Here(CodecProgram):
    """A CodecProgram whose class lives here: a worker finds a registered program in
    its class's module."""


#: Runs in a worker; numpy's BLAS is loaded there before any call.
blas_threads = _Here(
    Codec(
        "blasthreads",
        calctypes={CalcType.energy},
        encode=lambda inp: PythonInput(report_threads),
        decode=single_point_decoder("blasthreads"),
    ),
    in_process=False,
)
