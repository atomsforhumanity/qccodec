"""QCEngine compatibility: any program QCEngine supports, through its Python API."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any

from qcdata import CalcType, ProgramInput, Provenance, SinglePointData
from qcdata.qcel import from_qcel_output_results, to_qcel_input

from ..codec import Codec
from ..exceptions import CodecNotFoundError, DependencyNotFoundError, EncoderError
from ..native import NativeOutput, PythonInput, PythonOutput

CALCTYPES = frozenset({CalcType.energy, CalcType.gradient, CalcType.hessian})


def encode(inp: ProgramInput, **options: Any) -> PythonInput:
    """Prepare the QCEngine call without importing or running it."""
    if inp.model is None:
        raise EncoderError("QCEngine requires a scientific model.")
    return PythonInput(
        function=invoke,
        kwargs={
            "input_data": to_qcel_input(inp),
            "program": inp.program,
            "options": options,
        },
        files=inp.files,
    )


def invoke(
    *, input_data: dict[str, Any], program: str, options: dict[str, Any]
) -> dict[str, Any]:
    """Delegate execution to QCEngine and forward its logs to stdout/stderr."""
    from qcengine import compute, get_program
    from qcengine.exceptions import InputError, ResourceError

    try:
        get_program(program)
    except InputError as exc:
        raise CodecNotFoundError(str(exc)) from exc
    except ResourceError as exc:
        raise DependencyNotFoundError(str(exc)) from exc
    options = dict(options)
    config = dict(options.pop("task_config", {}))
    config.update(scratch_directory=str(Path.cwd()), scratch_messy=True)
    output = compute(
        input_data,
        program,
        raise_error=True,
        return_dict=True,
        task_config=config,
        **options,
    )
    print(output.get("stdout") or "", end="")
    print(output.get("stderr") or "", end="", file=sys.stderr)
    return output


def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
    """Convert QCEngine's result dictionary; empty data when invocation failed."""
    assert isinstance(out, PythonOutput)
    if out.failed:
        return SinglePointData(provenance=Provenance(program=inp.program))
    return from_qcel_output_results(out.value)


def codec_for(
    program: str,
    *,
    task_config: dict[str, Any] | None = None,
    local_options: dict[str, Any] | None = None,
) -> Codec:
    """Create a codec delegating ``program`` to QCEngine with optional settings."""
    options: dict[str, Any] = {}
    if task_config is not None:
        options["task_config"] = task_config
    if local_options is not None:
        options["local_options"] = local_options
    return Codec(
        program, calctypes=CALCTYPES, encode=partial(encode, **options), decode=decode
    )
