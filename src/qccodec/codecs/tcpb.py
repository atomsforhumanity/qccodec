"""Optional TeraChem frontend/protobuf Python bindings."""

from __future__ import annotations

from importlib.metadata import version

from qcdata import (
    CalcType,
    ProgramInput,
    ProgramOutput,
    Provenance,
    SinglePointData,
)

from ..definition import CodecDefinition
from ..exceptions import CalculationError, EncoderError
from ..native import NativeOutput, PythonInput, PythonOutput


def encode(inp: ProgramInput) -> PythonInput:
    """Prepare native arguments and files without executing the scientific program."""
    if inp.model is None:
        raise EncoderError("TeraChem requires a scientific model.")
    return PythonInput(
        function=invoke,
        kwargs={"input_data": inp.model_copy(deep=True)},
        files=inp.files,
    )


def invoke(*, input_data: ProgramInput) -> ProgramOutput[ProgramInput, SinglePointData]:
    """Call the selected client while retaining failed outputs supplied by tcpb."""
    import tcpb

    if version("tcpb") == "0.16.0":
        raise EncoderError(
            "tcpb 0.16.0 constructs legacy qcdata outputs. A compatible tcpb release is required."
        )
    client_class = (
        tcpb.TCFrontEndClient
        if input_data.program == "terachem-fe"
        else tcpb.TCProtobufClient
    )
    try:
        with client_class() as client:
            output = client.compute(input_data)
    except tcpb.exceptions.TCPBError as exc:
        if exc.program_output is not None:
            output = exc.program_output
        else:
            raise
    print(output.logs or "", end="")
    return output


def decode(
    _definition: CodecDefinition, inp: ProgramInput, out: NativeOutput
) -> SinglePointData:
    """Interpret native observations and preserve recoverable scientific data."""
    assert isinstance(out, PythonOutput)
    if out.value is None:
        return SinglePointData(provenance=Provenance(program="terachem"))
    output = out.value
    if not output.success:
        raise CalculationError(
            output.traceback or "TeraChem client failed.", data=output.results
        )
    return output.results


def make_codec(program: str) -> CodecDefinition:
    """Select a frontend or protobuf client while preserving TeraChem producer identity."""
    return CodecDefinition(
        program,
        frozenset({CalcType.energy, CalcType.gradient}),
        encode,
        decode,
        producer_program="terachem",
    )
