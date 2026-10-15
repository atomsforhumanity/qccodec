"""TeraChem servers through the tcpb client: ``terachem-fe`` and ``terachem-pbs``.

``terachem-fe`` talks to a TeraChem frontend server and ``terachem-pbs`` to a TeraChem
protocol-buffer server (``terachem -s <port>``), keeping TeraChem and its GPU state
running between calculations. Install the client with ``qccodec[tcpb]``.

The whole ``ProgramInput`` is handed to the tcpb client, which builds the server job
itself. ``model.method`` must be a method the server protocol knows (``hf``,
``b3lyp``, ``wb97x``, ...); unrestricted runs set ``{"restricted": False,
"closed_shell": False}`` rather than a "u" prefix; other ``keywords`` follow TeraChem's
``tc.in`` names, with values sent as text (write ``"yes"``/``"no"``, not booleans).

The client finds the server through the environment variables ``TCPB_HOST`` and
``TCPB_PORT`` (``terachem-pbs``) or ``TCPB_FRONTEND_HOST`` and ``TCPB_FRONTEND_PORT``
(``terachem-fe``), read when tcpb is first imported; nothing in ``ProgramInput`` sets
them.

Calculation types: ``energy`` and ``gradient``.

tcpb 0.16.0 builds outputs with the pre-0.19 qcdata models, so this codec refuses to
run with it; it needs a tcpb release built against current qcdata.
"""

from __future__ import annotations

from importlib.metadata import version

from qcdata import CalcType, ProgramInput, ProgramOutput, Provenance, SinglePointData

from ..codec import Codec
from ..exceptions import CalculationError, EncoderError
from ..native import NativeOutput, PythonInput, PythonOutput

CALCTYPES = frozenset({CalcType.energy, CalcType.gradient})


def encode(inp: ProgramInput) -> PythonInput:
    """Prepare the client call without importing tcpb."""
    if inp.model is None:
        raise EncoderError("TeraChem requires a scientific model.")
    return PythonInput(
        function=invoke,
        kwargs={"input_data": inp.model_copy(deep=True)},
        files=inp.files,
    )


def invoke(*, input_data: ProgramInput) -> ProgramOutput[ProgramInput, SinglePointData]:
    """Call the selected client, retaining failed outputs supplied by tcpb."""
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
        if exc.program_output is None:
            raise
        output = exc.program_output
    print(output.logs or "", end="")
    return output


def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
    """Unwrap the client's ProgramOutput; raise with its data if it failed."""
    assert isinstance(out, PythonOutput)
    if out.value is None:
        return SinglePointData(provenance=Provenance(program="terachem"))
    output = out.value
    if not output.success:
        raise CalculationError(
            output.traceback or "TeraChem client failed.", data=output.results
        )
    return output.results


codecs: dict[str, Codec] = {
    name: Codec(name, calctypes=CALCTYPES, encode=encode, decode=decode)
    for name in ("terachem-fe", "terachem-pbs")
}
