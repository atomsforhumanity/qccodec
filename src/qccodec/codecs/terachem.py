"""Complete terachem invocations and decode their native artifacts."""

from __future__ import annotations

from qcdata import CalcType, ProgramInput, StructuredData

from ..definition import CodecDefinition
from ..encoders import terachem as encoder
from ..exceptions import CalculationError, EncoderError
from ..native import NativeOutput
from ..parsers.terachem import codec


def decode(
    definition: CodecDefinition, inp: ProgramInput, out: NativeOutput
) -> StructuredData:
    """Interpret native observations and preserve recoverable scientific data."""
    directory = (
        next(iter(sorted(out.directory.glob("scr*"))), None) if out.directory else None
    )
    missing = directory is None and not out.failed
    data = definition.decode_artifacts(
        inp.calctype,
        stdout=out.stdout,
        directory=directory,
        input_data=inp,
        failed=out.failed or missing,
    )
    if missing:
        raise CalculationError(
            "TeraChem did not create a scratch output directory.", data=data
        )
    return data


def restart(inp: ProgramInput, previous: StructuredData) -> ProgramInput:
    """Bind retained orbital files as the next calculation's initial guess."""
    matches = {
        name.rsplit("/", 1)[-1]: value
        for name, value in previous.files.items()
        if name.rsplit("/", 1)[-1] in {"c0", "ca0", "cb0"}
    }
    names = ["c0"] if "c0" in matches else ["ca0", "cb0"]
    if not all(name in matches for name in names):
        raise EncoderError("Could not find c0 or ca0/cb0 restart files.")
    values = inp.model_dump()
    values["files"].update({name: matches[name] for name in names})
    values["keywords"]["guess"] = " ".join(names)
    return ProgramInput.model_validate(values)


codec.encoder = encoder.encode
codec.decoder = decode
codec.restart = restart
codec.supported_calctypes = frozenset(
    {CalcType.energy, CalcType.gradient, CalcType.hessian, CalcType.optimization}
)
