"""Complete crest invocations and decode their native artifacts."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from qcdata import ProgramInput, StructuredData

from ..definition import CodecDefinition
from ..encoders import crest as encoder
from ..exceptions import CalculationError
from ..native import NativeOutput
from ..parsers.crest import codec, parse_rotamers


def decode(
    definition: CodecDefinition,
    inp: ProgramInput,
    out: NativeOutput,
    *,
    collect_rotamers: bool = False,
) -> StructuredData:
    """Recover artifacts and report CREST's scientific failure marker."""
    failed = "FAILED" in out.stdout
    selected = (
        definition
        if collect_rotamers
        else replace(
            definition,
            parsers=[
                spec for spec in definition.parsers if spec.parser is not parse_rotamers
            ],
        )
    )
    data = selected.decode_artifacts(
        inp.calctype,
        stdout=out.stdout,
        directory=out.directory,
        input_data=inp,
        failed=out.failed or failed,
    )
    if failed:
        raise CalculationError(
            "CREST reported FAILED. See calculation logs.", data=data
        )
    return data


codec.encoder = encoder.encode
codec.decoder = decode
codec.supported_calctypes = frozenset(encoder.SUPPORTED_CALCTYPES)


def make_codec(*, collect_rotamers: bool = False) -> CodecDefinition:
    """Bind execution-time rotamer selection without changing standalone parsing."""
    return replace(codec, decoder=partial(decode, collect_rotamers=collect_rotamers))
