"""Conventional lazy discovery. External codecs can be passed directly to compute."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ..definition import CodecDefinition
from ..exceptions import CodecNotFoundError, EncoderError

_MODULES = {name: name for name in ("orca", "terachem", "crest", "xtb", "geometric")}
_MODULES.update({"terachem-fe": "tcpb", "terachem-pbs": "tcpb"})


def supported_programs() -> tuple[str, ...]:
    """List built-in integration names independently of import history."""
    return tuple(_MODULES)


def get_codec(
    program: str, *, qcng_fallback: bool = False, **options: Any
) -> CodecDefinition:
    """Resolve a built-in integration lazily, optionally delegating to QCEngine."""
    factory: Callable[..., CodecDefinition] | None
    args: tuple[str, ...]
    module = _MODULES.get(program)
    if module is None:
        if not qcng_fallback:
            raise CodecNotFoundError(f"No codec found for program '{program}'.")
        from .qcengine import make_codec

        factory = make_codec
        args = (program,)
    else:
        loaded = import_module(f"qccodec.codecs.{module}")
        factory = getattr(loaded, "make_codec", None)
        args = (program,) if module == "tcpb" else ()
        if factory is None:
            if options:
                raise EncoderError(
                    f"{program} does not accept codec options: {sorted(options)}"
                )
            return loaded.codec
    assert factory is not None
    try:
        return factory(*args, **options)
    except TypeError as exc:
        raise EncoderError(f"Invalid {program} codec options: {exc}") from exc
