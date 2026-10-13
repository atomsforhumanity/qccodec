"""Optional composition of small parsers into a self-contained codec."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

from qcdata import CalcType, ProgramInput, StructuredData

from .exceptions import EncoderError, RegistryError
from .native import NativeInput, NativeOutput
from .registry import ParserSpec

Parser = TypeVar("Parser", bound=Callable[..., Any])


@dataclass
class CodecDefinition:
    """Compose local parser specifications with native input and output translation."""

    program: str
    supported_calctypes: frozenset[CalcType] = frozenset()
    encoder: Callable[[ProgramInput], NativeInput] | None = None
    decoder: (
        Callable[[CodecDefinition, ProgramInput, NativeOutput], StructuredData] | None
    ) = None
    iter_files: (
        Callable[[str | None, str | Path | None], Iterable[tuple[Enum | str, Any]]]
        | None
    ) = None
    parsers: list[ParserSpec] = field(default_factory=list)
    restart: Callable[[ProgramInput, StructuredData], ProgramInput] | None = None
    producer_program: str | None = None

    def encode(self, input_data: ProgramInput) -> NativeInput:
        """Validate supported calculation types before preparing native input."""
        if input_data.program != self.program:
            raise EncoderError(
                f"Codec {self.program!r} cannot encode {input_data.program!r}."
            )
        if input_data.calctype not in self.supported_calctypes:
            raise EncoderError(
                f"{self.program} does not support {input_data.calctype.value}."
            )
        if self.encoder is None:
            raise EncoderError(f"No encoder for {self.program}.")
        return self.encoder(input_data)

    def decode(
        self, input_data: ProgramInput, native_output: NativeOutput
    ) -> StructuredData:
        """Run a custom decoder or compose the definition's small artifact parsers."""
        if self.decoder is None:
            data = self.decode_artifacts(
                input_data.calctype,
                stdout=native_output.stdout,
                directory=native_output.directory,
                input_data=input_data,
                failed=native_output.failed,
            )
            assert not isinstance(data, dict)
            return data
        return self.decoder(self, input_data, native_output)

    def parser(
        self,
        *,
        filetype: Enum | str,
        calctypes: list[CalcType] | None = None,
        required: bool = True,
        target: str | tuple[str, ...] | None = None,
    ) -> Callable[[Parser], Parser]:
        """Register locally; importing a module never mutates another codec."""

        def decorate(func: Parser) -> Parser:
            """Keep the original parser callable while recording its decoding metadata."""
            spec = ParserSpec(
                parser=func,
                filetype=filetype,
                calctypes=list(CalcType) if calctypes is None else calctypes,
                required=required,
                target=target,
                explicit_calctypes=calctypes is not None,
            )
            for existing in self.parsers:
                if (
                    spec.target is not None
                    and existing.target == spec.target
                    and set(spec.calctypes) & set(existing.calctypes)
                ):
                    raise RegistryError(
                        f"Duplicate parser target {spec.target!r} for overlapping calculation types."
                    )
            self.parsers.append(spec)
            return func

        return decorate

    @overload
    def decode_artifacts(
        self, calctype: CalcType | str, *, as_dict: Literal[True], **kwargs: Any
    ) -> dict[str, Any]:
        """Select dictionary or validated result output from the requested mode."""
        ...

    @overload
    def decode_artifacts(
        self,
        calctype: CalcType | str,
        *,
        as_dict: Literal[False] = False,
        **kwargs: Any,
    ) -> StructuredData:
        """Select dictionary or validated result output from the requested mode."""
        ...

    @overload
    def decode_artifacts(
        self, calctype: CalcType | str, *, as_dict: bool, **kwargs: Any
    ) -> StructuredData | dict[str, Any]:
        """Select dictionary or validated result output from the requested mode."""
        ...

    def decode_artifacts(
        self, calctype: CalcType | str, **kwargs: Any
    ) -> StructuredData | dict[str, Any]:
        """Decode saved files using this definition, without launching a program."""
        from .codec import decode_artifacts

        return decode_artifacts(self, calctype, **kwargs)
