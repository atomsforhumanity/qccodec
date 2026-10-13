from importlib import metadata

__version__ = metadata.version(__name__)

from .codec import decode, encode
from .codecs import get_codec, supported_programs
from .definition import CodecDefinition
from .native import (
    Codec,
    NativeInput,
    NativeOutput,
    PythonInput,
    PythonOutput,
    SubprocessInput,
    SubprocessOutput,
)

__all__ = [
    "decode",
    "encode",
    "get_codec",
    "supported_programs",
    "Codec",
    "CodecDefinition",
    "NativeInput",
    "NativeOutput",
    "PythonInput",
    "PythonOutput",
    "SubprocessInput",
    "SubprocessOutput",
]
