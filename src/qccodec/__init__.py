"""Translate qcdata inputs to native program inputs and native outputs back to qcdata."""

from importlib import metadata

__version__ = metadata.version(__name__)

from .codec import Codec, Context, ParserSpec
from .model_cache import clear_model_cache
from .native import (
    NativeInput,
    NativeOutput,
    PythonInput,
    PythonOutput,
    SubprocessInput,
    SubprocessOutput,
)
from .registry import (
    decode,
    encode,
    get_codec,
    register,
    supported_programs,
    unregister,
)
from .utils import re_finditer, re_search

__all__ = [
    "Codec",
    "Context",
    "ParserSpec",
    "NativeInput",
    "NativeOutput",
    "PythonInput",
    "PythonOutput",
    "SubprocessInput",
    "SubprocessOutput",
    "clear_model_cache",
    "decode",
    "encode",
    "get_codec",
    "register",
    "supported_programs",
    "unregister",
    "re_finditer",
    "re_search",
]
