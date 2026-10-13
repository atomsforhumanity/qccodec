"""ORCA's complete file encoder and locally composed parsers."""

from ..encoders.orca import SUPPORTED_CALCTYPES, encode
from ..parsers.orca import codec

codec.encoder = encode
codec.supported_calctypes = frozenset(SUPPORTED_CALCTYPES)
