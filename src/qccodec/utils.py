"""Small helpers for writing regex-based parsers."""

from __future__ import annotations

import re

from .exceptions import MatchNotFoundError


def re_search(regex: str, contents: str, flags: int = 0) -> re.Match[str]:
    """Return the first match of ``regex`` in ``contents``.

    Raises:
        MatchNotFoundError: No match is found.
    """
    match = re.search(regex, contents, flags)
    if not match:
        raise MatchNotFoundError(regex, contents)
    return match


def re_finditer(regex: str, contents: str, flags: int = 0) -> list[re.Match[str]]:
    """Return all matches of ``regex`` in ``contents``.

    Raises:
        MatchNotFoundError: No match is found.
    """
    matches = list(re.finditer(regex, contents, flags))
    if not matches:
        raise MatchNotFoundError(regex, contents)
    return matches
