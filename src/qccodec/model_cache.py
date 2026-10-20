"""Keep loaded machine-learned models in memory between calculations.

qccompute runs codecs for Python libraries (tblite and the machine-learned potentials)
in the calling process, so a model loaded for one calculation can serve the next one.
Loading weights takes seconds, and often a download the first time; an optimization or
reaction path pays it once instead of for every gradient.

Loaded models hold memory (GPU memory on a GPU) until they are dropped. The cache
keeps the most recently used models, ``QCCODEC_MAX_MODELS`` of them (2 by default),
and unloads the least recently used one when another is loaded. ``0`` turns caching
off, so every calculation loads its model afresh. :func:`clear_model_cache` unloads
everything, for example in a long-running service before switching models.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

MAX_MODELS_VARIABLE = "QCCODEC_MAX_MODELS"
DEFAULT_MAX_MODELS = 2

_M = TypeVar("_M")
_MODELS: OrderedDict[tuple[Hashable, ...], Any] = OrderedDict()
_LOCK = threading.Lock()


def max_models() -> int:
    """Return how many models may stay loaded, from ``QCCODEC_MAX_MODELS``.

    Raises:
        ValueError: The variable is not a non-negative integer.
    """
    raw = os.environ.get(MAX_MODELS_VARIABLE, "").strip()
    if not raw:
        return DEFAULT_MAX_MODELS
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(
            f"{MAX_MODELS_VARIABLE} must be a non-negative integer, not {raw!r}."
        )
    return value


def cached_model(key: tuple[Hashable, ...], load: Callable[[], _M]) -> _M:
    """Return the model for ``key``, loading it with ``load`` if it is not cached.

    Args:
        key: Everything that selects the model: program, model name, checkpoint,
            device, and any loading options.
        load: Builds the model.
    """
    with _LOCK:
        if key in _MODELS:
            _MODELS.move_to_end(key)
            return _MODELS[key]
        limit = max_models()
        evicted = False
        while _MODELS and len(_MODELS) >= max(limit, 1):
            _MODELS.popitem(last=False)
            evicted = True
        if evicted:
            _release_memory()
        model = load()
        if limit > 0:
            _MODELS[key] = model
        return model


def cached_models() -> list[tuple[Hashable, ...]]:
    """Return the keys of the loaded models, least recently used first."""
    with _LOCK:
        return list(_MODELS)


def clear_model_cache() -> None:
    """Unload every cached model and return its memory, including GPU memory."""
    with _LOCK:
        _MODELS.clear()
        _release_memory()


def _release_memory() -> None:
    """Collect dropped models and hand PyTorch's cached GPU memory back to the driver."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
