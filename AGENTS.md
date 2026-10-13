## Development

- Use type annotations for all Python function and method parameters
  (except `self` and `cls`) and return values, including `-> None`.
- Annotate class and instance attributes, including dataclass and
  Pydantic model fields.
- Annotate local variables only when inference is insufficient or an
  explicit type meaningfully improves clarity. Avoid redundant
  annotations such as `count: int = 0`.
- Prefer precise types; avoid `Any` unless genuinely necessary.
- Use Google-style docstrings for all functions and methods.
- Start with a concise summary. Include Args, Returns, Yields, and
  Raises sections where applicable.
- Describe parameter and return semantics without repeating types
  already provided in annotations.
- Use a one-line docstring when no additional explanation is needed.