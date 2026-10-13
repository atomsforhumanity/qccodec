"""Simple data models to support parsing of QM program output files."""

from typing import Any

from .exceptions import DataCollectorError


class DataCollector(dict):
    """A dictionary for collecting data from parsers."""

    def add_data(self, target: str | tuple[str, ...], value: Any) -> None:
        """
        Assign a value into the DataCollector at the specified target.

        If target is a string, the value is assigned to that key.
        If target is a tuple, the method navigates through nested dictionaries,
        creating them as needed, and assigns the value at the final key.
        """
        keys = target if isinstance(target, tuple) else (target,)
        d = self
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        if keys[-1] in d:
            raise DataCollectorError(
                f"Target '{keys}' already exists in DataCollector. You cannot add the same target twice."
            )
        d[keys[-1]] = value
