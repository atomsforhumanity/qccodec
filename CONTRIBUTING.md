# Parsing Framework Overview

Hey there 👋! Look at you wanting to contribute! This package is designed to make it as easy as possible for you to contribute new parsers for quantum chemistry programs and to make it as easy as possible to maintain the code. This document will walk you through the design decisions and how to add new parsers.

This framework standardizes how output files from quantum chemistry programs are parsed and converted into structured result objects. It separates parsers into two main categories:

1. Standard File Parsers:
   These functions operate on a single file’s contents (e.g., stdout/log text). They accept a single input (which may be a str or bytes) and return one parsed value (which may be a single `float` or a list of objects).

2. Directory Parsers:
   These functions handle more complex parsing tasks that involve an entire directory of output files and may need to operate on more than one file to create their final parsed value(s). They have a different signature and receive the directory path with optional additional context such as the stdout string and an input object. They may return a single value like standard parsers or a dictionary mapping keys to parsed values. When no target is specified at registration, the returned dictionary is merged into the final results data with each value at its specified key.

This approach keeps the interface consistent and clear:

- Simple Parsers: Return a single value and specify a target.
- Directory Parsers: Return a single value and specify a target, or return a dictionary and have their keys merged into the overall results.

## Registration of Parsers

All parsers are registered via the `@register` decorator. The decorator’s parameters are as follows:

- `filetype`: A value (typically from a program-specific enum such as `CrestFileType`) indicating the type of file the parser handles. For example, `CrestFileType.stdout` for log output or `CrestFileType.directory` for full-directory processing.

- `calctypes` (optional): A list of calculation types (from the `qcdata.CalcType` enum) on which this parser should run. If omitted, the parser applies to all calculation types.

- `required`: A boolean indicating whether this parser is mandatory. If `True` and the parser fails to parse the expected data, a `MatchNotFoundError` is raised when the parser is executed from within the `decode` function. The parser should always raise a `MatchNotFoundError` if the value is not found and it is called on its own.

- `target` (optional for directory parsers): For standard file parsers, this key (or nested key as a tuple) indicates where in the final results object the parser’s output should be stored. For directory parsers—those registered with `FileType.directory`—if no target is provided the parser is expected to return a dictionary, which will be merged into the final results object.

The registry itself enforces that no two parsers for the same program register the same target.

## Example: Standard File Parser

```python
from qccodec import register
from qcdata import CalcType
from qccodec.utils import re_search


@register(filetype=TeraChemFileType.STDOUT, calctypes=[CalcType.energy, CalcType.gradient], target="energy")
def parse_energy(contents: str) -> float:
    """
    Parse the final energy from TeraChem stdout.

    Args:
        contents: The contents of the TeraChem stdout file.

    Returns:
        The energy as a float.
    """
    regex = r"FINAL ENERGY: (-?\d+(?:\.\d+)?)"
    return float(re_search(regex, contents).group(1))

```

## Example: Directory Parser (Dictionary Output)

In this case, because the parser processes the whole directory output and returns multiple pieces of data, we omit the target. The returned dictionary is merged by the top-level parser.

```python
from qccodec import register
from qcdata import CalcType, ProgramInput

@register(filetype=CrestFileType.DIRECTORY, calctypes=[CalcType.conformer_search])
def parse_conformers(directory: Path, stdout: Optional[str], input_data: ProgramInput, *, failed: bool = False) -> dict[str, Any]:
    """Parse the conformers from the output directory of a CREST conformer search calculation.

    Args:
        directory: Path to the directory containing the CREST output files.
        stdout: The contents of the CREST stdout file (not used).
        input_data: The input object used for the calculation

    Returns:
        The parsed conformers and their energies as a dictionary.
    """
    directory = Path(directory)
    conformers = Structure.open_multi(
        directory / "crest_conformers.xyz",
        charge=input_data.structure.charge,
        multiplicity=input_data.structure.multiplicity,
    )

    # CREST places the energy as the only value in the comment line
    conf_energies = [
        float(conf.extras[Structure._xyz_comment_key][0]) for conf in conformers
    ]

    return {
        "conformers": conformers,
        "conformer_energies": conf_energies,
    }

```

## Example: Directory Parser (Single Output)

Directory parsers may also return a single target value instead of a dictionary.

```python
@register(
  filetype=TeraChemFileType.DIRECTORY,
    calctypes=[CalcType.optimization],
    target="trajectory", # Note target!
)
def parse_trajectory(directory: Path, stdout: str, input_data: ProgramInput, *, failed: bool = False) -> list[ProgramOutput]:
    """Parse the output directory of a TeraChem optimization calculation into a trajectory.

    Args:
        directory: Path to the directory containing the TeraChem output files.
        stdout: The contents of the TeraChem stdout file.
        input_data: The input object used for the calculation.

    Returns:
        A list of ProgramOutput objects.
    """
    # Create the trajectory
    trajectory: list[ProgramOutput] = []
    # Parsing logic here...
    return trajectory
```

See `parsers/terachem.py` for examples of both types of parsers.

## Top-Level Decode Function

The decode function orchestrates the parsing process:

1.  It collects input files using a unified generator (`prog.iter_files()`) that will discover all parsable files.

2.  It retrieves the appropriate parsers from the registry based on the program, filetype, and calctype.

3.  It then dispatches parsers against these file:

    - For Standard File Parsers: Calls the parser with the file’s contents and assigns the returned value to the specified target.

    - For Composite (Directory) Parsers: Calls the parser with additional context (directory, stdout, input object). If the parser returns a dictionary, its key–value pairs are merged into the data collector, otherwise it assigns the returned value at the specified target.

4.  Finally, the assembled data (a dictionary) is passed into the correct result model from `qcdata` and returned.

## Duplicate Target Registration

The registry enforces unique targets per program per `CalcType`. This ensures that each parsed value is uniquely associated with a key in the final results. For example, if two parsers attempt to register with the same target for a given program for a given `CalcType`, the registry will raise an error, preventing ambiguity in the final result.

## Parsing failed calculations

`decode(..., failed=True)` returns the appropriate scientific data type with all
recoverable values and program provenance. Missing matches and missing files are
tolerated; malformed recovered values still fail validation. The default remains
strict about required parser matches and artifacts.

Directory parsers accept a keyword-only `failed=False` argument and forward it to
nested `decode()` calls. Use `MatchNotFoundError` or `FileNotFoundError` for absent
content. Other parser errors remain errors. Reconstructed energy-only trajectory
snapshots use energy inputs; a missing gradient file does not imply a failed step.
