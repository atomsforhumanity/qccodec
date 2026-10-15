# Decoding program output

qccompute decodes every calculation it runs. You can also decode output you already
have, from a cluster job, a colleague, or an old project, into the same qcdata objects.

## From Python

```python
from pathlib import Path
from qcdata import CalcType
from qccodec import decode

stdout = Path("tc.out").read_text()
data = decode("terachem", CalcType.hessian, stdout=stdout)

data.energy
data.hessian
data.provenance.program_version  # parsed version, or None
data.calcinfo_nmo                # number of molecular orbitals
```

`data` is the qcdata result type for the calculation type: `SinglePointData`,
`OptimizationData`, `ConformerSearchData`, ... Save and reload it as JSON:

```python
data.save("data.json")

from qcdata import SinglePointData
data = SinglePointData.open("data.json")
```

Many values live in output files rather than stdout. Pass the calculation directory,
and the `ProgramInput` that produced the run when the codec needs it (optimization
trajectories, for example):

```python
from qcdata import ProgramInput

data = decode(
    "terachem",
    CalcType.optimization,
    stdout=Path("tc.out").read_text(),
    directory=Path("."),                         # holds scr.geometry/optim.xyz
    input_data=ProgramInput.open("input.json"),
)
```

## From the command line

```sh
qccodec terachem hessian tc.out > data.json          # stdout only
qccodec terachem optimization tc.out . > data.json   # stdout plus a directory
```

## Failed calculations

A failed run usually still has useful values in it. Decode it in recovery mode to get
whatever can be parsed:

```python
partial = decode("terachem", "gradient", stdout=logs, directory=scratch_dir, failed=True)
```

With `failed=True`, missing files and values are tolerated and the recovered data is
returned. Without it, a missing required value raises `ParserError` (or
`MatchNotFoundError`), and a program that reported failure raises `CalculationError`;
either way the recovered data is attached as `.data`. Pass `report_failure=True` with
`failed=True` to get the program's explanation as `CalculationError` as well. The CLI
supports recovery with `--failed`.

## Encoding without running

`encode` produces the native input without running anything, which is useful for
checking what a program will receive or running it yourself:

```python
from qcdata import ProgramInput, Structure
from qccodec import encode

native = encode(
    ProgramInput(
        program="terachem",
        calctype="energy",
        structure=Structure.open("molecule.xyz"),
        model={"method": "hf", "basis": "sto-3g"},
    )
)
native.command  # ("terachem", "tc.in")
native.files    # {"tc.in": "...", "geometry.xyz": "..."}
```

`encode` returns a `SubprocessInput` (a command plus the files it needs) or a
`PythonInput` (a callable plus its arguments).
