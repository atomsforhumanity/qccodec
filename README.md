# qccodec

Encode `qcdata` inputs into native quantum chemistry files and decode (parse) program outputs into structured qcdata objects. Uses data structures from [qcdata](https://github.com/atomsforhumanity/qcdata).

[![image](https://img.shields.io/pypi/v/qccodec.svg)](https://pypi.python.org/pypi/qccodec)
[![image](https://img.shields.io/pypi/l/qccodec.svg)](https://pypi.python.org/pypi/qccodec)
[![image](https://img.shields.io/pypi/pyversions/qccodec.svg)](https://pypi.python.org/pypi/qccodec)
[![Actions status](https://github.com/atomsforhumanity/qccodec/workflows/Tests/badge.svg)](https://github.com/atomsforhumanity/qccodec/actions)
[![Actions status](https://github.com/atomsforhumanity/qccodec/workflows/Basic%20Code%20Quality/badge.svg)](https://github.com/atomsforhumanity/qccodec/actions)

`qccodec` works in harmony with a suite of other quantum chemistry tools for fast, structured, and interoperable quantum chemistry.

## The QC Suite of Programs

The QC Suite works in harmony to provide fast, structured, and interoperable quantum chemistry tools.

- [qcconst](https://github.com/atomsforhumanity/qcconst) - Physical constants, conversion factors, and a periodic table with clear source information for every value.
- [qcdata](https://github.com/atomsforhumanity/qcdata) - Elegant and intuitive data structures for quantum chemistry, featuring seamless Jupyter Notebook visualizations. [Documentation](https://qcdata.docs.atomsforhumanity.org)
- [qcinf](https://github.com/atomsforhumanity/qcinf) - Cheminformatics algorithms and structure utilities using standardized [qcdata](https://qcdata.docs.atomsforhumanity.org/) data structures.
- [qccodec](https://github.com/atomsforhumanity/qccodec) - A package for translating between standardized [qcdata](https://github.com/atomsforhumanity/qcdata) data structures and native QC program inputs and outputs.
- [qccompute](https://github.com/atomsforhumanity/qccompute) - A package for operating quantum chemistry programs using standardized [qcdata](https://qcdata.docs.atomsforhumanity.org/) data structures. Compatible with `TeraChem`, `psi4`, `QChem`, `NWChem`, `ORCA`, `Molpro`, `geomeTRIC` and many more.
- [BigChem](https://github.com/mtzgroup/bigchem) - A distributed application for running quantum chemistry calculations at scale across clusters of computers or the cloud. Bring multi-node scaling to your favorite quantum chemistry program.
- `ChemCloud` - A [web application](https://github.com/mtzgroup/chemcloud-server) and associated [Python client](https://github.com/mtzgroup/chemcloud-client) for exposing a BigChem cluster securely over the internet.

## ✨ Basic Usage

- Installation:

  ```sh
  python -m pip install qccodec
  ```

- Parse QC program outputs into structured data files with a single line of code.

  ```python
  from pathlib import Path
  from qcdata import CalcType
  from qccodec import decode

  stdout = Path("tc.out").read_text()
  data = decode("terachem", CalcType.gradient, stdout=stdout)
  ```

- The `data` object will be a `qcdata` object, either `SinglePointData`, `OptimizationData`, `ConformerSearchData` or other `*Data` structure depending on the `calctype`. Run `dir(data)` inside a Python interpreter to see the various values you can access. A few prominent values are shown here as an example:

  ```python
  from pathlib import Path
  from qcdata import CalcType
  from qccodec import decode

  stdout = Path("tc.out").read_text()
  data = decode("terachem", CalcType.hessian, stdout=stdout)

  data.provenance.program  # Producer identity
  data.provenance.program_version  # Parsed version, or None when unavailable
  data.energy
  data.gradient # If a gradient calc
  data.hessian # If a hessian calc
  data.calcinfo_nmo # Number of molecular orbitals
  ```

- Parsed values can be written to disk like this:

  ```py
  with open("data.json", "w") as f:
      f.write(data.model_dump_json())
  ```

- And read from disk like this:

  ```py
  from qcdata import SinglePointData

  data = SinglePointData.open("data.json")
  ```

- You can also run `qccodec` from the command line like this:

  ```sh
  qccodec -h # Get help message for cli

  qccodec terachem hessian tests/data/terachem/water.frequencies.out > data.json # Parse TeraChem stdout to json
  ```

- More complex parsing can be accomplished by passing the directory containing the scratch files to `decode` and optionally the input data used to generate the calculation (usually done from `qccompute` which uses structure data):

  ```python
  from pathlib import Path
  from qcdata import CalcType, ProgramInput
  from qccodec import decode

  stdout = Path("tc.out").read_text()
  directory = Path(".") / "scr.geom"
  input_data = ProgramInput.open("prog_inp.json")

  data = decode("terachem", CalcType.hessian, stdout=stdout, directory=directory, input_data=input_data)
  ```

## 💻 Contributing

Please see the [contributing guide](./CONTRIBUTING.md) for details on how to contribute new parsers to this project :)

If there's data you'd like parsed from output files or want to support input files for a new program, please open an issue in this repo explaining the data items you'd like parsed and include an example output file containing the data, like [this](https://github.com/atomsforhumanity/qccodec/issues/2).

## Encoding inputs

```python
from qcdata import ProgramInput, Structure
from qccodec import encode

input_data = ProgramInput(
    program="terachem",
    calctype="energy",
    structure=Structure.open("molecule.xyz"),
    model={"method": "hf", "basis": "sto-3g"},
)
native = encode(input_data)   # SubprocessInput(command=("terachem", "tc.in"), files={...})
```

`encode()` returns a `SubprocessInput` (a command plus the files it needs) or a
`PythonInput` (a callable plus its arguments). Nothing is executed; `qccompute`
runs these.

## Recovering failed calculations

```python
partial = decode("terachem", "gradient", stdout=logs, directory=scratch_dir, failed=True)
```

With `failed=True`, missing files and values are tolerated and whatever was parsed
is returned. Without it, a missing required value raises `ParserError` (or
`MatchNotFoundError`) with the recovered data attached as `.data`. The CLI supports
the same behavior with `--failed`.

## One `Codec` per program

Everything qccodec knows about a program lives in one `Codec` object: its name,
the calculation types it supports, an encoder, and either a set of small parsers
(file-based programs) or a decode function (Python APIs). Built-ins live in
`qccodec/programs/<name>.py`, one file per program, and you can build your own in a
script the same way:

```python
from qcdata import CalcType, ProgramInput
from qccodec import Codec, SubprocessInput, re_search

codec = Codec("myprog", calctypes={CalcType.energy})

@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    return SubprocessInput(
        command=("myprog", "run.inp"),
        files={"run.inp": f"{inp.model.method}\n{inp.structure.to_xyz()}"},
    )

@codec.parser(target="energy", calctypes=[CalcType.energy])
def parse_energy(stdout: str) -> float:
    return float(re_search(r"Total energy:\s+(-?\d+\.\d+)", stdout).group(1))
```

Then run it with qccompute, either explicitly or by name after registering it:

```python
import qccodec, qccompute

output = qccompute.compute(input_data, codec=codec)

qccodec.register(codec)              # now resolvable as program="myprog",
output = qccompute.compute(input_data)   # including as a geomeTRIC gradient child
```

`register()` also lets you override a built-in codec by name for the current process.

### Parsers

A parser is a plain function from file contents to a value. The decorator records
where the value goes and when the parser runs, and returns the function unchanged so
it stays easy to test directly.

```python
@codec.parser(target="energy", calctypes=[CalcType.energy, CalcType.gradient])
def parse_energy(stdout: str) -> float: ...                      # reads stdout (default)

@codec.parser(file="crest.engrad", target="gradient", calctypes=[CalcType.gradient])
def parse_gradient(contents: str) -> list[list[float]]: ...      # reads a named file or glob

@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]: ...   # needs several files
```

- `target` is the result field to fill, or a tuple for nested fields such as
  `("provenance", "program_version")`. Directory parsers may omit it and return a
  mapping that is merged into the result.
- `calctypes` restricts when the parser runs; omit it to run for every calculation type.
- `file` is `"stdout"` (default), `"directory"`, a file name or glob relative to the
  calculation directory, or a function of the `Context` returning such a name.
- `required=False` tolerates a missing file or `MatchNotFoundError`.

Directory parsers receive a `Context` with `stdout`, `directory`, `input_data`,
`failed`, `options`, `path(pattern)`, `read(pattern)`, and `decode(calctype, stdout=...)`
for nested decoding with the same codec.

### Failure detection and Python-native programs

```python
@codec.failure
def failure(ctx: Context) -> str | None:
    return "CREST reported FAILED." if "FAILED" in (ctx.stdout or "") else None
```

Many programs exit 0 after failing. When the hook returns a message, decoding
recovers what it can and raises `CalculationError` with that data attached.
Programs driven through a Python API skip parsers entirely:

```python
codec = Codec("xtb", calctypes={CalcType.energy, CalcType.gradient}, encode=encode, decode=decode)
```

where `decode(input_data, native_output)` maps `native_output.value` into a qcdata
object.

### Options and wavefunction propagation

A codec declares the decode options it accepts, with defaults. Callers override them
per calculation and parsers read them from the context:

```python
codec = Codec("crest", calctypes={...}, options={"collect_rotamers": False})

@codec.parser(file="directory", calctypes=[CalcType.conformer_search])
def parse_rotamers(ctx: Context) -> dict:
    if not ctx.options["collect_rotamers"]:
        return {}
    ...

decode("crest", "conformer_search", directory=d, input_data=inp, options={"collect_rotamers": True})
qccompute.compute(inp, options={"collect_rotamers": True})
```

Unknown option names raise `EncoderError`. For optimizers that chain calculations, a
codec may set `codec.propagate_wfn = fn` where `fn(output, input_data)` returns a new
input carrying the previous output's wavefunction as the initial guess; TeraChem copies
`c0`/`ca0`/`cb0` and sets `guess`. qccompute calls it when `propagate_wfn=True`.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full walkthrough of adding a program.

## Development

qccodec requires `qcdata>=0.19.0`. Run `uv sync --all-groups --locked` to install the
project and development dependencies. Optional integrations install with
`qccodec[geometric]`, `qccodec[xtb]`, `qccodec[qcengine]`, or `qccodec[tcpb]`; their
libraries are imported only when a calculation is invoked.
