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

## Development

qccodec requires published `qcdata>=0.19.0`. Run `uv sync --all-groups --locked`
to install the project and development dependencies; no sibling checkouts are needed.

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
native = encode(input_data)
```

`decode(program, calctype, ...)` still takes the producer explicitly because a
standalone output file may have no associated input object. Decoded `*Data`
objects save producer provenance themselves. Parsed optimization trajectories
contain `ProgramOutput` records with scientific values in `.results` and runtime
metadata in `.execution`.

## Recovering failed calculations

```python
partial = decode("terachem", "gradient", stdout=logs, directory=scratch_dir, failed=True)
```

The decoder owns partial-data construction and provenance. Missing values and
artifacts are tolerated in this mode; recovered values remain validated. The CLI
supports the same behavior with `--failed`. This does not relax qcdata's strict
contract for successful `ProgramOutput` records.


## Complete invocations and local codecs

`encode(input_data)` now returns `SubprocessInput` or `PythonInput`. The former
contains `command` and `files`; the latter contains `function`,
`args`, `kwargs`, and `files`. Encoding does not execute the calculation.
`get_codec(input_data.program)` resolves built-ins lazily. Optional Python packages
are imported when their invocation runs, not when their codec is discovered.
Install optional dependencies with `qccodec[geometric]`, `qccodec[xtb]`,
`qccodec[qcengine]`, or `qccodec[tcpb]` as needed.

A scientist can supply any object or module implementing these two methods directly
as `qccompute.compute(input_data, codec=my_codec)`:

```python
def encode(input_data) -> PythonInput | SubprocessInput:
    ...

def decode(input_data, native_output):
    ...  # Return a qcdata.StructuredData result.
```

For Python libraries requiring several method calls, return a module-level wrapper
function. It constructs native objects, applies native settings, invokes the library,
extracts a native snapshot, and releases native resources. The decoder maps that
snapshot into qcdata fields and units. This keeps scientific decoding testable using
saved snapshots without the library installed. The xTB codec is an example.
Both runners merge stdout and stderr into the native output's `stdout` field. Python capture includes native file-descriptor writes. Native outputs also contain
a workspace directory and execution
status. Python outputs additionally contain `value`; subprocess outputs contain
`returncode`. A failed invocation still reaches the decoder. Raise `CalculationError`
or `ParserError` with `data=partial_results` for scientific failure or parsing failure.
Do not manufacture successful results from incomplete native output.

## Small decorated parsers remain the standard composition pattern

`CodecDefinition` is an optional convenience implementation. Each definition owns
its parsers. The decorator returns the original function, so direct parser tests
remain simple. For example, this entirely local demonstration needs no QC program:

```python
import sys
from qcdata import CalcType
from qccodec import CodecDefinition, SubprocessInput

codec = CodecDefinition(
    "example",
    supported_calctypes=frozenset({CalcType.energy}),
    encoder=lambda inp: SubprocessInput((sys.executable, "-c", "print(-1.0)")),
    iter_files=lambda stdout, directory: [("stdout", stdout)],
)

@codec.parser(filetype="stdout", calctypes=[CalcType.energy], target="energy")
def parse_energy(contents: str) -> float:
    return float(contents)

assert parse_energy("-1.0") == -1.0
# For a ProgramInput with program="example":
# output = qccompute.compute(input_data, codec=codec)
```

File types may be strings or program-local enums. `calctypes` filters applicability;
`target` can be a field name or tuple for nested fields. `required=False` tolerates
`MatchNotFoundError`, not arbitrary parser exceptions. Directory parsers retain their
existing `(directory, stdout, input_data, *, failed=False)` signature for multi-file
results and may return a mapping with `target=None`. A dictionary returned with an
explicit target is assigned at that target, not merged at the root.

Standalone `decode(program, calctype, stdout=..., directory=..., failed=...)` remains
available, as does `codec.decode_artifacts(...)`. The existing required-artifact rule
is retained: required parsers with explicit calculation types require their artifacts;
parsers applying to all calculation types operate when their artifact is present.
The global `registry` and program-keyed `ParserRegistry` have been removed.
Specifications live in `codec.parsers`; `ParserSpec` no longer repeats the program
name. To isolate composition tests, copy a definition with `dataclasses.replace`
and a selected `parsers` list. Copied definitions use their own parser selections.

Most file codecs use the composed decoder directly. When a definition needs custom
scientific completion checks, its optional decoder callable receives
`(definition, input_data, native_output)` so it can call
`definition.decode_artifacts(...)` without referencing global state. This internal
callback differs from the two-method public `Codec` interface shown above.

Built-in configuration uses small program-specific factories: for example,
`get_codec("crest", collect_rotamers=True)` binds rotamer collection to its decoder and
`get_codec("geometric", gradient_codec=...)` binds only its encoder. There is no
generic option bag or `configured()` method. Configure external integrations before
passing them as `codec=`. File encoders return `SubprocessInput` directly; the
intermediate `EncodedFiles` model has been removed.

`PythonInput.child_callback` optionally names the keyword into which qccompute injects
`compute_child(input_data, *, previous=None, codec=None)`. geomeTRIC uses this to bind each trial
geometry to its gradient child specification. The callback returns `ProgramOutput`,
including failed child outputs. Native optimizer construction and the geomeTRIC engine
bridge remain in this package; no dependency on qccompute is introduced.

`PythonInput.loggers` identifies Python loggers to capture. `wavefunction_files` on
either invocation specifies relative glob patterns for optional restart-file retention.
A codec may provide `restart(input_data, previous_results)` to prepare a new input;
this is optional and not part of the minimum two-method interface.


Invocations may also specify `environment` defaults required by a native binding.
qccompute merges the inherited environment, then invocation defaults, then explicit
`compute(environment=...)` overrides. xTB defaults to one OpenMP thread (preserving
its previous integration behavior) and enables unbuffered Fortran output before
import. CREST also starts with `OMP_NUM_THREADS=1` in its child environment,
independently of its native `threads` keyword. The xTB wrapper binds a native output file, releases it, and forwards the text
to stdout for generic capture. Import-time settings cannot reset a library already
initialized by the caller.


A local external gradient integration can also drive geomeTRIC without registration:

```python
optimizer_codec = get_codec("geometric", gradient_codec=my_gradient_codec)
result = qccompute.compute(optimization_input, codec=optimizer_codec)
```

The gradient child's `ProgramSpec` still records its program, model, and keywords;
`gradient_codec` supplies its local implementation. Each child receives the same
normal qccompute lifecycle and produces its own ProgramOutput.
