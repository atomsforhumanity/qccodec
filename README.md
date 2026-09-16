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
