# qccodec

[![image](https://img.shields.io/pypi/v/qccodec.svg)](https://pypi.python.org/pypi/qccodec)
[![image](https://img.shields.io/pypi/l/qccodec.svg)](https://pypi.python.org/pypi/qccodec)
[![image](https://img.shields.io/pypi/pyversions/qccodec.svg)](https://pypi.python.org/pypi/qccodec)
[![CI](https://github.com/atomsforhumanity/qccodec/actions/workflows/ci.yaml/badge.svg)](https://github.com/atomsforhumanity/qccodec/actions/workflows/ci.yaml)

Translates [qcdata](https://github.com/atomsforhumanity/qcdata) inputs into native program inputs and parses program outputs back into qcdata objects, one codec per program.

**Documentation, including a copy-and-paste example script for every program and calculation type: [qccodec.docs.atomsforhumanity.org](https://qccodec.docs.atomsforhumanity.org)**

## The QC Suite

These packages work together to make quantum chemistry fast, structured, and interoperable.

- [qcref](https://github.com/atomsforhumanity/qcref) - Reference data for computational chemistry: CODATA physical constants, conversion factors, a periodic table, and solvent properties, each value with its source.
- [qcdata](https://github.com/atomsforhumanity/qcdata) - Data structures for quantum chemistry inputs and results, with Jupyter visualizations. [Documentation](https://qcdata.docs.atomsforhumanity.org)
- [qcinf](https://github.com/atomsforhumanity/qcinf) - Cheminformatics algorithms and structure utilities for [qcdata](https://github.com/atomsforhumanity/qcdata) structures.
- [qccodec](https://github.com/atomsforhumanity/qccodec) - Translates [qcdata](https://github.com/atomsforhumanity/qcdata) inputs into native program inputs and parses program outputs back into qcdata objects, one codec per program. [Documentation](https://qccodec.docs.atomsforhumanity.org)
- [qccompute](https://github.com/atomsforhumanity/qccompute) - Runs quantum chemistry programs on [qcdata](https://github.com/atomsforhumanity/qcdata) inputs using [qccodec](https://github.com/atomsforhumanity/qccodec) codecs and returns structured results. [Documentation](https://qccompute.docs.atomsforhumanity.org)
- [BigChem](https://github.com/mtzgroup/bigchem) - Distributed execution of qccompute calculations across clusters of computers or the cloud.
- [ChemCloud](https://github.com/mtzgroup/chemcloud-client) - A [web application](https://github.com/mtzgroup/chemcloud-server) and [Python client](https://github.com/mtzgroup/chemcloud-client) that expose a BigChem cluster securely over the internet.

## ✨ Basic Usage

qccodec never runs programs itself; [qccompute](https://github.com/atomsforhumanity/qccompute) runs them and uses qccodec on both ends.

```sh
python -m pip install qccompute   # also installs qccodec
```

```python
from qcdata import CalcType, Model, ProgramInput, Structure
from qccompute import compute

prog_input = ProgramInput(
    program="psi4",
    calctype=CalcType.gradient,
    structure=Structure(
        symbols=["O", "H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    ),
    model=Model(method="wb97x-d", basis="def2-svp"),
    keywords={"d_convergence": 1e-8},  # Psi4's own option names
)
output = compute(prog_input)
print(output.results.energy, output.results.gradient)
```

Decode output you already have, from Python or the command line:

```python
from pathlib import Path
from qccodec import decode

data = decode("terachem", "hessian", stdout=Path("tc.out").read_text())
data.energy, data.hessian, data.freqs_wavenumber
```

```sh
qccodec terachem hessian tc.out > data.json
```

See [Decoding program output](https://qccodec.docs.atomsforhumanity.org/guides/decoding/) for directories, failed calculations, and encoding without running.

## Supported programs

Each program has a documentation page and a runnable example script for every calculation type in [`examples/`](./examples).

| `program` | Calculation types | Notes |
| --- | --- | --- |
| [`crest`](https://qccodec.docs.atomsforhumanity.org/programs/crest/) | energy, gradient, hessian, optimization, conformer_search | GFN-xTB and GFN-FF built in |
| [`dftbplus`](https://qccodec.docs.atomsforhumanity.org/programs/dftbplus/) | energy, gradient, hessian, optimization | DFTB3, SCC-DFTB, DFTB, and xTB; parameter-set settings from qcref |
| [`geometric`](https://qccodec.docs.atomsforhumanity.org/programs/geometric/) | optimization, transition_state | Drives any gradient program (`qccodec[geometric]`) |
| [`sella`](https://qccodec.docs.atomsforhumanity.org/programs/sella/) | optimization, transition_state | Drives any gradient program; optional Hessian program (`qccodec[sella]`) |
| [`orca`](https://qccodec.docs.atomsforhumanity.org/programs/orca/) | energy, gradient, hessian, optimization, transition_state | |
| [`psi4`](https://qccodec.docs.atomsforhumanity.org/programs/psi4/) | energy, gradient, hessian, optimization, transition_state | Any Psi4 method; point charges for QM/MM |
| [`terachem`](https://qccodec.docs.atomsforhumanity.org/programs/terachem/) | energy, gradient, hessian, optimization | GPU quantum chemistry |
| [`terachem-fe`, `terachem-pbs`](https://qccodec.docs.atomsforhumanity.org/programs/terachem-servers/) | energy, gradient | TeraChem servers through tcpb (`qccodec[tcpb]`) |
| [`xtb`](https://qccodec.docs.atomsforhumanity.org/programs/xtb/) | energy, gradient, hessian, optimization | GFN0/1/2-xTB, GFN-FF, g-xTB; point charges for QM/MM |
| [`tblite`](https://qccodec.docs.atomsforhumanity.org/programs/tblite/) | energy, gradient | GFN1/GFN2/IPEA1-xTB as a Python library |
| [`aimnet2`](https://qccodec.docs.atomsforhumanity.org/programs/aimnet2/) | energy, gradient, hessian | AIMNet2 machine-learned potentials |
| [`fairchem`](https://qccodec.docs.atomsforhumanity.org/programs/fairchem/) | energy, gradient, hessian | UMA and eSEN (gated weights) |
| [`mace`](https://qccodec.docs.atomsforhumanity.org/programs/mace/) | energy, gradient, hessian | MACE-OFF, MACE-MP, MACE-OMOL, custom models |
| [`orb`](https://qccodec.docs.atomsforhumanity.org/programs/orb/) | energy, gradient | Orb v3 and OrbMol |
| [`torchani`](https://qccodec.docs.atomsforhumanity.org/programs/torchani/) | energy, gradient, hessian | ANI-2x, ANI-1ccx, ... |

The last six run in the calling Python process: install the library next to qccompute (for example `pip install tblite`) instead of putting a program on `PATH`. Run any of them under `geometric` or `sella` for optimizations and transition states.

To support a new program, see [Writing a codec](https://qccodec.docs.atomsforhumanity.org/guides/writing-a-codec/) and the step-by-step walkthrough in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, the pre-commit workflow, running tests, and the release process.

If there's data you'd like parsed from output files or a program you'd like supported, open an issue describing the data and include an example output file, like [this](https://github.com/atomsforhumanity/qccodec/issues/2).

## Optional dependencies

Optional integrations install with `qccodec[geometric]`, `qccodec[sella]`, or `qccodec[tcpb]`. To run any program from ASE-based tools, see [ASE workflows](https://qccodec.docs.atomsforhumanity.org/guides/ase/). The tblite and machine-learned-potential codecs need their own libraries, installed separately (several of them, such as `mace-torch` and `fairchem-core`, cannot share an environment). Every library is imported only when a calculation is invoked.
