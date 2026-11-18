# qccodec

qccodec knows how to talk to quantum chemistry programs. For each program it has one
**codec** that turns a [qcdata](https://qcdata.docs.afh.org/next/)
`ProgramInput` into the program's native input files and command, and turns the
program's output back into structured qcdata results.

Each codec is also a program that [qccompute](https://qccompute.docs.afh.org/next/) runs:
qccodec registers it under qccompute's `qccompute.programs` entry points, and its
adapter (`qccodec.CodecProgram`) encodes the input, runs the program in a scratch
directory with its output captured and a deadline, and decodes the result. qccompute
finds it by name, runs it on its thread count, and stores results. A thread request
larger than the calculation's count (ORCA `pal`, Psi4 `-n`) is lowered to the count,
with a `resource_capped` warning. An ML potential granted no GPU runs on the CPU; it warns
`backend_fallback` only when it was granted one and still ran on the CPU. A calculation
with nothing to converge (ML potentials, GFN-FF and non-SCC DFTB single points) meets an
accuracy target (the shared keyword `accuracy`) as it is and records it as met, with no
thresholds; every other codec refuses one with kind `unsupported`, since none turns a
target into its program's thresholds yet.

## Install

These pages document the alpha, at
[qccodec.docs.afh.org/next](https://qccodec.docs.afh.org/next/); the site root documents
the stable release. Install the alpha with minimum versions of its matching packages
(`--prerelease=allow` lets the resolver take pre-releases):

```sh
uv pip install --prerelease=allow "qcdata>=0.24.0a8" "qccompute[qccodec]>=0.23.0rc1" "qccodec>=0.22.0rc1"
```

You also need the program itself; each [program page](programs/index.md) says how to
install it and get it found.

## Run a calculation

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

Every program page has complete, runnable scripts like this one for each calculation
type it supports, with every common option shown and the program's quirks explained.
Copy one, delete what you don't need, and run it.

## Supported programs

| `program` | Calculation types | Notes |
| --- | --- | --- |
| [`crest`](programs/crest.md) | energy, gradient, hessian, optimization, conformer_search | GFN-xTB and GFN-FF built in |
| [`dftbplus`](programs/dftbplus.md) | energy, gradient, hessian, optimization | DFTB3, SCC-DFTB, DFTB, and xTB; parameter-set settings from qcref |
| [`geometric`](programs/geometric.md) | optimization, transition_state, irc | Drives any gradient program (`qccodec[geometric]`) |
| [`sella`](programs/sella.md) | optimization, transition_state, irc | Drives any gradient program; optional Hessian program (`qccodec[sella]`) |
| [`orca`](programs/orca.md) | energy, gradient, hessian, optimization, transition_state | |
| [`psi4`](programs/psi4.md) | energy, gradient, hessian, optimization, transition_state | Any Psi4 method; point charges for QM/MM |
| [`terachem`](programs/terachem.md) | energy, gradient, hessian, optimization | GPU quantum chemistry |
| [`terachem-fe`, `terachem-pbs`](programs/terachem-servers.md) | energy, gradient | TeraChem servers through tcpb (`qccodec[tcpb]`) |
| [`xtb`](programs/xtb.md) | energy, gradient, hessian, optimization | GFN0/1/2-xTB, GFN-FF, g-xTB; point charges for QM/MM |
| [`tblite`](programs/tblite.md) | energy, gradient | GFN1/GFN2/IPEA1-xTB as a Python library |
| [`aimnet2`](programs/aimnet2.md) | energy, gradient, hessian | AIMNet2 machine-learned potentials |
| [`fairchem`](programs/fairchem.md) | energy, gradient, hessian | UMA and eSEN (gated weights) |
| [`mace`](programs/mace.md) | energy, gradient, hessian | MACE-OFF, MACE-MP, MACE-OMOL, custom models |
| [`orb`](programs/orb.md) | energy, gradient | Orb v3 and OrbMol |
| [`torchani`](programs/torchani.md) | energy, gradient, hessian | ANI-2x, ANI-1ccx, ... |

## Without qccompute

qccodec also works on its own: [decode output you already have](guides/decoding.md)
into qcdata objects, or [write a codec](guides/writing-a-codec.md) for a program
that isn't supported yet.

## The QC Suite

- [qcref](https://github.com/atomsforhumanity/qcref): physical constants, conversion
  factors, the periodic table, and solvent properties, each with its source.
- [qcdata](https://qcdata.docs.afh.org/next/): data structures for quantum
  chemistry inputs and results.
- [qcinf](https://github.com/atomsforhumanity/qcinf): cheminformatics algorithms for
  qcdata structures.
- **qccodec**: program-specific encoders and parsers (this site).
- [qccompute](https://qccompute.docs.afh.org/next/): runs programs and
  returns structured results.
- [qcstore](https://github.com/atomsforhumanity/qcstore): keeps results under their
  calculation key: immutable runs, tags, and an optional index.
- [BigChem](https://github.com/mtzgroup/bigchem) and
  [ChemCloud](https://github.com/mtzgroup/chemcloud-client): distributed and cloud
  execution of qccompute calculations.
