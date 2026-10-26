# qccodec

qccodec knows how to talk to quantum chemistry programs. For each program it has one
**codec** that turns a [qcdata](https://qcdata.docs.afh.org)
`ProgramInput` into the program's native input files and command, and turns the
program's output back into structured qcdata results.

qccodec never runs anything itself.
[qccompute](https://qccompute.docs.afh.org) runs the program and uses
the codec on both ends, so most scientists use the two together.

## Install

```sh
python -m pip install qccompute   # also installs qccodec
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
| [`geometric`](programs/geometric.md) | optimization, transition_state | Drives any gradient program |
| [`sella`](programs/sella.md) | optimization, transition_state, irc | Drives any gradient program; optional Hessian program |
| [`orca`](programs/orca.md) | energy, gradient, hessian, optimization, transition_state | |
| [`psi4`](programs/psi4.md) | energy, gradient, hessian, optimization, transition_state | Any Psi4 method; point charges for QM/MM |
| [`terachem`](programs/terachem.md) | energy, gradient, hessian, optimization | GPU quantum chemistry |
| [`terachem-fe`, `terachem-pbs`](programs/terachem-servers.md) | energy, gradient | TeraChem servers through tcpb |
| [`xtb`](programs/xtb.md) | energy, gradient, hessian, optimization | GFN0/1/2-xTB, GFN-FF, g-xTB |

## Without qccompute

qccodec also works on its own: [decode output you already have](guides/decoding.md)
into qcdata objects, or [write a codec](guides/writing-a-codec.md) for a program
that isn't supported yet.

## The QC Suite

- [qcref](https://github.com/atomsforhumanity/qcref): physical constants, conversion
  factors, the periodic table, and solvent properties, each with its source.
- [qcdata](https://qcdata.docs.afh.org): data structures for quantum
  chemistry inputs and results.
- [qcinf](https://github.com/atomsforhumanity/qcinf): cheminformatics algorithms for
  qcdata structures.
- **qccodec**: program-specific encoders and parsers (this site).
- [qccompute](https://qccompute.docs.afh.org): runs programs and
  returns structured results.
- [BigChem](https://github.com/mtzgroup/bigchem) and
  [ChemCloud](https://github.com/mtzgroup/chemcloud-client): distributed and cloud
  execution of qccompute calculations.
