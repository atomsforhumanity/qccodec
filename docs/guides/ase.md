# ASE workflows

ASE-based tools (ASE's optimizers, NEB, molecular dynamics, Sella run by hand) get
energies and forces from an ASE calculator. `qccodec.ase.ProgramCalculator` is one
whose every request is a calculation of any qccodec program: give it a `ProgramSpec`
and a function that computes a `ProgramInput`, such as `qccompute.compute`. The
program runs through its codec, so its results, failures, and provenance are the same
as anywhere else in qc*.

```python
from ase.io import read
from ase.optimize import BFGS
from qccompute import compute
from qcdata import CalcType, Model, ProgramSpec

from qccodec.ase import ProgramCalculator

spec = ProgramSpec(program="xtb", calctype=CalcType.gradient, model=Model(method="gfn2"))
atoms = read("water.xyz")
atoms.calc = ProgramCalculator(spec, compute, charge=0, multiplicity=1)
BFGS(atoms).run(fmax=0.01)

print(atoms.get_potential_energy())  # eV
print(len(atoms.calc.outputs), "calculations")  # every ProgramOutput, in order
```

- qccodec never runs programs, so you pass `compute`. Any function taking a
  `ProgramInput` and returning a `ProgramOutput` works.
- Units are converted at the boundary: positions in angstrom, energies in eV, forces
  in eV/angstrom, as ASE expects.
- Charge and multiplicity come from `charge` and `multiplicity`, or from a
  `structure=` whose charge, multiplicity, and identifiers every calculation keeps.
- A failed calculation raises ASE's `CalculationFailed`, with the `ProgramOutput` as
  its `.output`. With `qccompute.compute`'s default `raise_exc=True`, qccompute's own
  exception is raised first.
- Each request is one calculation, so ASE workflows do not get qccompute's batching
  or parallel lists. For optimizations and transition states, the
  [sella](../programs/sella.md) and [geomeTRIC](../programs/geometric.md) codecs run
  the whole optimization as one calculation instead. Requires ASE
  (`pip install ase`; `qccodec[sella]` includes it).

::: qccodec.ase.ProgramCalculator
    options:
      members: false
