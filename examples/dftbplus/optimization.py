"""DFTB+ geometry optimization with its own optimizer (LBFGS by default).

Requires: DFTB+ on PATH (``conda install -c conda-forge dftbplus``) and, for the DFTB
methods, the Slater-Koster files from https://github.com/dftbparams in a directory
named by ``QCCODEC_DFTB_PARAMETERS`` (one subdirectory per set). See
single_point.py for how methods, parameter sets, and keywords map onto DFTB+.

Optimization specifics:

- The codec writes ``Driver = GeometryOptimisation`` with LBFGS, all atoms moving,
  200 steps at most, and a largest-gradient criterion of 1e-4 Hartree/bohr. Change
  any of it through ``keywords["Driver"]``, which merges into that block.
- ``MovedAtoms`` takes DFTB+'s own syntax with 1-based atom numbers, e.g. "2:-1" to
  freeze the first atom.
- Every step in ``results.trajectory`` carries its structure and energy; the last one
  also carries the gradient. The first step is your input structure: DFTB+ does not
  move or reorient it.
- A run that hits MaxSteps raises with "did not converge", trajectory attached.
- For transition states, use geomeTRIC with DFTB+ gradients (``program="geometric"``).
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

# Coordinates are in bohr.
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="dftbplus",
    calctype=CalcType.optimization,
    structure=structure,
    model=Model(method="DFTB3", basis="3ob-3-1"),  # or Model(method="GFN2-xTB")
    keywords={
        "Driver": {
            "MaxSteps": 100,
            "Convergence": {"GradElem": 1e-4},  # Hartree/bohr
            # "Optimiser": "Rational {}",  # or "Fire {}", "SteepestDescent {}"
            # "MovedAtoms": "2:-1",  # 1-based: keep atom 1 fixed
        },
    },
)

output = compute(prog_input)

results = output.results
print("Steps:", len(results.trajectory))
print("Energies (Hartree):", results.energies)
print("Final structure (bohr):\n", results.final_structure.geometry)
