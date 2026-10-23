"""Sella geometry optimization driving another program's gradients.

Requires: ``pip install qccompute qccodec[sella]`` (Sella and ASE) and the gradient
program on PATH. This example uses xtb for fast GFN2-xTB gradients:
``conda install -c conda-forge xtb``.

How it works (https://github.com/zadorlab/sella):

- Sella runs in Python and asks for a gradient at every step. Each one is a full
  ``gradient`` calculation of the program in ``subprograms``, with that program's own
  model and keywords, so any program with gradients works: xtb, CREST, Psi4, ORCA,
  TeraChem, AIMNet2, ... ASE is only the interface Sella talks to; no ASE calculator
  runs.
- ``keywords`` are Sella's own, in its units (eV, angstrom). ``fmax`` and ``steps``
  go to ``Sella.run``; everything else to the ``Sella`` constructor. Sella's defaults
  apply to anything left out: ``fmax`` 0.05 eV/angstrom and no practical step limit,
  so set ``steps``. A run that uses up ``steps`` fails, with its trajectory attached.
- ``constraints`` are calls to Sella's ``Constraints`` methods with 0-based atom
  indices; targets in angstrom and degrees.
- For minimizations Sella estimates curvature from gradients. A Hessian program in
  ``subprograms`` is used only with ``"eig": True`` (see transition_state.py).
- Every step in ``results.trajectory`` is the child's full output.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

# Coordinates are in bohr.
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

# The program that computes energies and gradients: xtb with GFN2-xTB.
gradient_program = ProgramSpec(
    program="xtb",
    calctype=CalcType.gradient,
    model=Model(method="gfn2"),
    keywords={"alpb": "water"},  # xtb's own keywords; see examples/xtb/
)

prog_input = ProgramInput(
    program="sella",
    calctype=CalcType.optimization,
    structure=structure,
    subprograms=[gradient_program],
    keywords={
        "fmax": 0.01,  # largest force at convergence (eV/angstrom)
        "steps": 100,  # maximum optimizer steps
        # "internal": True,  # step in internal coordinates
        #
        # Constraints (0-based atom indices; targets in angstrom and degrees):
        # "constraints": {
        #     "fix_translation": [0],  # keep the oxygen in place
        #     "fix_angle": [{"indices": [1, 0, 2], "target": 104.5}],
        # },
    },
)

output = compute(
    prog_input,
    # propagate_wfn=True,  # TeraChem: start each step from the previous orbitals
    # executables={"xtb": "/path/to/xtb"},  # the child's executable
)

results = output.results
print("Steps:", len(results.trajectory))
print("Energies (Hartree):", results.energies)
print("Final gradient (Hartree/bohr):\n", results.trajectory[-1].results.gradient)
print("Final structure (bohr):\n", results.final_structure.geometry)
print("Sella version:", results.provenance.program_version)
