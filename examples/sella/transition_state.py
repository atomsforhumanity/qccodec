"""Sella transition-state search driving another program's gradients and Hessians.

Requires: ``pip install qccompute qccodec[sella]`` (Sella and ASE) and the child
program on PATH. This example uses xtb (GFN2-xTB): ``conda install -c conda-forge xtb``.

Notes for Sella transition-state searches (https://github.com/zadorlab/sella):

- ``CalcType.transition_state`` runs Sella's saddle-point search (``order=1``).
- Without a Hessian program, Sella estimates the curvature from extra gradients at
  displaced geometries; those are not steps and stay out of ``results.frames``.
- With a second ``ProgramSpec`` of ``calctype="hessian"`` in ``subprograms`` (here
  xtb's Hessian), Sella starts from that program's Hessian instead, which usually
  takes fewer steps. ``"diag_every_n": N`` recomputes it every N steps. Each Hessian
  is attached to the frame at its geometry (``frame.results.hessian``).
- Start close to the saddle point. This example starts from a bent HCN, between HCN
  and HNC; the isomerization transition state has one imaginary frequency.
- ``fmax`` and ``steps`` are Sella's (eV/angstrom); the defaults are Sella's.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

# H between C and N, coordinates in bohr.
structure = Structure(
    symbols=["C", "N", "H"],
    geometry=[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [1.1, 1.9, 0.0]],
    charge=0,
    multiplicity=1,
)

model = Model(method="gfn2")
gradient_program = ProgramSpec(program="xtb", calctype=CalcType.gradient, model=model)
hessian_program = ProgramSpec(program="xtb", calctype=CalcType.hessian, model=model)

prog_input = ProgramInput(
    program="sella",
    calctype=CalcType.transition_state,
    structure=structure,
    subprograms=[gradient_program, hessian_program],  # drop the Hessian to estimate it
    keywords={
        "fmax": 0.001,  # largest force at convergence (eV/angstrom)
        "steps": 200,  # maximum optimizer steps
        # "diag_every_n": 5,  # recompute the Hessian every 5 steps
        # "internal": True,  # step in internal coordinates
    },
)

output = compute(prog_input)
results = output.results
print("Steps:", len(results.frames))
print("Energies (Hartree):", results.energies)
print("Transition state (bohr):\n", results.final_structure.geometry)

# Check: a transition state has exactly one imaginary frequency.
check = compute(
    ProgramInput(
        program="xtb",
        calctype=CalcType.hessian,
        structure=results.final_structure,
        model=model,
    )
)
print("Lowest frequencies (cm^-1):", check.results.freqs_wavenumber[:3])
