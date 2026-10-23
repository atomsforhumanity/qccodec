"""Sella intrinsic reaction coordinate (IRC) from a transition state.

Requires: ``pip install qccompute qccodec[sella]`` (Sella and ASE) and the child
program on PATH. This example uses xtb (GFN2-xTB): ``conda install -c conda-forge xtb``.

Notes for Sella IRCs (https://github.com/zadorlab/sella):

- ``CalcType.irc`` follows the steepest-descent path in mass-weighted coordinates
  down both sides of the transition state in ``structure``: Sella's ``IRC``, run in
  the reverse and then the forward direction from one analysis of the saddle.
- ``results`` is a ``PathData``: ``frames`` run from the end of the reverse branch,
  through the transition state (``ts_index``), to the end of the forward branch. Each
  frame holds its structure and the child's energy and gradient;
  ``forward_barrier`` and ``reverse_barrier`` are the transition state's energy above
  the first and last frames, in Hartree.
- A ``hessian`` subprogram (here xtb's Hessian) gives Sella the transition state's
  curvature instead of estimating it from gradients; it is attached to the
  transition-state frame.
- ``fmax``, ``fmax_inner``, ``steps`` go to ``IRC.run``; ``dx`` (step size,
  angstrom * amu^1/2), ``keep_going`` (continue past an inner-loop failure),
  ``eta``, ``gamma`` to ``IRC``. Sella's defaults apply to anything left out; set
  ``steps``.
- The codec keeps the path in the input's frame (Sella's IRC lets the molecule drift
  and spin) and ends a branch when the curvature of the vibrations, not the rigid
  rotations, is positive.
- An IRC stops short of the minima. Optimize the end structures before comparing
  energies (``path.frames[0].structure``, ``path.frames[-1].structure``).
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

# The HCN <-> HNC transition state (GFN2-xTB), coordinates in bohr.
transition_state = Structure(
    symbols=["C", "N", "H"],
    geometry=[
        [0.017, 0.036, 0.0],
        [2.283, -0.137, 0.0],
        [1.0, 2.0, 0.0],
    ],
    charge=0,
    multiplicity=1,
)

model = Model(method="gfn2")
prog_input = ProgramInput(
    program="sella",
    calctype=CalcType.irc,
    structure=transition_state,
    subprograms=[
        ProgramSpec(program="xtb", calctype=CalcType.gradient, model=model),
        ProgramSpec(program="xtb", calctype=CalcType.hessian, model=model),
    ],
    keywords={
        "fmax": 0.01,  # force at which a branch counts as at its minimum (eV/angstrom)
        "steps": 300,  # maximum steps per direction
        # "dx": 0.1,  # step size (angstrom * amu^1/2)
        # "keep_going": True,  # step on past an inner-loop convergence failure
    },
)

output = compute(prog_input)
path = output.results
print("Frames:", len(path.frames), "transition state at", path.ts_index)
print(f"Forward barrier: {path.forward_barrier * 627.5095:.1f} kcal/mol")
print(f"Reverse barrier: {path.reverse_barrier * 627.5095:.1f} kcal/mol")
print("Reverse end (bohr):\n", path.frames[0].structure.geometry)
print("Forward end (bohr):\n", path.frames[-1].structure.geometry)
# Save the path, e.g. to view in a molecule viewer: path.save("irc.xyz")
