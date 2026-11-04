"""geomeTRIC intrinsic reaction coordinate (IRC) driving another program's gradients.

Requires: ``pip install qccompute qccodec[geometric]`` (geomeTRIC itself) and the
gradient program on PATH. This example uses CREST for fast GFN2-xTB gradients:
https://github.com/crest-lab/crest/releases (official static build).

Notes for geomeTRIC IRCs (https://geometric.readthedocs.io/):

- ``CalcType.irc`` follows the IRC down both sides of the transition state in
  ``structure``, forward first, then backward. geomeTRIC needs the saddle's Hessian,
  which it computes by finite differences of child gradients (6N of them) and which is
  attached to the transition-state frame.
- ``results`` is a ``PathData``: ``frames`` are the points geomeTRIC accepted, from
  the end of its backward branch, through the transition state (``ts_index``), to the
  end of its forward branch. Each frame is the child's calculation at that point;
  rejected trial steps are left out. ``forward_barrier`` and ``reverse_barrier`` are
  the transition state's energy above the first and last frames, in Hartree.
- Once a branch's IRC has converged, geomeTRIC switches to minimizing, so the ends are
  minima. ``"irc_direction": "forward"`` (or ``"backward"``) runs one branch only.
- ``trust`` sets the IRC step size (angstrom); ``maxiter`` limits each direction.
- The codec keeps the path in the input's frame, removing the small rigid drift of
  geomeTRIC's internal-coordinate steps.
- Any program with gradients works as the child (examples/geometric/optimization.py
  shows xtb, Psi4, and TeraChem).
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

# The HCN <-> HNC transition state (GFN2-xTB), coordinates in bohr.
transition_state = Structure(
    symbols=["C", "N", "H"],
    geometry=[[0.017, 0.036, 0.0], [2.283, -0.137, 0.0], [1.0, 2.0, 0.0]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="geometric",
    calctype=CalcType.irc,
    structure=transition_state,
    subprograms=[
        ProgramSpec(
            program="crest",
            calctype=CalcType.gradient,
            model=Model(method="gfn2"),
            keywords={"threads": 1},
        )
    ],
    keywords={
        "maxiter": 300,  # maximum steps per direction
        # "trust": 0.05,  # IRC step size (angstrom)
        # "irc_direction": "forward",  # or "backward"; both by default
    },
)

output = compute(prog_input)
path = output.results
print("Frames:", len(path.frames), "transition state at", path.ts_index)
print(f"Forward barrier: {path.forward_barrier * 627.5095:.1f} kcal/mol")
print(f"Reverse barrier: {path.reverse_barrier * 627.5095:.1f} kcal/mol")
print("Backward end (bohr):\n", path.frames[0].structure.geometry)
print("Forward end (bohr):\n", path.frames[-1].structure.geometry)
# Save the path, e.g. to view in a molecule viewer: path.save("irc.xyz")
