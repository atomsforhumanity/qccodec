"""geomeTRIC nudged elastic band (NEB): a minimum-energy path between two structures.

Requires: ``pip install qccompute qccodec[geometric]`` (geomeTRIC itself) and the
gradient program on PATH. This example uses CREST for fast GFN2-xTB gradients:
https://github.com/crest-lab/crest/releases (official static build).

Notes for geomeTRIC NEB (https://geometric.readthedocs.io/):

- ``CalcType.path_optimization`` optimizes a chain of structures. The chain, endpoints
  included and in order, goes in ``structures["initial_chain"]``; ``structure`` is
  left out. geomeTRIC does not interpolate from two endpoints, so build the starting
  chain yourself (here: straight lines through a bent guess; a geodesic interpolation
  gives better starts). It uses ``images`` structures (11 by default) picked evenly
  from the chain.
- Every image of every iteration is a full ``gradient`` calculation of the program in
  ``subprograms``. All images of an iteration go to qccompute at once
  (qccompute 0.23+), so machine-learned potentials are evaluated in one batch and
  programs run side by side within the machine's cores.
- ``results`` is a ``PathOptimizationData``: ``paths`` holds the chain at each
  iteration and ``final_path`` the optimized one, a ``PathData`` whose ``ts_index``
  is the climbing image (or the highest one). Each frame is the child's calculation at
  that image.
- ``keywords`` are geomeTRIC's NEB parameters (``geometric.params.NEBParams``):
  ``images``, ``maxg`` and ``avgg`` (force thresholds, eV/angstrom), ``nebk`` (spring
  constant, kcal/mol/angstrom^2), ``neb_maxcyc``, ``climb``, ``optep`` (optimize the
  endpoints first), ``align``, ``trust``, ... geomeTRIC's defaults apply to anything
  left out.
- The climbing image is a transition-state guess: refine it with
  ``CalcType.transition_state`` and confirm it with an IRC (see irc.py).
"""

import numpy as np
from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

# HCN -> HNC: straight lines from HCN through a bent guess to HNC (bohr).
hcn = np.array([[0.0, 0.0, 0.0], [2.18, 0.0, 0.0], [-2.02, 0.0, 0.0]])
bent = np.array([[0.017, 0.036, 0.0], [2.283, -0.137, 0.0], [1.0, 2.0, 0.0]])
hnc = np.array([[0.0, 0.0, 0.0], [2.21, 0.0, 0.0], [4.10, 0.0, 0.0]])
steps = np.linspace(0.0, 1.0, 6)
geometries = [hcn + (bent - hcn) * t for t in steps]
geometries += [bent + (hnc - bent) * t for t in steps[1:]]
chain = [Structure(symbols=["C", "N", "H"], geometry=g) for g in geometries]

prog_input = ProgramInput(
    program="geometric",
    calctype=CalcType.path_optimization,
    structures={"initial_chain": chain},
    subprograms=[
        ProgramSpec(
            program="crest",
            calctype=CalcType.gradient,
            model=Model(method="gfn2"),
            keywords={"threads": 1},
        )
    ],
    keywords={
        "images": 11,  # structures in the band, endpoints included
        "neb_maxcyc": 100,  # maximum iterations
        # "maxg": 0.05,  # largest force at convergence (eV/angstrom)
        # "avgg": 0.025,  # average force at convergence (eV/angstrom)
        # "nebk": 1.0,  # spring constant (kcal/mol/angstrom^2)
        # "optep": True,  # optimize the endpoints first
    },
)

output = compute(prog_input)
results = output.results
path = results.final_path
print("Iterations:", len(results.paths))
print("Highest image:", path.ts_index)
print(f"Barrier from HCN: {path.forward_barrier * 627.5095:.1f} kcal/mol")
print(f"Barrier from HNC: {path.reverse_barrier * 627.5095:.1f} kcal/mol")
print("Transition-state guess (bohr):\n", path.transition_state.structure.geometry)
# Save the final path, e.g. to view in a molecule viewer: path.save("neb.xyz")
