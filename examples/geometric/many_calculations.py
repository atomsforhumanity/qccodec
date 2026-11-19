"""geomeTRIC for many structures at once: optimizations run side by side.

Requires: what examples/geometric/optimization.py requires (geomeTRIC and CREST).

How lists of Python calculations work (geomeTRIC, tblite, the ML potentials):

- geomeTRIC runs in this process: it writes only in its calculation directory and
  logs through one logger, so a list's optimizations run side by side in threads, and
  a program registered in this script can be its gradient program. tblite and the ML
  potentials run in worker processes instead (``python -m qccompute.serve``), which
  stay alive between calls until the runner closes; this script is never imported
  again there and needs no ``if __name__ == "__main__":`` guard.
- A list is packed into the runner's threads (all the cores this process may use, or
  ``qccompute.Runner(threads=...)``): an optimization counts the threads of its
  gradient program (CREST ``threads`` here). A worker sets each call's threads for
  PyTorch and the BLAS and OpenMP runtimes, so numpy does not start a thread per core.
- The gradient calculations run inside each optimization, through the runner.
- Failures are reported as for any list: see examples/xtb/many_calculations.py.
"""

import random
import time

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

random.seed(0)
start_geometry = [[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]


def jiggled() -> Structure:
    """The starting water with every coordinate moved by up to 0.2 bohr."""
    return Structure(
        symbols=["O", "H", "H"],
        geometry=[
            [x + random.uniform(-0.2, 0.2) for x in atom] for atom in start_geometry
        ],
    )


gradient_program = ProgramSpec(
    program="crest",
    calctype=CalcType.gradient,
    model=Model(method="gfn2"),
    keywords={"threads": 1},  # threads for each gradient, packed into the runner's
)
inputs = [
    ProgramInput(
        program="geometric",
        calctype=CalcType.optimization,
        structure=jiggled(),
        subprograms=[gradient_program],
        keywords={"maxiter": 100},
    )
    for _ in range(4)
]

start = time.perf_counter()
outputs = compute(inputs)  # four optimizations side by side
# with qccompute.Runner(threads=2) as runner:  # at most two at a time
#     outputs = runner.compute(inputs)
elapsed = time.perf_counter() - start

for i, output in enumerate(outputs):
    results = output.results
    print(
        f"Optimization {i}: {len(results.frames)} frames, final energy "
        f"{results.energies[-1]:.8f} Hartree"
    )
print(f"Four optimizations side by side: {elapsed:.2f} s")
