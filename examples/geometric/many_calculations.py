"""geomeTRIC for many structures at once: optimizations run side by side.

Requires: what examples/geometric/optimization.py requires (geomeTRIC and CREST), and
qccompute 0.21.0 or newer.

How lists of Python calculations work (geomeTRIC, tblite):

- geomeTRIC runs in Python, and a Python calculation takes over its process's output
  streams and working directory while it runs. So when a list holds two or more,
  qccompute runs them side by side in worker processes, started with
  ``python -m qccompute._worker``: this script is never imported again and needs no
  ``if __name__ == "__main__":`` guard.
- ``max_workers`` is the budget of CPU threads, by default the cores this process may
  use. An optimization counts the threads of its gradient program (CREST ``threads``
  here). Each worker starts with ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, and
  ``MKL_NUM_THREADS`` set to its share (unless you set them), so numpy in each worker
  does not start a thread per core.
- The gradient calculations run inside each optimization as usual.
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
    keywords={"threads": 1},  # threads for each gradient, counted against max_workers
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
outputs = compute(inputs)  # four workers, one optimization each
# outputs = compute(inputs, max_workers=2)  # at most two at a time
elapsed = time.perf_counter() - start

for i, output in enumerate(outputs):
    results = output.results
    print(
        f"Optimization {i}: {len(results.trajectory)} steps, final energy "
        f"{results.energies[-1]:.8f} Hartree"
    )
print(f"Four optimizations side by side: {elapsed:.2f} s")
