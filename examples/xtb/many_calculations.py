"""xtb for many structures at once: calculations run side by side.

Requires: what examples/xtb/single_point.py requires.

How lists work (the same for every program that runs as its own process: ORCA,
Psi4, CREST, DFTB+, TeraChem, ...):

- Pass ``compute`` a list of inputs to get a list of outputs in the same order. Each
  calculation runs in its own process and directory, so several run at once in
  threads; no ``if __name__ == "__main__":`` guard is needed.
- A list is packed into the runner's threads: all the cores this process may use for
  ``qccompute.compute``, or ``qccompute.Runner(threads=...)``. Each calculation counts
  the threads it asks for: here xtb's ``parallel`` (``-P``), for ORCA ``pal``, for
  Psi4 ``-n``, for CREST ``threads``; one when it asks for none. With 16 cores and
  ``parallel`` 2, eight xtb calculations run at once. GPU programs (TeraChem) get
  their own GPUs through ``CUDA_VISIBLE_DEVICES``.
- The inputs may mix programs, methods, and calculation types.
- Every calculation runs first. Then, if any failed, ``ComputeErrors`` (an
  ``ExceptionGroup``) is raised: each error is a ``qcdata.ProgramError`` whose
  ``output`` is the failed output, and ``.outputs`` holds every output in input
  order. With ``raise_exc=False`` the outputs are returned as they are; check
  ``output.success``. ``runner.as_completed(inputs)`` yields them as they finish.
"""

import math
import time

from qccompute import ComputeErrors, Runner, compute
from qcdata import CalcType, Model, ProgramInput, Structure


def water(r: float) -> Structure:
    """Water with one O-H bond stretched to r bohr, the other at 1.81 bohr."""
    angle = math.radians(104.5)
    return Structure(
        symbols=["O", "H", "H"],
        geometry=[
            [0.0, 0.0, 0.0],
            [r, 0.0, 0.0],
            [1.81 * math.cos(angle), 1.81 * math.sin(angle), 0.0],
        ],
    )


distances = [1.4 + 0.1 * i for i in range(20)]  # bohr
inputs = [
    ProgramInput(
        program="xtb",
        calctype=CalcType.gradient,
        structure=water(r),
        model=Model(method="gfn2"),
        keywords={"parallel": 2},  # threads for each xtb, packed into the runner's
    )
    for r in distances
]

start = time.perf_counter()
try:
    outputs = compute(
        inputs,
        # timeout=600,  # seconds each calculation may run
    )
except ComputeErrors as errors:
    outputs = errors.outputs  # every output, failed ones with success=False
    for i, output in enumerate(outputs):
        if not output.success:
            print(f"Input {i} failed: {output.error}")
parallel = time.perf_counter() - start

start = time.perf_counter()
with Runner(threads=2) as runner:  # one at a time, for comparison
    runner.compute(inputs)
serial = time.perf_counter() - start

print("r (bohr)  energy (Hartree)")
for r, output in zip(distances, outputs, strict=True):
    print(f"{r:8.2f}  {output.results.energy:16.8f}")
print(f"Side by side: {parallel:.2f} s; one at a time: {serial:.2f} s")
