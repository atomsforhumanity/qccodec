"""xtb for many structures at once: calculations run side by side.

Requires: what examples/xtb/single_point.py requires, and qccompute 0.21.0 or newer.

How lists work (the same for every program that runs as its own process: ORCA,
Psi4, CREST, DFTB+, TeraChem, ...):

- Pass ``compute`` a list of inputs to get a list of outputs in the same order. Each
  calculation runs in its own process and directory, so several run at once in
  threads; no ``if __name__ == "__main__":`` guard is needed.
- ``max_workers`` is a budget of CPU threads, by default the cores this process may
  use. Each calculation counts the threads it asks for: here xtb's ``parallel``
  (``-P``), for ORCA ``pal``, for Psi4 ``-n``, for CREST ``threads``. With 16 cores
  and ``parallel`` 2, eight xtb calculations run at once. GPU programs (TeraChem) get
  one GPU each through ``CUDA_VISIBLE_DEVICES``.
- The inputs may mix programs, methods, and calculation types.
- Every calculation runs first. Then, if any failed, ``ComputeErrors`` (an
  ``ExceptionGroup``) is raised; each error has the failed input's position as
  ``input_index``, and ``.outputs`` holds every output. With ``raise_exc=False`` the
  outputs are returned as they are; check ``output.success``.
- ``print_logs=True`` prints each line prefixed with its input's index, such as
  ``[3]``. An ``update_func`` receives ``(index, all_logs, new_logs)`` from the thread
  running that calculation.
"""

import math
import time

from qccompute import compute
from qccompute.exceptions import ComputeErrors
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
        keywords={"parallel": 2},  # threads for each xtb, counted against max_workers
    )
    for r in distances
]

start = time.perf_counter()
try:
    outputs = compute(
        inputs,
        # max_workers=8,  # CPU threads all calculations may use at once
        # print_logs=True,  # stream every program's output, prefixed with [index]
        # timeout=600,  # seconds each calculation may run
    )
except ComputeErrors as errors:
    for error in errors.exceptions:
        print(f"Input {error.input_index} failed: {error}")
    outputs = errors.outputs  # every output, failed ones with success=False
parallel = time.perf_counter() - start

start = time.perf_counter()
compute(inputs, max_workers=2)  # one at a time, for comparison
serial = time.perf_counter() - start

print("r (bohr)  energy (Hartree)")
for r, output in zip(distances, outputs, strict=True):
    print(f"{r:8.2f}  {output.results.energy:16.8f}")
print(f"Side by side: {parallel:.2f} s; one at a time: {serial:.2f} s")
