"""AIMNet2 for many structures at once: a bond scan evaluated in batches.

Requires: what examples/aimnet2/single_point.py requires, and qccompute 0.21.0 or newer.

How batching works:

- Pass ``compute`` a list of inputs. Energy and gradient inputs with the same program,
  model, and keywords are evaluated in batches, one model call for up to
  ``get_codec("aimnet2").max_batch`` (32) structures, in this Python process. On a GPU
  that is many times faster than one call per structure.
- Each structure's results are what computing it alone gives, to the model's precision.
  Structures in one batch may differ in size, composition, and charge.
- Inputs with different models or keywords go in separate batches. Hessians are
  computed one structure at a time.
- A batch that runs out of GPU memory is split in half until it fits. A batch that
  fails otherwise is computed again one structure at a time, so each error belongs to
  its own input.
- The model stays loaded in this process, so the next ``compute`` call (the next step
  of a path optimization, say) does not load it again.
- ``output.execution.extras`` records ``batch_size`` and the structure's
  ``batch_index``; ``logs`` and ``wall_time`` are the whole batch's.
"""

import math
import time

from qccompute import compute
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
        program="aimnet2",
        calctype=CalcType.gradient,  # or CalcType.energy
        structure=water(r),
        model=Model(method="aimnet2"),
        keywords={
            # "device": "cpu",  # or "cuda"; leave out to use a GPU when present
        },
    )
    for r in distances
]

compute(inputs)  # load the model and warm up, so the timings compare evaluation only

start = time.perf_counter()
outputs = compute(inputs)  # one batch of 20
batched = time.perf_counter() - start

start = time.perf_counter()
alone = [compute(inp) for inp in inputs]  # 20 separate calls, for comparison
separate = time.perf_counter() - start

print("r (bohr)  energy (Hartree)  batch_index/batch_size")
for r, output in zip(distances, outputs, strict=True):
    extras = output.execution.extras
    print(
        f"{r:8.2f}  {output.results.energy:16.8f}  "
        f"{extras['batch_index']}/{extras['batch_size']}"
    )
largest = max(
    abs(a.results.energy - b.results.energy)
    for a, b in zip(alone, outputs, strict=True)
)
print(f"Batched: {batched:.3f} s; one at a time: {separate:.3f} s")
print(f"Largest energy difference from computing alone: {largest:.1e} Hartree")
