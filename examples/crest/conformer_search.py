"""CREST conformer search (iMTD-GC: metadynamics, then optimization and sorting).

Requires: ``pip install qccompute`` (it installs qccodec) and CREST 3 on PATH, the
official static build from https://github.com/crest-lab/crest/releases. No separate
xtb is needed.

Notes for CREST conformer searches (https://crest-lab.github.io/crest-docs/):

- The default workflow is iMTD-GC (``runtype = "imtd-gc"``). Other conformer
  workflows: "imtd-smtd", "entropy", "nci", "nci-mtd"; any other runtype is an error
  for this calculation type.
- Searches are long: minutes to hours for drug-sized molecules. GFN-FF ("gfnff") is
  much faster than GFN2-xTB and is a common choice for the search itself.
- ``model.method`` applies to every level of the calculation, so a GFN-FF search
  with GFN2-xTB re-ranking needs two runs: search here, then single points or
  optimizations of the conformers you keep.
- ``cregen`` settings control how structures are sorted and deduplicated: ``ewin``
  (energy window, kcal/mol), ``ethr`` (energy threshold, kcal/mol), ``rthr`` (RMSD
  threshold, angstrom), ``bthr`` (rotational-constant threshold).
- ``results.conformers`` are frames sorted by energy, lowest first: each holds a
  ``structure`` and its ``results.energy``. ``conformer_energies`` lists those
  energies in Hartree; ``results.structures`` lists the structures. Rotamers (often thousands) are read only when asked for, with
  ``options={"collect_rotamers": True}``.
- Water has a single conformer; replace it with your molecule (Structure.open reads
  an xyz file in angstrom) and expect a longer run.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

# Coordinates are in bohr.
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="crest",
    calctype=CalcType.conformer_search,
    structure=structure,
    model=Model(method="gfnff"),  # or "gfn2" (slower)
    keywords={
        "threads": 4,  # default: number of CPUs, at most 16
        # "runtype": "nci",  # for non-covalent clusters; default "imtd-gc"
        # "topo": False,  # allow changes in bonding topology between conformers
        "calculation": {
            "level": [
                {
                    "alpb": "water",  # implicit solvent; or "gbsa": "water"
                }
            ],
        },
        # Sorting and deduplication of the ensemble:
        "cregen": {
            "ewin": 6.0,  # keep conformers within 6 kcal/mol of the lowest
            # "ethr": 0.05,  # energy threshold for duplicates (kcal/mol)
            # "rthr": 0.125,  # RMSD threshold for duplicates (angstrom)
            # "bthr": 0.01,  # rotational-constant threshold for duplicates
        },
    },
)

output = compute(
    prog_input,
    # options={"collect_rotamers": True},  # also read crest_rotamers.xyz
    # timeout=3600,  # seconds before CREST is killed
    # collect_files=True,  # keep every output file in output.results.files
    # executables={"crest": "/path/to/crest"},  # or QCCOMPUTE_CREST_EXECUTABLE
)

results = output.results
print("Conformers:", len(results.conformers))
lowest = results.conformer_energies[0]
for i, energy in enumerate(results.conformer_energies):
    print(f"  {i}: {energy:.8f} Hartree, {(energy - lowest) * 627.5095:.2f} kcal/mol")
print("Lowest conformer (bohr):\n", results.conformers[0].structure.geometry)
# Save the ensemble, e.g. one xyz file per conformer:
# for i, conformer in enumerate(results.conformers):
#     conformer.structure.save(f"conformer_{i}.xyz")
