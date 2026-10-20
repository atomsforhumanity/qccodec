"""DFTB+ single point: DFTB3, SCC-DFTB, or xTB energy, gradient, or Hessian.

Requires: ``pip install qccompute`` (it installs qccodec) and DFTB+ on PATH, e.g.
``conda install -c conda-forge dftbplus`` (the program is ``dftb+``). The DFTB
methods also need Slater-Koster parameter files, which DFTB+ does not ship: download
sets from https://github.com/dftbparams (for example 3ob-3-1 and mio-1-1) into one
directory, each in its own subdirectory, and set ``QCCODEC_DFTB_PARAMETERS`` to it
(or pass ``slater_koster_dir`` below). The xTB methods need no files.

How the codec talks to DFTB+ (https://dftbplus.org/documentation):

- ``model.method`` is the Hamiltonian: "DFTB3", "SCC-DFTB", "DFTB" (non-self-
  consistent), or "GFN2-xTB", "GFN1-xTB", "IPEA1-xTB".
- ``model.basis`` names the parameter set for the DFTB methods ("3ob-3-1" for DFTB3,
  "mio-1-1" for SCC-DFTB). The codec fills in the per-element settings the set
  prescribes (angular momenta, DFTB3 Hubbard derivatives, H-bond damping) from
  ``qcref.dftb_parameter_sets``. Cite the set's references: see its ``citation``.
- ``keywords`` are DFTB+ input (HSD) as a nested dictionary merged into what the
  codec writes. Strings are written verbatim in HSD syntax, so quote HSD strings
  ('"water"') and write typed blocks as text, one assignment per line (DFTB+ reads
  ``a = 1  b = 2`` on one line as a single value).
- DFTB+ stops on any keyword it does not recognize, and the error says which.
- Hessians come from finite differences of gradients at displaced geometries only,
  so a Hessian result has no energy or gradient at your structure; run a gradient
  for those.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure
from qcref import dftb_parameter_sets

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,  # open shells also need the set's SpinConstants in keywords
)

prog_input = ProgramInput(
    program="dftbplus",
    calctype=calctype,
    structure=structure,
    model=Model(method="DFTB3", basis="3ob-3-1"),  # or "SCC-DFTB"/"mio-1-1"
    # model=Model(method="GFN2-xTB"),  # xTB: no basis, no parameter files
    keywords={
        # "slater_koster_dir": "/path/to/dftb-params",  # or QCCODEC_DFTB_PARAMETERS
        #
        # DFTB+ settings, exactly as in its manual (HSD as nested dictionaries):
        "Hamiltonian": {
            "SCCTolerance": 1e-7,  # SCC convergence (charge)
            "MaxSCCIterations": 200,
            # D3(BJ) dispersion with the 3ob README's DFTB3-D3(BJ) parameters. Verbatim
            # HSD must keep one assignment per line, so use a multi-line string:
            # "Dispersion": """DftD3 {
            #     Damping = BeckeJohnson {
            #       a1 = 0.746
            #       a2 = 4.191
            #     }
            #     s6 = 1.0
            #     s8 = 3.209
            #   }""",
            # Implicit solvent: "Solvation" with a GeneralizedBorn or Cosmo block as in
            # the DFTB+ manual (it needs its solvent parameters spelled out).
        },
        # "Parallel": {"UseOmpThreads": True},  # threads come from OMP_NUM_THREADS
    },
)

output = compute(
    prog_input,
    # timeout=600,  # seconds before DFTB+ is killed
    # collect_files=True,  # keep detailed.out, results.tag, ... in output.results.files
    # executables={"dftbplus": "/path/to/dftb+"},  # or QCCOMPUTE_DFTBPLUS_EXECUTABLE
)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Hessian (Hartree/bohr^2), shape:", results.hessian.shape)
print("Mulliken charges:", results.atomic_charges)
print("Dipole (e*bohr):", results.scf_dipole_moment)
print("DFTB+ version:", results.provenance.program_version)
print("Cite:", dftb_parameter_sets.by_name("3ob-3-1").citation)
