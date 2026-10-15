"""TeraChem single point: energy, gradient, or Hessian.

Requires: ``pip install qccompute`` (it installs qccodec) and ``terachem`` on PATH.
TeraChem runs only on NVIDIA CUDA GPUs and needs a license file (from PetaChem). A
Docker launcher script that provides ``terachem`` without an install, with the license
given by TERACHEM_LICENSE_PATH, comes with the images at
https://hub.docker.com/r/mtzgroup/terachem.

How the codec talks to TeraChem (manual: https://mtzweb.stanford.edu/terachem/,
or the PDF manual shipped in the TeraChem ``doc`` directory):

- The codec writes ``tc.in``. ``run``, ``coordinates``, ``charge``, ``spinmult``,
  ``method``, and ``basis`` come from ``calctype``, ``structure``, and ``model``;
  setting any of them in ``keywords`` is an error.
- Every other ``keywords`` entry becomes one ``key value`` line in ``tc.in``, so use
  TeraChem's keyword names. Values are written lowercased (``True`` becomes ``true``);
  TeraChem itself expects ``yes``/``no`` for switches. TeraChem ignores unknown or
  misspelled keywords with only an "Unrecognized or unnecessary keywords" warning
  in ``output.logs``, so check there when a setting seems to have no effect.
- Unrestricted methods are spelled in the method name: "uhf", "ub3lyp", ...;
  restricted open shell is "rohf", "rob3lyp".
- Hessians (``run frequencies``) come from finite differences of analytic
  gradients, so they take 6N gradients; the gradient at the input geometry is
  returned as well. TeraChem stops with "Maximum component of gradient is too large"
  unless the structure is a stationary point; ``mincheck false`` turns that check
  off (set below because this example water is not at a minimum; energy and
  gradient runs list it as an unnecessary keyword, which is harmless).
- ``purify no`` is set below because density purification is TeraChem's default for
  large systems and can fail to converge for small molecules.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="terachem",
    calctype=calctype,
    structure=structure,
    model=Model(method="hf", basis="sto-3g"),  # e.g. "wb97x"/"6-31gss", "ub3lyp"
    keywords={
        # Any tc.in keyword, exactly as in TeraChem's manual:
        "purify": "no",
        "convthre": 1e-6,  # SCF convergence (wavefunction error)
        "threall": 1e-12,  # integral screening threshold
        "maxit": 100,  # SCF iterations
        "mincheck": "false",  # allow Hessians away from a stationary point
        # "precision": "double",  # "single", "mixed" (default), "dynamic", "double"
        # "dftd": "d3",  # empirical dispersion for DFT methods
        # "gpus": "1 0",  # number of GPUs, then the device indices to use
        #
        # Implicit solvent (COSMO); "epsilon" is the dielectric constant:
        "pcm": "cosmo",
        "epsilon": 78.39,  # water
        # "pcm_scale": 1.2,  # scale factor for the cavity atomic radii
        #
        # Excited states (TDDFT/CIS); the gradient is for state "cistarget":
        # "cis": "yes",
        # "cisnumstates": 3,
        # "cistarget": 1,
    },
)

output = compute(
    prog_input,
    # timeout=3600,  # seconds before TeraChem is killed
    # collect_files=True,  # keep every output file in output.results.files
    # collect_wfn=True,  # keep the orbitals (scr*/c0) for a later guess
    # executables={"terachem": "/path/to/terachem"},  # or QCCOMPUTE_TERACHEM_EXECUTABLE
)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Hessian (Hartree/bohr^2), shape:", results.hessian.shape)
print("Number of orbitals:", results.calcinfo_nmo)
if results.extras.get("excited_states"):
    print("Excited states:", results.extras["excited_states"])
print("TeraChem version:", results.provenance.program_version)
