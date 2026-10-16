"""Psi4 single point: energy, gradient, or Hessian.

Requires: ``pip install qccompute`` (it installs qccodec) and Psi4 on PATH, e.g.
``conda install -c conda-forge psi4`` (add ``pyddx`` for ddX implicit solvent).

How the codec talks to Psi4 (https://psicode.org/psi4manual/master/):

- ``model.method`` and ``model.basis`` become Psi4's method string, e.g. "wb97x-d/def2-svp".
  Leave ``basis`` out for composite methods such as "mp2/cc-pv[dt]z".
- ``keywords`` go to ``psi4.set_options`` unchanged, so use Psi4's option names.
  Module-scoped options are written "module__option", e.g. "scf__maxiter".
- A few keywords are not Psi4 options and are routed where Psi4 expects them:
  ``external_potentials`` (point charges), ``dertype``, ``project_trans``,
  ``project_rot`` (driver arguments) and ``symmetry`` (the molecule block).
- The molecule stays in your coordinates: Psi4 is told not to move or reorient it,
  so gradients and Hessians line up with ``structure``.
- Psi4 errors (unknown option, SCF not converged, ...) raise with Psi4's message.
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
    program="psi4",
    calctype=calctype,
    structure=structure,
    model=Model(method="wb97x-d", basis="def2-svp"),  # e.g. "b3lyp-d3bj"/"def2-tzvp"
    keywords={
        # Any Psi4 option, exactly as in Psi4's documentation:
        # "scf_type": "pk",  # exact integrals instead of density fitting
        "e_convergence": 1e-8,
        "d_convergence": 1e-8,
        "scf__maxiter": 100,  # module-scoped option
        # "reference": "uhf",  # open shells: also set structure multiplicity
        #
        # Implicit solvent (ddX; needs the pyddx package next to Psi4):
        "ddx": True,
        "ddx_model": "pcm",  # or "cosmo", "lpb"
        "ddx_solvent": "water",
        #
        # Point charges for QM/MM electrostatic embedding, [charge, [x, y, z]] in
        # bohr. Psi4 1.11 handles them only for energies and gradients; the
        # gradient on the charges is returned as point_charge_gradient.
        # "external_potentials": [[-0.8, [0.0, 0.0, 5.0]], [0.4, [1.0, 1.0, 6.0]]],
        #
        # Symmetry: Psi4 detects any symmetry your frame already has without
        # moving atoms. Name a point group to override, or "c1" to turn it off.
        # "symmetry": "c1",
        #
        # Hessians: frequency analysis projects out translations and rotations
        # only at stationary points unless told otherwise.
        # "project_rot": True,
        # "dertype": 1,  # Hessian by finite differences of analytic gradients
    },
    # Arguments for the psi4 command itself: threads and memory.
    cmdline_args=["-n", "1", "--memory", "2GB"],
)

output = compute(
    prog_input,
    # timeout=3600,  # seconds before Psi4 is killed
    # collect_files=True,  # keep every output file in output.results.files
    # executables={"psi4": "/path/to/psi4"},  # or set QCCOMPUTE_PSI4_EXECUTABLE
)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Frequencies (cm^-1):", results.freqs_wavenumber)
print("Mulliken charges:", results.atomic_charges)
print("Dipole (e*bohr):", results.scf_dipole_moment)
print("Psi4 variables:", sorted(results.extras["psi4_variables"])[:5], "...")
