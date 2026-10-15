"""Psi4 driver script, run by Psi4's own interpreter as ``psi4 -k input.py``.

The psi4 codec writes this file as ``input.py`` next to ``job.json`` (what to run)
and decodes the ``results.json`` it writes. It must not import qccodec or qcdata:
it runs in Psi4's environment, not the caller's. Everything Psi4-specific happens
inside ``main``, so the helpers can be tested without Psi4.

``job.json``::

    {"calctype": "energy" | "gradient" | "hessian" | "optimization",
     "method": "hf/sto-3g",                  # first argument to psi4.energy() etc.
     "options": {...},                       # passed to psi4.set_options() unchanged
     "kwargs": {...},                        # extra driver keyword arguments
     "molecule": {"symbols": [...], "geometry": [[x, y, z], ...],  # bohr
                  "charge": 0, "multiplicity": 1}}

``results.json`` holds plain numbers and lists (atomic units, bohr), plus ``error``
when the calculation raised. The process exits 1 on error.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

JOB_FILE = "job.json"
RESULTS_FILE = "results.json"


def molecule_string(molecule: dict[str, Any]) -> str:
    """Build a Psi4 molecule block in bohr that Psi4 will not move or symmetrize.

    The frame is fixed (``no_com``, ``no_reorient``) so gradients and Hessians come
    back in the input's coordinates, and ``symmetry c1`` keeps orbital data in one
    block in input order.
    """
    lines = [f"{molecule['charge']} {molecule['multiplicity']}"]
    for symbol, (x, y, z) in zip(
        molecule["symbols"], molecule["geometry"], strict=True
    ):
        lines.append(f"{symbol} {x!r} {y!r} {z!r}")
    lines += ["units bohr", "no_com", "no_reorient", "symmetry c1"]
    return "\n".join(lines)


def wavenumbers(omega: list[complex]) -> list[float]:
    """Convert Psi4's complex frequencies to floats; imaginary ones become negative."""
    return [w.real if abs(w.imag) < 1e-12 else -abs(w.imag) for w in omega]


def to_plain(value: Any) -> Any:
    """Convert numpy arrays, Psi4 matrices, and numbers to JSON-serializable values."""
    if hasattr(value, "np"):  # psi4.core.Matrix / Vector
        value = value.np
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def _try(results: dict[str, Any], key: str, getter: Any) -> None:
    """Store ``getter()`` under ``key``; skip quantities a method does not provide."""
    try:
        value = getter()
    except Exception:
        return
    if value is not None:
        results[key] = to_plain(value)


def _wavefunction_results(psi4: Any, wfn: Any, molecule: Any) -> dict[str, Any]:
    """Collect what the final wavefunction and Psi4 variables provide."""
    results: dict[str, Any] = {}
    _try(results, "nuclear_repulsion_energy", molecule.nuclear_repulsion_energy)
    _try(results, "nbasis", wfn.nso)
    _try(results, "nmo", wfn.nmo)
    _try(results, "nalpha", wfn.nalpha)
    _try(results, "nbeta", wfn.nbeta)
    _try(results, "eigenvalues_a", wfn.epsilon_a)
    _try(results, "eigenvalues_b", wfn.epsilon_b)
    _try(results, "occupations_a", wfn.occupation_a)
    _try(results, "occupations_b", wfn.occupation_b)
    _try(results, "dipole", lambda: psi4.variable("SCF DIPOLE"))
    try:
        psi4.oeprop(wfn, "MULLIKEN_CHARGES")
        _try(results, "mulliken_charges", wfn.atomic_point_charges)
    except Exception:
        pass
    if wfn.external_pot() is not None:
        _try(
            results,
            "point_charge_gradient",
            lambda: wfn.external_pot().gradient_on_charges(),
        )
    variables = {}
    for name, value in psi4.core.variables().items():
        if isinstance(value, (int, float)):
            variables[name] = float(value)
    results["variables"] = variables
    return results


def run(psi4: Any, job: dict[str, Any], results: dict[str, Any]) -> None:
    """Run the calculation described by ``job``, filling ``results`` as it goes."""
    results["psi4_version"] = psi4.__version__
    molecule = psi4.geometry(molecule_string(job["molecule"]))
    psi4.set_options(job["options"])
    calctype = job["calctype"]
    kwargs = {**job["kwargs"], "return_wfn": True}

    if calctype == "optimization":
        energy, wfn, history = psi4.optimize(
            job["method"], return_history=True, **kwargs
        )
        results["trajectory"] = [
            {"geometry": to_plain(xyz), "energy": float(e), "gradient": to_plain(g)}
            for xyz, e, g in zip(
                history["coordinates"],
                history["energy"],
                history["gradient"],
                strict=True,
            )
        ]
    elif calctype == "hessian":
        energy, wfn = psi4.frequency(job["method"], **kwargs)
        results["hessian"] = to_plain(wfn.hessian())
        _try(results, "gradient", wfn.gradient)
        vib = wfn.frequency_analysis
        vibrational = [i for i, kind in enumerate(vib["TRV"].data) if kind == "V"]
        omega = list(vib["omega"].data)
        modes = vib["x"].data  # columns are un-mass-weighted displacements, bohr
        results["frequencies"] = wavenumbers([omega[i] for i in vibrational])
        results["normal_modes"] = [
            to_plain(modes[:, i].reshape(-1, 3)) for i in vibrational
        ]
    elif calctype == "gradient":
        gradient, wfn = psi4.gradient(job["method"], **kwargs)
        energy = psi4.variable("CURRENT ENERGY")
        results["gradient"] = to_plain(gradient)
    elif calctype == "energy":
        energy, wfn = psi4.energy(job["method"], **kwargs)
    else:
        raise ValueError(f"Unsupported calctype {calctype!r}")

    results["energy"] = float(energy)
    results.update(_wavefunction_results(psi4, wfn, molecule))


def main() -> int:
    """Run ``job.json`` and write ``results.json``; return the process exit code."""
    with open(JOB_FILE) as f:
        job = json.load(f)
    results: dict[str, Any] = {}
    status = 0
    try:
        import psi4  # type: ignore[import-not-found]

        run(psi4, job, results)
    except Exception as exc:
        traceback.print_exc()
        results["error"] = f"{type(exc).__name__}: {exc}"
        status = 1
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)
    return status


if __name__ == "__main__":
    # Psi4 reports any SystemExit, even status 0, as a crash; exit only on failure.
    if main():
        sys.exit(1)
