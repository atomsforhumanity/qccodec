"""An ASE calculator whose energies and forces are calculations of any qccodec program.

ASE-based tools (Sella, ASE's optimizers, NEB, and molecular dynamics) ask an
``ase.calculators.calculator.Calculator`` for energies and forces. ``ProgramCalculator``
answers each request with one calculation: it builds a ``ProgramInput`` from a
``ProgramSpec`` and the atoms, and hands it to a ``compute`` function, so the program
runs through its codec rather than through an ASE calculator of its own::

    from ase.optimize import BFGS
    from qccompute import compute
    from qccodec.ase import ProgramCalculator

    atoms.calc = ProgramCalculator(spec, compute)  # spec: a gradient ProgramSpec
    BFGS(atoms).run(fmax=0.01)

qccodec translates and never runs programs, so the caller supplies ``compute``:
``qccompute.compute``, or the child-calculation callback inside a codec such as
``sella``. Requires ASE (``pip install ase``; ``qccodec[sella]`` includes it).

Units are converted at the boundary: ASE positions in angstrom become a ``Structure``
in bohr, and energies and gradients in Hartree and Hartree/bohr come back as eV and
eV/angstrom forces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import CalculationFailed, Calculator, all_changes
from numpy.typing import NDArray
from qcdata import (
    CalcType,
    Data,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    SinglePointData,
    Structure,
)
from qcref import constants as c

Compute = Callable[[ProgramInput], ProgramOutput[ProgramInput, Data]]


def atoms_structure(atoms: Atoms, template: Structure) -> Structure:
    """Return ``template`` (charge, multiplicity, ...) at the atoms' positions.

    Raises:
        ValueError: The atoms' elements differ from the template's.
    """
    symbols = atoms.get_chemical_symbols()
    if symbols != list(template.symbols):
        raise ValueError(
            f"The atoms ({''.join(symbols)}) do not match the structure "
            f"({''.join(template.symbols)})."
        )
    return Structure.model_validate(
        {
            **template.model_dump(),
            "geometry": atoms.get_positions() * c.ANGSTROM_TO_BOHR,
        }
    )


class ProgramCalculator(Calculator):  # type: ignore[misc]
    """An ASE calculator running a qccodec program for every energy and force.

    Attributes:
        outputs: Every output computed, in order, including a failed one.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        spec: ProgramSpec,
        compute: Compute,
        *,
        charge: int = 0,
        multiplicity: int = 1,
        structure: Structure | None = None,
        structures: dict[str, Structure] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the calculator.

        Args:
            spec: The program, model, and keywords; ``calctype`` ``gradient`` gives
                energies and forces, ``energy`` only energies.
            compute: Runs one input and returns its output, such as
                ``qccompute.compute``.
            charge: Total charge, when no ``structure`` is given.
            multiplicity: Spin multiplicity, when no ``structure`` is given.
            structure: A structure whose charge, multiplicity, and identifiers every
                calculation keeps; its elements must match the atoms'.
            structures: Named structures for the program, such as QM/MM environments.
            **kwargs: Passed to ``ase.calculators.calculator.Calculator``.

        Raises:
            ValueError: ``spec`` is not an energy or gradient calculation.
        """
        if spec.calctype not in (CalcType.energy, CalcType.gradient):
            raise ValueError(
                f"ProgramCalculator needs an energy or gradient spec, not "
                f"{spec.calctype.value}."
            )
        super().__init__(**kwargs)
        self.spec = spec
        self.compute = compute
        self.charge = charge
        self.multiplicity = multiplicity
        self.template = structure
        self.structures = structures or {}
        self.outputs: list[ProgramOutput[ProgramInput, Data]] = []

    def structure(self, atoms: Atoms) -> Structure:
        """Return the qcdata structure of the atoms, in bohr."""
        if self.template is not None:
            return atoms_structure(atoms, self.template)
        return Structure(
            symbols=atoms.get_chemical_symbols(),
            geometry=atoms.get_positions() * c.ANGSTROM_TO_BOHR,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = tuple(all_changes),
    ) -> None:
        """Compute the energy and, for a gradient spec, the forces of the atoms.

        Raises:
            CalculationFailed: The calculation failed or lacks the results; the
                output is attached as ``.output``.
        """
        super().calculate(atoms, properties, system_changes)
        child = ProgramInput.from_spec(
            self.spec, self.structure(self.atoms), structures=self.structures
        )
        output = self.compute(child)
        self.outputs.append(output)
        data = output.results
        reason = None
        if not output.success:
            reason = f"The {self.spec.program} calculation failed."
        elif not isinstance(data, SinglePointData) or data.energy is None:
            reason = f"The {self.spec.program} calculation returned no energy."
        elif self.spec.calctype == CalcType.gradient and data.gradient is None:
            reason = f"The {self.spec.program} calculation returned no gradient."
        if reason is not None:
            error = CalculationFailed(reason)
            error.output = output  # type: ignore[attr-defined]
            raise error
        assert isinstance(data, SinglePointData) and data.energy is not None
        energy = data.energy * c.HARTREE_TO_EV
        self.results = {"energy": energy, "free_energy": energy}
        if data.gradient is not None:
            self.results["forces"] = -gradient_to_ase(data.gradient)


def gradient_to_ase(gradient: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a gradient in Hartree/bohr to eV/angstrom."""
    return np.asarray(gradient) * c.HARTREE_TO_EV * c.ANGSTROM_TO_BOHR


def hessian_to_ase(hessian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a Hessian in Hartree/bohr^2 to eV/angstrom^2."""
    return np.asarray(hessian) * c.HARTREE_TO_EV * c.ANGSTROM_TO_BOHR**2
