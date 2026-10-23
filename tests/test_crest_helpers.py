"""CREST directory parsers take charge and multiplicity from the input."""

from qcdata import CalcType, ProgramInput

from qccodec import Context, get_codec
from qccodec.programs.crest import parse_conformers, parse_rotamers


def _context(test_data_dir, prog_input_factory, charge, multiplicity, **options):
    prog_input = prog_input_factory("conformer_search", program="crest")
    values = prog_input.model_dump()
    values["structure"]["charge"] = charge
    values["structure"]["multiplicity"] = multiplicity
    return Context(
        get_codec("crest"),
        CalcType.conformer_search,
        directory=test_data_dir / "crest" / "conformers",
        input_data=ProgramInput(**values),
        options={"collect_rotamers": True, **options},
    )


def test_parse_conformers_charge_multiplicity_updates(
    test_data_dir, prog_input_factory
):
    data = parse_conformers(_context(test_data_dir, prog_input_factory, -2, 3))
    for struct in data["conformers"]:
        assert struct.structure.charge == -2 and struct.structure.multiplicity == 3


def test_parse_rotamers_charge_multiplicity_updates(test_data_dir, prog_input_factory):
    data = parse_rotamers(_context(test_data_dir, prog_input_factory, 3, 5))
    for struct in data["rotamers"]:
        assert struct.structure.charge == 3 and struct.structure.multiplicity == 5


def test_trajectory_frames_return_to_the_input_frame() -> None:
    """CREST's center-of-mass shift is undone; anything else is left alone."""
    import numpy as np
    from qcdata import Structure

    from qccodec.programs.crest import _to_input_frame

    start = Structure(
        symbols=["O", "H", "H"],
        geometry=np.array([[0.0, 0.0, 0.0], [0.5, 1.7, 0.5], [1.1, -0.5, -1.4]]),
    )
    shift = np.array([0.1, -0.2, 0.3])
    frames = [
        start.model_copy(update={"geometry": start.geometry - shift}),
        start.model_copy(update={"geometry": start.geometry - shift + 0.01}),
    ]
    restored = _to_input_frame(frames, start)
    assert np.allclose(restored[0].geometry, start.geometry)
    assert np.allclose(restored[1].geometry, start.geometry + 0.01)

    rotated = [start.model_copy(update={"geometry": start.geometry[:, [1, 0, 2]]})]
    assert _to_input_frame(rotated, start) is rotated
