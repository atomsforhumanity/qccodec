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
        assert struct.charge == -2 and struct.multiplicity == 3


def test_parse_rotamers_charge_multiplicity_updates(test_data_dir, prog_input_factory):
    data = parse_rotamers(_context(test_data_dir, prog_input_factory, 3, 5))
    for struct in data["rotamers"]:
        assert struct.charge == 3 and struct.multiplicity == 5
