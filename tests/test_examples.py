"""Every built-in program has an example script for each kind of calculation it runs."""

from pathlib import Path

import pytest
from qcdata import CalcType

from qccodec import get_codec
from qccodec.registry import _BUILTIN

EXAMPLES = Path(__file__).parent.parent / "examples"

# One script per group: single points differ only in the calctype line.
GROUPS = {
    CalcType.energy: "single_point",
    CalcType.gradient: "single_point",
    CalcType.hessian: "single_point",
    CalcType.optimization: "optimization",
    CalcType.transition_state: "transition_state",
    CalcType.conformer_search: "conformer_search",
    CalcType.irc: "irc",
    CalcType.path_optimization: "path_optimization",
}


# Lists of inputs: required where the codec batches, optional for any program.
MANY = "many_calculations"


@pytest.mark.parametrize("program", sorted(_BUILTIN))
def test_every_calctype_has_an_example(program: str) -> None:
    """examples/<program>/ holds one script per calculation-type group, and no others.

    Codecs that batch also show a list of inputs in ``many_calculations.py``, which any
    program may have.
    """
    codec = get_codec(program)
    expected = {GROUPS[calctype] for calctype in codec.calctypes}
    if codec._encode_batch is not None:
        expected.add(MANY)
    found = {path.stem for path in (EXAMPLES / program).glob("*.py")}
    if codec._encode_batch is None:
        found.discard(MANY)
    assert found == expected, f"examples/{program}/ should hold {sorted(expected)}"


def test_no_examples_for_unknown_programs() -> None:
    """Every example directory belongs to a built-in program."""
    directories = {path.name for path in EXAMPLES.iterdir() if path.is_dir()}
    assert directories <= set(_BUILTIN), sorted(directories - set(_BUILTIN))


def test_every_example_is_on_the_docs_site() -> None:
    """Each script is included in its program's documentation page."""
    pages = "".join(p.read_text() for p in (EXAMPLES.parent / "docs").rglob("*.md"))
    for script in EXAMPLES.rglob("*.py"):
        name = script.relative_to(EXAMPLES.parent).as_posix()
        assert f"{{!{name}!}}" in pages, f"{name} is not included in any docs page"
