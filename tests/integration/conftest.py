"""Fixtures for integration tests that run real programs through qccompute.

Integration tests are marked ``integration`` and deselected by default. Run them with::

    uv sync --group integration --extra geometric
    uv run pytest -m integration

Each test skips when its program is not on ``PATH``. Pass ``--record-cases`` to save
each run that names a case as ``tests/data/<program>/<case>_<version>/`` for the
offline parser tests, with this machine's home directory and hostname replaced in
text files; review and prune the files before committing them. Existing
cases are never overwritten, so delete one first to re-record it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qcdata import CalcType, ProgramInput, ProgramOutput, SinglePointData, Structure

from qccodec import get_codec
from qccodec.native import SubprocessInput
from qccodec.testing import load_case, run_case

DATA_DIR = Path(__file__).parent.parent / "data"
# Output files larger than this are not copied into recorded cases (wavefunctions,
# checkpoints, ...). Parsers that need them get a hand-made case instead.
MAX_RECORDED_FILE_BYTES = 1_000_000


def required_programs() -> set[str]:
    """Programs that must not be skipped, from ``QCCODEC_REQUIRE_PROGRAMS``.

    CI sets this (comma-separated) so a runner image that lacks a program fails its
    tests instead of silently skipping them.
    """
    return set(os.environ.get("QCCODEC_REQUIRE_PROGRAMS", "").replace(",", " ").split())


def requires(*programs: str) -> pytest.MarkDecorator:
    """Skip a test unless every program is on ``PATH`` or listed as required."""
    required = required_programs()
    missing = [p for p in programs if shutil.which(p) is None and p not in required]
    return pytest.mark.skipif(
        bool(missing), reason=f"not on PATH: {', '.join(missing)}"
    )


def requires_modules(program: str, *modules: str) -> pytest.MarkDecorator:
    """Skip a test unless a library codec's modules import, or ``program`` is required.

    tblite and the machine-learned potentials are Python libraries run in-process, so
    what matters is whether they import, not what is on ``PATH``.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if program in required_programs():
        missing = []
    return pytest.mark.skipif(
        bool(missing), reason=f"not importable: {', '.join(missing)}"
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--record-cases``."""
    parser.addoption(
        "--record-cases",
        action="store_true",
        default=False,
        help="Save integration runs that name a case as tests/data/<program>/<case>/.",
    )


Run = Callable[..., ProgramOutput]


@pytest.fixture
def run(request: pytest.FixtureRequest, tmp_path: Path) -> Run:
    """Return ``run(input, case=None, **compute_kwargs)``, which calls qccompute.

    With ``--record-cases`` and a ``case`` name, the run happens in a kept scratch
    directory and is saved as a fixture case, which is then replayed through
    ``qccodec.testing.run_case`` to prove it is self-consistent.
    """
    # Imported here so the unit-test suite collects without the integration group.
    from qccompute import compute

    record = bool(request.config.getoption("--record-cases"))

    def _run(
        input_data: ProgramInput, case: str | None = None, **kwargs: Any
    ) -> ProgramOutput:
        scratch = tmp_path / "scratch"
        if record and case:
            kwargs.update(scratch_dir=scratch, rm_scratch_dir=False)
        try:
            output = compute(input_data, **kwargs)
        except Exception as exc:
            # Show the program's own output; pytest prints it under "Captured stdout".
            logs = getattr(exc, "logs", None)
            if logs:
                print(f"--- last 100 lines of {input_data.program} output ---")
                print("\n".join(logs.splitlines()[-100:]))
            raise
        if record and case:
            _record_case(output, scratch, DATA_DIR / input_data.program, case)
        return output

    return _run


def _record_case(output: ProgramOutput, scratch: Path, root: Path, case: str) -> None:
    """Write ``output`` and its scratch directory as ``root/<case>_<version>``.

    The program version in the name keeps fixtures from different releases apart.
    """
    input_data = output.input_data
    codec = get_codec(input_data.program)
    native = codec.encode(input_data)
    version = output.results.provenance.program_version or "unknown"
    if codec.version_key is not None:
        version = codec.version_key(version)
    dest = root / f"{case}_{re.sub(r'[^0-9A-Za-z.]+', '_', version).strip('_')}"
    if dest.exists():
        # Fixtures from older program versions cannot be regenerated; never replace one.
        pytest.fail(f"{dest} exists; delete it first to re-record this case.")
    (dest / "encoded").mkdir(parents=True)

    (dest / "input.json").write_text(input_data.model_dump_json(indent=2) + "\n")
    expected = _without_execution(json.loads(output.results.model_dump_json()))
    (dest / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    if output.logs is not None:
        (dest / "stdout").write_text(output.logs)
    if isinstance(native, SubprocessInput):
        (dest / "encoded" / "command.txt").write_text("\n".join(native.command) + "\n")
    for name, content in native.files.items():
        path = dest / "encoded" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)

    for path in sorted(p for p in scratch.rglob("*") if p.is_file()):
        name = path.relative_to(scratch).as_posix()
        if name in native.files or path.stat().st_size > MAX_RECORDED_FILE_BYTES:
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    _scrub(dest)
    # A recorded case must replay offline, or it is useless as a fixture.
    run_case(codec, load_case(dest), rtol=1e-6)


def _scrub(dest: Path) -> None:
    """Replace this machine's home directory and hostname in recorded text files."""
    replacements = {str(Path.home()): "/home/user", socket.gethostname(): "localhost"}
    for path in dest.rglob("*"):
        if not path.is_file() or path.parent.name == "encoded":
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        scrubbed = text
        for old, new in replacements.items():
            scrubbed = scrubbed.replace(old, new)
        if scrubbed != text:
            path.write_text(scrubbed)


def _without_execution(value: Any) -> Any:
    """Drop ``execution`` metadata (local paths, timings); fixture replay ignores it."""
    if isinstance(value, dict):
        return {k: _without_execution(v) for k, v in value.items() if k != "execution"}
    if isinstance(value, list):
        return [_without_execution(v) for v in value]
    return value


@pytest.fixture(scope="session")
def hydrogen() -> Structure:
    """H2 at 1.4 bohr. Reference values depend on this geometry; do not change."""
    return Structure(
        symbols=["H", "H"], geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    )


@pytest.fixture(scope="session")
def water() -> Structure:
    """Distorted water (bohr). Reference values depend on this geometry; do not change."""
    return Structure(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]
        ),
    )


def requires_hugging_face(program: str) -> pytest.MarkDecorator:
    """Skip unless this machine has a Hugging Face login (gated FAIR-Chem weights)."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        token = None
    else:
        token = get_token()
    return pytest.mark.skipif(
        not token and program not in required_programs(),
        reason="gated weights need `hf auth login` or HF_TOKEN",
    )


def finite_difference_check(
    run: Run,
    input_data: ProgramInput,
    *,
    step: float = 1e-3,
    atol: float = 1e-5,
    hessian: bool = False,
) -> None:
    """Compare a gradient (or Hessian) with central differences of energies (gradients).

    Catches unit, sign, and frame errors in codecs that convert library results. Only
    the first atom's coordinates are displaced, which is enough for that and cheap.
    """
    calctype = CalcType.hessian if hessian else CalcType.gradient
    lower = CalcType.gradient if hessian else CalcType.energy
    analytic = run(input_data.model_copy(update={"calctype": calctype})).results
    reference = np.asarray(analytic.hessian if hessian else analytic.gradient)
    geometry = input_data.require_structure().geometry
    for k in range(3):
        values = []
        for sign in (1.0, -1.0):
            moved = geometry.copy()
            moved[0, k] += sign * step
            structure = input_data.require_structure().model_copy(
                update={"geometry": moved}
            )
            result = run(
                input_data.model_copy(
                    update={"calctype": lower, "structure": structure}
                )
            ).results
            values.append(np.ravel(result.gradient) if hessian else result.energy)
        numeric = (values[0] - values[1]) / (2 * step)
        expected = reference[k] if hessian else reference[0, k]
        assert np.allclose(expected, numeric, atol=atol), (k, expected, numeric)


def batch_molecules(charged: bool = True) -> list[Structure]:
    """Molecules of different sizes for batch tests; OH- too unless ``charged`` is False."""
    rng = np.random.default_rng(0)
    water = np.array([[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]])
    molecules = [
        Structure(symbols=["O", "H", "H"], geometry=water + rng.normal(0, 0.05, (3, 3)))
        for _ in range(3)
    ]
    molecules.append(
        Structure(
            symbols=["C", "H", "H", "H", "H"],
            geometry=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.19, 1.19, 1.19],
                    [-1.19, -1.19, 1.19],
                    [-1.19, 1.19, -1.19],
                    [1.19, -1.19, -1.19],
                ]
            ),
        )
    )
    if charged:
        molecules.append(
            Structure(
                symbols=["O", "H"],
                geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]]),
                charge=-1,
            )
        )
    return molecules


def check_batch_matches_single(
    run: Run, inputs: list[ProgramInput], *, atol: float = 1e-6
) -> None:
    """A codec's batch gives each molecule what computing it alone gives."""
    from qccodec import PythonOutput

    codec = get_codec(inputs[0].program)
    batch = codec.encode_batch(inputs)
    values = batch.function(*batch.args, **batch.kwargs)
    assert len(values) == len(inputs)
    for inp, value in zip(inputs, values, strict=True):
        batched = codec.decode(inp, PythonOutput(value=value))
        alone = run(inp).results
        assert isinstance(batched, SinglePointData)
        assert isinstance(alone, SinglePointData)
        assert batched.energy is not None and alone.energy is not None
        assert np.isclose(batched.energy, alone.energy, atol=atol)
        assert batched.gradient is not None and alone.gradient is not None
        assert np.allclose(batched.gradient, alone.gradient, atol=atol)
