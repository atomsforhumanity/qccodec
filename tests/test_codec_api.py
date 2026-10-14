"""The Codec class: registration rules, the decode pipeline, and configuration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from qcdata import CalcType, FileInput, Model, ProgramInput, SinglePointData
from qcdata.utils import water

import qccodec
from qccodec import (
    Codec,
    Context,
    NativeOutput,
    SubprocessInput,
    SubprocessOutput,
    get_codec,
)
from qccodec.exceptions import (
    CalculationError,
    CodecNotFoundError,
    DataCollectorError,
    EncoderError,
    MatchNotFoundError,
    ParserError,
    RegistryError,
)


def energy(contents: str) -> float:
    """Parse a directly callable scalar value."""
    return float(contents)


def single_point(data: object) -> SinglePointData:
    """Narrow a parse result for attribute checks."""
    assert isinstance(data, SinglePointData)
    return data


def no_decode(inp: object, out: NativeOutput) -> SinglePointData:
    """A decoder stand-in for registration tests; never called."""
    raise AssertionError("decoder should not run")


def test_parser_decorator_returns_function_and_is_local() -> None:
    """Decorating records a spec on one codec only and preserves direct calls."""
    first, second = (
        Codec("first", {CalcType.energy}),
        Codec("second", {CalcType.energy}),
    )
    decorated = first.parser(target="energy", calctypes=[CalcType.energy])(energy)
    assert decorated is energy and decorated("-1") == -1
    assert len(first.parsers) == 1 and second.parsers == []
    spec = first.parsers[0]
    assert spec.file == "stdout" and spec.calctypes == {CalcType.energy}
    assert spec.required and spec.target == "energy"


def test_parser_validation() -> None:
    """Targets are required outside directory parsers and may not collide."""
    codec = Codec("local", {CalcType.energy})
    with pytest.raises(RegistryError, match="must have a target"):
        codec.parser()(energy)
    with pytest.raises(RegistryError, match="string or tuple"):
        codec.parser(target=["energy"])(energy)  # type: ignore[arg-type]
    codec.parser(target="energy", calctypes=[CalcType.energy])(energy)
    with pytest.raises(RegistryError, match="Duplicate parser target"):
        codec.parser(
            file="other",
            target="energy",
            calctypes=[CalcType.energy, CalcType.gradient],
        )(energy)
    codec.parser(target="energy", calctypes=[CalcType.gradient])(energy)
    codec.parser(file="directory")(energy)
    codec.parser(target=("provenance", "program_version"), required=False)(energy)
    assert len(codec.parsers) == 4
    assert codec.parsers[2].target is None
    assert codec.parsers[3].calctypes == frozenset(CalcType)


def test_decoder_and_parsers_are_exclusive() -> None:
    """A custom decoder replaces the parser pipeline, so both is a mistake."""
    codec = Codec("local", {CalcType.energy}, decode=no_decode)
    with pytest.raises(RegistryError, match="decoder"):
        codec.parser(target="energy")(energy)
    codec = Codec("local", {CalcType.energy})
    codec.parser(target="energy")(energy)
    with pytest.raises(RegistryError, match="parsers"):
        codec.decoder(no_decode)


def test_encode_validates_program_and_calctype(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Encode guards the program name and supported calculation types."""
    codec = Codec("local", {CalcType.energy})
    with pytest.raises(EncoderError, match="no encoder"):
        codec.encode(prog_input_factory("energy", "local"))
    codec.encoder(lambda inp: SubprocessInput(("local",)))
    native = codec.encode(prog_input_factory("energy", "local"))
    assert isinstance(native, SubprocessInput) and native.command == ("local",)
    with pytest.raises(EncoderError, match="cannot encode"):
        codec.encode(prog_input_factory("energy", "other"))
    with pytest.raises(EncoderError, match="does not support gradient"):
        codec.encode(prog_input_factory("gradient", "local"))


def test_pipeline_collects_values_and_tolerates_optional(tmp_path: Path) -> None:
    """Values land on their targets; optional misses and dict merges work."""
    codec = Codec("local", {CalcType.energy})
    codec.parser(target="energy", calctypes=[CalcType.energy])(energy)

    @codec.parser(target="gradient", required=False)
    def gradient(contents: str) -> list[list[float]]:
        """Simulate an optional scientific value missing from native output."""
        raise MatchNotFoundError("gradient", contents)

    @codec.parser(target="wavefunction")
    def orbitals(contents: str) -> dict[str, list[float]]:
        """Return a dictionary assigned at an explicit target, not merged."""
        return {"scf_eigenvalues_a": [-1]}

    @codec.parser(file="directory")
    def extras(ctx: Context) -> dict[str, object]:
        """Return a mapping merged into the result at the root."""
        assert ctx.codec is codec and ctx.calctype == CalcType.energy
        return {"calcinfo_natoms": 3}

    data = codec.parse("energy", stdout="-1", directory=tmp_path)
    assert isinstance(data, SinglePointData)
    assert data.energy == -1 and data.gradient is None
    assert data.wavefunction is not None
    assert data.wavefunction.scf_eigenvalues_a is not None
    assert data.wavefunction.scf_eigenvalues_a.tolist() == [-1]
    assert data.calcinfo_natoms == 3
    assert data.provenance.program == "local"
    as_dict = codec.parse("energy", stdout="-1", directory=tmp_path, as_dict=True)
    assert as_dict["energy"] == -1


def test_required_values_raise_with_partial_data(tmp_path: Path) -> None:
    """Every parser runs before a required miss is raised with the recovered data."""
    codec = Codec("local", {CalcType.energy})

    @codec.parser(target="energy")
    def missing(contents: str) -> float:
        """Always fail to match."""
        raise MatchNotFoundError("energy", contents)

    codec.parser(target="calcinfo_natoms")(lambda contents: 3)
    with pytest.raises(MatchNotFoundError) as raised:
        codec.parse("energy", stdout="x")
    assert single_point(raised.value.data).calcinfo_natoms == 3
    assert (
        single_point(codec.parse("energy", stdout="x", failed=True)).calcinfo_natoms
        == 3
    )

    with pytest.raises(ParserError, match="stdout") as missing_stdout:
        codec.parse("energy", directory=tmp_path)
    assert missing_stdout.value.data is not None


def test_file_patterns_and_context_functions(tmp_path: Path) -> None:
    """Parsers read named files, globs, or names computed from the context."""
    (tmp_path / "job.engrad").write_text("-2")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "optim.xyz").write_text("-3")
    codec = Codec("local", {CalcType.energy})
    codec.parser(file="job.engrad", target="energy")(energy)
    codec.parser(file="*/optim.xyz", target="calcinfo_natoms")(
        lambda c: int(float(c)) * -1
    )
    codec.parser(
        file=lambda ctx: f"{(ctx.stdout or '').strip()}.engrad",
        target="calcinfo_nmo",
        required=False,
    )(lambda c: int(float(c)) * -2)
    data = single_point(codec.parse("energy", stdout="job", directory=tmp_path))
    assert data.energy == -2 and data.calcinfo_natoms == 3 and data.calcinfo_nmo == 4

    with pytest.raises(ParserError, match="Missing required output artifact") as raised:
        codec.parse("energy", directory=tmp_path / "sub")
    assert "job.engrad" in str(raised.value)
    partial = codec.parse("energy", directory=tmp_path / "sub", failed=True)
    assert single_point(partial).energy is None

    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "optim.xyz").write_text("-3")
    with pytest.raises(ParserError, match="matched several files"):
        codec.parse("energy", stdout="job", directory=tmp_path)
    ctx = Context(codec, CalcType.energy)
    with pytest.raises(FileNotFoundError):
        ctx.path("anything")


def test_failure_hook_recovers_then_raises(tmp_path: Path) -> None:
    """A reported failure yields partial data on the raised CalculationError."""
    codec = Codec("local", {CalcType.gradient})
    codec.parser(target="energy")(lambda contents: -1.0)

    @codec.parser(target="gradient")
    def gradient(contents: str) -> list[list[float]]:
        """Fail like a parser would on a truncated log."""
        raise MatchNotFoundError("gradient", contents)

    codec.failure(
        lambda ctx: "program said FAILED" if "FAILED" in (ctx.stdout or "") else None
    )
    with pytest.raises(CalculationError, match="FAILED") as raised:
        codec.parse("gradient", stdout="FAILED")
    assert single_point(raised.value.data).energy == -1
    assert (
        single_point(codec.parse("gradient", stdout="FAILED", failed=True)).energy == -1
    )
    with pytest.raises(MatchNotFoundError):
        codec.parse("gradient", stdout="fine")
    with pytest.raises(CalculationError):
        codec.decode(
            ProgramInput(
                program="local",
                calctype=CalcType.gradient,
                structure=water,
                model=Model(method="hf"),
            ),
            SubprocessOutput(stdout="FAILED", directory=tmp_path, returncode=0),
        )


def test_duplicate_assignment_is_rejected() -> None:
    """Two directory parsers may not fill the same key."""
    codec = Codec("local", {CalcType.energy})
    codec.parser(file="directory")(lambda ctx: {"energy": 1})
    codec.parser(file="directory")(lambda ctx: {"energy": 2})
    with pytest.raises(DataCollectorError):
        codec.parse("energy", directory=Path("."))


def test_options_are_declared_validated_and_reach_parsers() -> None:
    """Options default from the codec, override per call, and reject unknown keys."""
    codec = Codec("local", {CalcType.energy}, options={"collect_extra": False})

    @codec.parser(file="directory")
    def extra(ctx: Context) -> dict[str, int]:
        """Fill a field only when asked to."""
        return {"calcinfo_natoms": 3} if ctx.options["collect_extra"] else {}

    assert (
        single_point(codec.parse("energy", directory=Path("."))).calcinfo_natoms is None
    )
    data = codec.parse("energy", directory=Path("."), options={"collect_extra": True})
    assert single_point(data).calcinfo_natoms == 3
    with pytest.raises(EncoderError, match="does not accept options"):
        codec.parse("energy", directory=Path("."), options={"typo": 1})
    with pytest.raises(RegistryError, match="declares options"):
        Codec("local", {CalcType.energy}, options={"x": 1}, decode=no_decode)


def test_encoder_return_type_is_checked(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """A script encoder returning the wrong thing fails before anything runs."""
    bad = lambda inp: {"command": "x"}  # noqa: E731
    codec = Codec("local", {CalcType.energy}, encode=bad)  # type: ignore[arg-type]
    with pytest.raises(
        EncoderError, match="must return SubprocessInput or PythonInput"
    ):
        codec.encode(prog_input_factory("energy", "local"))


def test_propagate_wfn_is_an_optional_attribute() -> None:
    """Codecs without wavefunction support simply leave the attribute unset."""
    assert Codec("local", {CalcType.energy}).propagate_wfn is None
    assert get_codec("terachem").propagate_wfn is not None


def test_register_and_get_codec(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Registered codecs resolve by name and override built-ins until unregistered."""
    codec = Codec(
        "myprog", {CalcType.energy}, encode=lambda inp: SubprocessInput(("myprog",))
    )
    with pytest.raises(CodecNotFoundError):
        get_codec("myprog")
    assert qccodec.register(codec) is codec
    try:
        assert get_codec("myprog") is codec
        assert get_codec(prog_input_factory("energy", "myprog")) is codec
        assert "myprog" in qccodec.supported_programs()
        native = qccodec.encode(prog_input_factory("energy", "myprog"))
        assert isinstance(native, SubprocessInput) and native.command == ("myprog",)
        override = qccodec.register(Codec("terachem", {CalcType.energy}))
        assert get_codec("terachem") is override
    finally:
        qccodec.unregister("myprog")
        qccodec.unregister("terachem")
    assert get_codec("terachem").parsers
    with pytest.raises(CodecNotFoundError):
        get_codec("myprog")


def test_file_input_resolves_to_file_codec() -> None:
    """FileInput uses the pass-through codec regardless of its program name."""
    inp = FileInput(program="/bin/echo", cmdline_args=["hi"], files={"a": "b"})
    codec = get_codec(inp)
    assert codec.program == "file"
    native = codec.encode(inp)
    assert isinstance(native, SubprocessInput)
    assert native.command == ("/bin/echo", "hi") and native.files == {"a": "b"}
    data = codec.decode(inp, SubprocessOutput(returncode=0))
    assert data.provenance.program == "/bin/echo"


def test_entry_point_codecs_are_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codecs published under the qccodec.codecs entry point group resolve by name."""
    from types import SimpleNamespace

    plugin = Codec("plugprog", {CalcType.energy})

    def fake_entry_points(*, group: str) -> list[SimpleNamespace]:
        """Stand in for installed distributions."""
        assert group == "qccodec.codecs"
        return [
            SimpleNamespace(name="plugprog", value="plug:codec", load=lambda: plugin),
            SimpleNamespace(name="broken", value="plug:nope", load=lambda: object()),
        ]

    monkeypatch.setattr("qccodec.registry.entry_points", fake_entry_points)
    assert get_codec("plugprog") is plugin
    assert "plugprog" in qccodec.supported_programs()
    with pytest.raises(CodecNotFoundError, match="not a qccodec.Codec"):
        get_codec("broken")
    override = qccodec.register(Codec("plugprog", {CalcType.gradient}))
    try:
        assert get_codec("plugprog") is override  # registered beats installed
    finally:
        qccodec.unregister("plugprog")


def versioned_codec() -> Codec:
    """A codec whose energy line changed format at version 2.0."""
    codec = Codec("vprog", {CalcType.energy})
    codec.parser(target=("provenance", "program_version"), required=False)(
        lambda stdout: qccodec.re_search(r"version (\S+)", stdout).group(1)
    )

    @codec.parser(target="energy", versions="<2.0")
    def energy_v1(stdout: str) -> float:
        """Old format: 'E = <value>'."""
        return float(qccodec.re_search(r"E = (\S+)", stdout).group(1))

    @codec.parser(target="energy", versions=">=2.0")
    def energy_v2(stdout: str) -> float:
        """New format: 'Total energy: <value>'."""
        return float(qccodec.re_search(r"Total energy: (\S+)", stdout).group(1))

    codec.parser(target="calcinfo_natoms")(lambda stdout: 3)
    return codec


def test_versioned_parsers_select_by_parsed_version() -> None:
    """The version parser runs first and gates which alternative applies."""
    codec = versioned_codec()
    old = single_point(codec.parse("energy", stdout="E = -1.0\nversion 1.5\n"))
    assert old.energy == -1.0 and old.provenance.program_version == "1.5"
    new = single_point(
        codec.parse("energy", stdout="Total energy: -2.0\nversion 2.1\n")
    )
    assert new.energy == -2.0
    # The new-format line under an old version is not read: the old parser misses.
    with pytest.raises(MatchNotFoundError):
        codec.parse("energy", stdout="Total energy: -2.0\nversion 1.5\n")


def test_versioned_parsers_with_unknown_or_given_version() -> None:
    """Unknown versions try alternatives in order; an explicit version wins."""
    codec = versioned_codec()
    assert (
        single_point(codec.parse("energy", stdout="Total energy: -2.0\n")).energy
        == -2.0
    )
    assert single_point(codec.parse("energy", stdout="E = -1.0\n")).energy == -1.0
    with pytest.raises(MatchNotFoundError):
        codec.parse("energy", stdout="nothing\n")
    given = single_point(codec.parse("energy", stdout="E = -1.0\n", version="1.0"))
    assert given.energy == -1.0 and given.provenance.program_version == "1.0"
    with pytest.raises(MatchNotFoundError):
        codec.parse("energy", stdout="E = -1.0\n", version="3.0")
    # Nested decodes inherit the version from their parent context.
    ctx = Context(codec, CalcType.energy, version="1.0")
    assert ctx.decode("energy", stdout="E = -4.0\n").provenance.program_version == "1.0"


def test_versions_registration_rules() -> None:
    """Specifiers are validated; same-target parsers must both declare versions."""
    codec = Codec("vprog", {CalcType.energy})
    with pytest.raises(RegistryError, match="Invalid versions"):
        codec.parser(target="energy", versions="not a spec")(energy)
    with pytest.raises(RegistryError, match="cannot declare versions"):
        codec.parser(target=("provenance", "program_version"), versions=">1")(energy)
    codec.parser(target="energy")(energy)
    with pytest.raises(RegistryError, match="Declare versions on both"):
        codec.parser(target="energy", versions=">=2")(energy)
    assert codec.parsers[0].versions is None


def test_version_key_and_check_codec() -> None:
    """version_key adapts non-PEP 440 strings; check_codec flags ungated detection."""
    from qccodec.programs.terachem import version_key
    from qccodec.testing import check_codec

    assert version_key("v1.9-2022.03-dev [0fe15afa]") == "2022.3"
    assert version_key("v1.5K [ccdev]") == "1.5"
    codec = Codec("dated", {CalcType.energy}, version_key=version_key)
    codec.parser(target=("provenance", "program_version"))(
        lambda stdout: stdout.split("|")[0]
    )
    codec.parser(target="energy", versions=">=2023.8")(lambda s: -2.0)
    codec.parser(target="energy", versions="<2023.8")(lambda s: -1.0)
    newer = single_point(codec.parse("energy", stdout="v1.9-2024.09-dev [abc]|"))
    older = single_point(codec.parse("energy", stdout="v1.5K [ccdev]|"))
    assert newer.energy == -2.0 and older.energy == -1.0

    blind = Codec("blind", {CalcType.energy})
    blind.parser(target="energy", versions=">=2")(energy)
    with pytest.raises(AssertionError, match="no parser fills"):
        check_codec(blind)
