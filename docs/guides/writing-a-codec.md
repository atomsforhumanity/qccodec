# Writing a codec

Everything qccodec knows about a program lives in one `Codec` object: its name,
the calculation types it supports, an encoder, and either a set of small parsers
(file-based programs) or a decode function (Python APIs). Built-ins live in
`qccodec/programs/<name>.py`, one file per program, and you can build your own in a
script the same way:

```python
from qcdata import CalcType, ProgramInput
from qccodec import Codec, SubprocessInput, re_search

codec = Codec("myprog", calctypes={CalcType.energy})

@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    return SubprocessInput(
        command=("myprog", "run.inp"),
        files={"run.inp": f"{inp.model.method}\n{inp.structure.to_xyz()}"},
    )

@codec.parser(target="energy", calctypes=[CalcType.energy])
def parse_energy(stdout: str) -> float:
    return float(re_search(r"Total energy:\s+(-?\d+\.\d+)", stdout).group(1))
```

If the program uses several CPU threads or a GPU, say so in the native input:
`SubprocessInput(..., threads=4)` or `SubprocessInput(..., gpus=1)`. qccompute runs
lists of calculations side by side and uses these to keep within the machine's cores
and give each GPU calculation its own device.

Then run it with qccompute, either explicitly or by name after registering it:

```python
import qccodec, qccompute

output = qccompute.compute(input_data, codec=codec)

qccodec.register(codec)              # now resolvable as program="myprog",
output = qccompute.compute(input_data)   # including as a geomeTRIC gradient child
```

`register()` also lets you override a built-in codec by name for the current process.

## Parsers

A parser is a plain function from file contents to a value. The decorator records
where the value goes and when the parser runs, and returns the function unchanged so
it stays easy to test directly.

```python
@codec.parser(target="energy", calctypes=[CalcType.energy, CalcType.gradient])
def parse_energy(stdout: str) -> float: ...                      # reads stdout (default)

@codec.parser(file="crest.engrad", target="gradient", calctypes=[CalcType.gradient])
def parse_gradient(contents: str) -> list[list[float]]: ...      # reads a named file or glob

@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]: ...   # needs several files
```

- `target` is the result field to fill, or a tuple for nested fields such as
  `("provenance", "program_version")`. Directory parsers may omit it and return a
  mapping that is merged into the result.
- `calctypes` restricts when the parser runs; omit it to run for every calculation type.
- `file` is `"stdout"` (default), `"directory"`, a file name or glob relative to the
  calculation directory, or a function of the `Context` returning such a name.
- `required=False` tolerates a missing file or `MatchNotFoundError`.
- `versions=">=6.0"` limits a parser to some program versions (a PEP 440 specifier);
  omit it to run on every version. See [Program versions](#program-versions) below.

Directory parsers receive a `Context` with `stdout`, `directory`, `input_data`,
`failed`, `options`, `path(pattern)`, `read(pattern)`, and `decode(calctype, stdout=...)`
for nested decoding with the same codec.

## Failure detection and Python-native programs

```python
@codec.failure
def failure(ctx: Context) -> str | None:
    return "CREST reported FAILED." if "FAILED" in (ctx.stdout or "") else None
```

Many programs exit 0 after failing (TeraChem prints `Job finished:` only on
success). When the hook returns a message, decoding recovers what it can and raises
`CalculationError` with that data attached. This also happens when the program
exits with an error, so the hook's explanation reaches the user rather than just an
exit status. `parse(..., failed=True)` stays quiet and returns the recovered data;
pass `report_failure=True` to raise the explanation there too.
Programs driven through a Python API skip parsers entirely:

```python
codec = Codec("mlp", calctypes={CalcType.energy, CalcType.gradient}, encode=encode, decode=decode)
```

where `decode(input_data, native_output)` maps `native_output.value` into a qcdata
object.

## Options and wavefunction propagation

A codec declares the decode options it accepts, with defaults. Callers override them
per calculation and parsers read them from the context:

```python
codec = Codec("crest", calctypes={...}, options={"collect_rotamers": False})

@codec.parser(file="directory", calctypes=[CalcType.conformer_search])
def parse_rotamers(ctx: Context) -> dict:
    if not ctx.options["collect_rotamers"]:
        return {}
    ...

decode("crest", "conformer_search", directory=d, input_data=inp, options={"collect_rotamers": True})
qccompute.compute(inp, options={"collect_rotamers": True})
```

Unknown option names raise `EncoderError`. For optimizers that chain calculations, a
codec may set `codec.propagate_wfn = fn` where `fn(output, input_data)` returns a new
input carrying the previous output's wavefunction as the initial guess; TeraChem copies
`c0`/`ca0`/`cb0` and sets `guess`. qccompute calls it when `propagate_wfn=True`.

## Program versions

When a program changes its output format, register a parser per version range on the
same target; the parser functions themselves stay version-free:

```python
@codec.parser(target="gradient", calctypes=[CalcType.gradient], versions="<6.0")
def parse_gradient_v5(stdout: str) -> list[list[float]]: ...

@codec.parser(target="gradient", calctypes=[CalcType.gradient], versions=">=6.0")
def parse_gradient(stdout: str) -> list[list[float]]: ...
```

The parser targeting `("provenance", "program_version")` runs first and its value
selects which parsers apply. Programs whose version strings are not PEP 440 give the
codec a `version_key` (TeraChem compares builds by release date, so `v1.9-2023.08-dev`
matches `versions=">=2023.8"`). When no version can be read, alternatives for a target
are tried in registration order and the first match wins. Pass `version=` to
`decode`/`parse` for logs that never print one.

## Codecs from other packages

A package can ship a codec by publishing it as an entry point; `get_codec` and
`qccompute.compute` then find it by program name with no registration call:

```toml
[project.entry-points."qccodec.codecs"]
myprog = "mypackage.codec:codec"
```

## Testing a codec

`qccodec.testing` gives every codec the same test harness. A recorded program run is a
directory holding the inputs, the outputs, and the data the codec must produce:

```
tests/data/myprog/water_energy/
    input.json      ProgramInput used for the run
    expected.json   the qcdata *Data object the codec must produce
    stdout          captured program output
    <files>         output files at their real relative paths, e.g. scr.geom/optim.xyz
    case.json       optional: {"options": {...}, "failed": true, "raises": "ParserError"}
    encoded/        optional: command.txt plus the native input files encode() must write
```

```python
import pytest
from qccodec import get_codec
from qccodec.testing import cases, check_codec, run_case

@pytest.mark.parametrize("case", cases("tests/data"), ids=str)
def test_case(case):
    run_case(get_codec(case.input_data.program), case)

def test_contract():
    check_codec(codec, inputs=[some_program_input])
```

`run_case` checks the encoded files, the full decode (or the expected exception and its
partial data), and every registered parser applied directly to its own file against the
matching slice of `expected.json`. `check_codec` verifies the structural contract: parser
targets name real result fields, decoding nothing yields typed empty data in recovery
mode and typed partial data on error, and encoded files stay inside the directory.

[CONTRIBUTING.md](https://github.com/atomsforhumanity/qccodec/blob/master/CONTRIBUTING.md) walks through adding a program to qccodec step by step, including the example scripts every program needs.
