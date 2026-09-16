import argparse
from pathlib import Path

from .codec import decode


def main():
    parser = argparse.ArgumentParser(
        description="Parse Quantum Chemistry output files into structured JSON or "
        "Python objects."
    )
    parser.add_argument("program", help="Name of the program")
    parser.add_argument(
        "calctype", help="Type of calculation. See qcdata.CalcType for options"
    )
    parser.add_argument(
        "stdout",
        help="Path to the stdout file (optional)",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "directory",
        help="Path to the directory containing the output files (optional)",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Recover partial data from a failed calculation",
    )
    args = parser.parse_args()

    stdout_contents = Path(args.stdout).read_text() if args.stdout else None
    results = decode(
        args.program,
        args.calctype,
        stdout=stdout_contents,
        directory=args.directory,
        failed=args.failed,
    )
    print(results.model_dump_json(indent=4, exclude_unset=True))


if __name__ == "__main__":
    main()
