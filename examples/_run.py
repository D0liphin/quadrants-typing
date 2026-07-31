"""Dispatch `pixi run example <name>` to the matching script in this folder."""

import runpy
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent


def available_names() -> list[str]:
    # Collect every example script name, skipping `_`-prefixed helper files.
    return sorted(p.stem for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_"))


def main() -> None:
    # Read the requested example name from the command line.
    if len(sys.argv) != 2:
        print("usage: pixi run example <name>")
        print("available examples:", ", ".join(available_names()))
        raise SystemExit(2)

    name = sys.argv[1]
    script = EXAMPLES_DIR / f"{name}.py"

    # Reject a name that does not resolve to a script and show the valid names.
    if not script.is_file():
        print(f"unknown example: {name}")
        print("available examples:", ", ".join(available_names()))
        raise SystemExit(1)

    # Run the resolved script as if it were invoked directly.
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
