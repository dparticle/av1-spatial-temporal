from __future__ import annotations

import importlib
import re
from pathlib import Path


def _latest_app_module() -> str:
    candidates: list[tuple[int, str]] = []
    for path in Path(__file__).parent.glob("app_v*.py"):
        match = re.fullmatch(r"app_v(\d+)", path.stem)
        if match:
            candidates.append((int(match.group(1)), path.stem))
    if not candidates:
        raise RuntimeError("No CLI implementation module is installed")
    return max(candidates)[1]


def main(argv: list[str] | None = None) -> int:
    implementation = importlib.import_module(f".{_latest_app_module()}", __package__)
    return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

