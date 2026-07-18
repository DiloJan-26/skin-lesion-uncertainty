"""Path helpers for repository-local experiment outputs."""

from pathlib import Path
from typing import Any


def get_repo_root() -> Path:
    """Return the repository root inferred from this source file."""
    return Path(__file__).resolve().parents[1]


def ensure_output_dirs(
    config: dict[str, Any], root: Path | None = None
) -> dict[str, Path]:
    """Create configured output directories and return their resolved paths."""
    repo_root = root if root is not None else get_repo_root()
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("config.outputs must be a mapping of output directory names.")

    created_dirs: dict[str, Path] = {}
    for key, raw_path in outputs.items():
        if not isinstance(raw_path, str):
            raise ValueError(f"outputs.{key} must be a path string.")

        output_path = Path(raw_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

        output_path.mkdir(parents=True, exist_ok=True)
        created_dirs[key] = output_path.resolve()

    return created_dirs
