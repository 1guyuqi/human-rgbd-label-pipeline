"""Repository root paths and optional YAML config loading."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HOI4D_DIR = ROOT / "hoi4d"
RVIDEO_DIR = ROOT / "rvideo"
CONFIG_DIR = ROOT / "config"
DEFAULT_PATHS_FILE = CONFIG_DIR / "paths.yaml"


def load_paths(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load tool checkpoint paths from YAML or JSON example config."""
    path = Path(config_path) if config_path else DEFAULT_PATHS_FILE
    if not path.is_file():
        path = CONFIG_DIR / "paths.example.yaml"
    if not path.is_file():
        return {}

    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required to load paths.yaml") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return data if isinstance(data, dict) else {}


def get_path(key: str, default: str = "", config: dict[str, Any] | None = None) -> str:
    """Resolve a path from config, then environment variable TOOL_<KEY>."""
    cfg = config or load_paths()
    env_key = f"TOOL_{key.upper()}"
    value = os.environ.get(env_key) or cfg.get(key) or default
    return str(value)
