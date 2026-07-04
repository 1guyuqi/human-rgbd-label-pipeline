#!/usr/bin/env python3
"""Remove Chinese comments and local absolute paths from repo sources."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u4e00-\u9fff]")
PATH_PATTERNS = [
    (re.compile(r"/home/ljx[^\s\"']*"), "PLACEHOLDER"),
    (re.compile(r"/media/ljx[^\s\"']*"), "PLACEHOLDER"),
    (re.compile(r"/cache0/[^\s\"']*"), "PLACEHOLDER"),
]

REPLACEMENTS = {
    "PLACEHOLDER": "",  # removed; use CLI args / config/paths.yaml
}


def clean_line(line: str) -> str | None:
    if not HAN.search(line):
        return line
    stripped = line.strip()
    if stripped.startswith("#"):
        return None
    if "#" in line:
        head, _tail = line.split("#", 1)
        if HAN.search(_tail):
            head = head.rstrip()
            return head if head else None
    if '"""' in line or "'''" in line:
        return line
    return line


def clean_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pat, _ in PATH_PATTERNS:
        text = pat.sub("", text)
    lines = []
    for line in text.splitlines():
        cleaned = clean_line(line)
        if cleaned is not None:
            lines.append(cleaned)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for py in ROOT.rglob("*.py"):
        if py.name == "sanitize_repo.py":
            continue
        clean_file(py)


if __name__ == "__main__":
    main()
