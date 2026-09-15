from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(namespace: str, *parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return f"{namespace}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
