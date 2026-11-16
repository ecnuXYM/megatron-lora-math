#!/usr/bin/env python3
"""Read archived summary JSON; this does not execute a model."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "evidence/metrics").glob("*.json")):
    print(path.name)
    print(json.dumps(json.loads(path.read_text()), ensure_ascii=False, indent=2))
