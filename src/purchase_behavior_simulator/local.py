from __future__ import annotations

import json
import sys
from pathlib import Path

from .bootstrap import build_service
from .runtime_api import handle_request


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m purchase_behavior_simulator.local REQUEST.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = handle_request(payload, service=build_service())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
