from __future__ import annotations

import json

from agno_runway.api.state import build_runtime_state


def main() -> None:
    runtime = build_runtime_state()
    result = runtime.simulation.run(duration_seconds=300.0)
    print(json.dumps(result["metrics"], indent=2))
    print(f"Decisions logged: {len(result['decisions'])}")


if __name__ == "__main__":
    main()
