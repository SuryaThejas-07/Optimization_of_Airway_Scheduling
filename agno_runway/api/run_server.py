from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("agno_runway.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
