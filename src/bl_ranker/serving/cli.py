"""Console entry point for the serving process."""
from __future__ import annotations

import uvicorn

from bl_ranker.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "bl_ranker.serving.app:app",
        host=settings.serving_host,
        port=settings.serving_port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
