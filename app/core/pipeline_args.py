from __future__ import annotations
import os
import shlex
from typing import List


def build_pipeline_argv() -> List[str]:
    """
    Reads pipeline CLI args from env and returns argv list.

    Example:
      export PIPELINE_ARGS='--src "rtsp://..." "rtsp://..." --use-db --db-url "postgresql+psycopg2://..." ...'
    """
    raw = os.getenv("PIPELINE_ARGS", "").strip()
    if not raw:
        raise RuntimeError(
            "PIPELINE_ARGS is not set.\n\n"
            "Set it to your pipeline args, e.g.\n"
            "PIPELINE_ARGS='--src \"rtsp://...\" \"rtsp://...\" --use-db --db-url \"postgresql+psycopg2://...\" ...'\n"
        )
    return shlex.split(raw)
