from __future__ import annotations

import os
import shlex
from typing import List


def build_pipeline_argv() -> List[str]:
    """
    Returns argv list for parse_args().
    Reads env vars:
      - PIPELINE_ARGS
      - pipeline_args
    """
    s = os.environ.get("PIPELINE_ARGS") or os.environ.get("pipeline_args") or ""
    s = str(s).strip()
    return shlex.split(s) if s else []
