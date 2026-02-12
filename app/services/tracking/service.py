# from __future__ import annotations

# import threading
# from typing import Any, Dict, List, Optional

# from app.services.tracking.pipeline import TrackingRunner, FrameBuffer


# class DetectionService:
#     def __init__(self, args):
#         self._args = args
#         self._lock = threading.Lock()
#         self._runner: Optional[TrackingRunner] = None

#     def start(self) -> None:
#         with self._lock:
#             if self._runner is not None:
#                 return
#             runner = TrackingRunner(self._args)
#             runner.start()
#             self._runner = runner

#     def stop(self) -> None:
#         with self._lock:
#             if self._runner is None:
#                 return
#             self._runner.stop()
#             self._runner = None

#     def status(self) -> Dict[str, Any]:
#         r = self._runner
#         if r is None:
#             return {"running": False}
#         return {
#             "running": True,
#             "num_cameras": r.num_cameras(),
#             "cameras": r.cameras_brief(),
#             "db_last_reload_ts": r.db_last_reload_ts(),
#         }

#     def list_cameras(self) -> List[Dict[str, Any]]:
#         r = self._runner
#         if r is None:
#             return []
#         return r.cameras_brief()

#     def get_camera_buffer(self, cam_id: int) -> Optional[FrameBuffer]:
#         r = self._runner
#         if r is None:
#             return None
#         return r.get_buffer(cam_id)

#     def write_report_snapshot(self, path: Optional[str] = None) -> str:
#         r = self._runner
#         if r is None:
#             raise RuntimeError("service not running")
#         return r.write_report_snapshot(path=path)

from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Optional, Union

from app.services.tracking.pipeline import TrackingRunner, parse_pipeline_args, RenderedFrame


class DetectionService:
    """
    FastAPI service wrapper.
    Accepts either:
      - pipeline_args as string (env-style)
      - pipeline_args as argparse.Namespace (already parsed)
    Cameras are started dynamically when MJPEG is requested.
    """

    def __init__(self, pipeline_args: Union[str, argparse.Namespace, None] = None):
        self._runner: TrackingRunner | None = None

        # If lifespan passes in a Namespace, keep it directly
        if isinstance(pipeline_args, argparse.Namespace):
            self._args: argparse.Namespace | None = pipeline_args
            self._pipeline_args_str: str = ""
        else:
            self._args = None
            s = pipeline_args
            if s is None:
                s = os.environ.get("PIPELINE_ARGS") or os.environ.get("pipeline_args") or ""
            self._pipeline_args_str = str(s).strip()

    def start(self) -> None:
        if self._runner is not None:
            return

        # Use already-parsed args if provided, else parse from string
        args = self._args if self._args is not None else parse_pipeline_args(self._pipeline_args_str)

        # DB url fallback
        if not getattr(args, "db_url", ""):
            args.db_url = os.environ.get("DATABASE_URL", "") or ""

        self._runner = TrackingRunner(args)
        self._runner.start()

    def stop(self) -> None:
        if self._runner is None:
            return
        try:
            self._runner.stop()
        finally:
            self._runner = None

    def get_camera_buffer(self, cam_id: int) -> Optional[RenderedFrame]:
        """
        Called by the router.
        Auto-starts camera from DB if not already running.
        """
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.get_camera_buffer(int(cam_id))

    def list_cameras(self) -> List[Dict[str, Any]]:
        """
        Returns DB cameras (active) with running status.
        """
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.list_db_cameras(active_only=True)

    def status(self) -> Dict[str, Any]:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.status()

    def write_report_snapshot(self, path: str | None = None) -> str:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.write_report_snapshot(path=path)
