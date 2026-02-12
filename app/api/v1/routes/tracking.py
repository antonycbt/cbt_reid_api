from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.deps import get_detection_service
from app.services.tracking.mjpeg import mjpeg_generator, mjpeg_generator_multi
from app.services.tracking.service import DetectionService

router = APIRouter()


def _parse_cam_ids(cam_ids: Optional[str]) -> List[int]:
    if not cam_ids:
        return []
    out: List[int] = []
    for part in cam_ids.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    # unique preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


@router.get("/status")
def status(svc: DetectionService = Depends(get_detection_service)):
    return svc.status()


@router.get("/cameras")
def list_cameras(svc: DetectionService = Depends(get_detection_service)):
    return {"cameras": svc.list_cameras()}
# Multi-camera (grid MJPEG): /api/v1/mjpeg?cam_ids=0,1,2
@router.get("/mjpeg")
def mjpeg_stream_multi(
    cam_ids: str = Query(..., description="Comma-separated camera ids, e.g. 0,1,2"),
    fps: int = Query(10, ge=1, le=30),
    quality: int = Query(80, ge=30, le=95),
    width: int = Query(1280, ge=320, le=3840),
    height: int = Query(720, ge=240, le=2160),
    grid_mode: str = Query("cover", pattern="^(cover|contain)$"),
    grid_rows: int = Query(0, ge=0, le=10),
    grid_cols: int = Query(0, ge=0, le=10),
    svc: DetectionService = Depends(get_detection_service),
):
    ids = _parse_cam_ids(cam_ids)
    if not ids:
        raise HTTPException(status_code=400, detail="cam_ids is empty")

    if len(ids) > 16:
        raise HTTPException(status_code=400, detail="Too many cam_ids (max 16)")

    bufs = []
    for cid in ids:
        b = svc.get_camera_buffer(cid)
        if b is None:
            raise HTTPException(status_code=404, detail=f"Camera id {cid} not found")
        bufs.append(b)

    return StreamingResponse(
        mjpeg_generator_multi(
            bufs,
            max_fps=fps,
            jpeg_quality=quality,
            out_w=width,
            out_h=height,
            grid_mode=grid_mode,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

# Single camera (most efficient: wait-on-new-frame with cached JPEG)
@router.get("/mjpeg/{cam_id}")
def mjpeg_stream_single(
    cam_id: int,
    fps: int = Query(10, ge=1, le=30),
    quality: int = Query(80, ge=30, le=95),
    svc: DetectionService = Depends(get_detection_service),
):
    buf = svc.get_camera_buffer(cam_id)
    if buf is None:
        raise HTTPException(status_code=404, detail=f"Camera id {cam_id} not found")

    return StreamingResponse(
        mjpeg_generator(buf, max_fps=fps, jpeg_quality=quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )





@router.post("/report/write")
def write_report(
    path: str | None = None,
    svc: DetectionService = Depends(get_detection_service),
):
    out_path = svc.write_report_snapshot(path=path)
    return {"saved_to": out_path}
