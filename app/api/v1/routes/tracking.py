from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse

from app.services.tracking.mjpeg import mjpeg_generator
router = APIRouter()

@router.get("/status")
def status(request: Request):
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")
    return svc.status()


@router.get("/cameras")
def cameras(request: Request):
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")
    return svc.list_cameras()


# ✅ MATCHES YOUR FRONTEND:
# /v1/tracking/mjpeg?cam_ids=3
@router.get("/mjpeg")
def mjpeg(
    request: Request,
    cam_ids: int = Query(..., description="Camera ID"),
    max_fps: int = 15,
    jpeg_quality: int = 80,
):
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")

    buf = svc.get_camera_buffer(int(cam_ids))
    if buf is None:
        raise HTTPException(status_code=404, detail=f"Camera {cam_ids} not available")

    return StreamingResponse(
        mjpeg_generator(buf, max_fps=max_fps, jpeg_quality=jpeg_quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
