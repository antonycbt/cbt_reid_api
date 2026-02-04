from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, conint
from typing import Optional, List, Dict, Any

# IMPORTANT: import your new service module here
# If you saved service above as app/services/embedding_service.py:
from app.services import embedding_service

router = APIRouter()


class StartRequest(BaseModel):
    member_id: conint(gt=0) = Field(..., description="Member/User ID to capture for")
    camera_ids: Optional[List[int]] = Field(None, description="Which cameras to use. If null -> all configured")
    show_viewer: bool = Field(True, description="Open OpenCV viewer window")
    clear_existing: bool = Field(False, description="Clear existing galleries before starting")


class StartResponse(BaseModel):
    status: str
    message: str = ""
    member_id: Optional[int] = None
    member_name: Optional[str] = None
    num_cams: int = 0
    camera_ids: List[int] = []
    configured_camera_ids: Optional[List[int]] = None


class StopResponse(BaseModel):
    status: str
    message: str = ""
    member_id: Optional[int] = None
    camera_ids: List[int] = []


class ExtractRequest(BaseModel):
    member_id: conint(gt=0)
    camera_ids: Optional[List[int]] = None
    sync: bool = False


@router.post("/start", response_model=StartResponse)
def start(req: StartRequest) -> StartResponse:
    try:
        res = embedding_service.start_extraction(
            member_id=req.member_id,
            camera_ids=req.camera_ids,
            show_viewer=req.show_viewer,
            clear_existing=req.clear_existing,
        )
    except TypeError:
        # If service doesn't have clear_existing param yet
        res = embedding_service.start_extraction(req.member_id, req.camera_ids, req.show_viewer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"start_extraction failed: {e}") from e

    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="start_extraction returned invalid response")

    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Unknown error"))

    cam_ids = list(res.get("camera_ids") or [])
    return StartResponse(
        status=str(res.get("status", "ok")),
        message=str(res.get("message", "")),
        member_id=res.get("member_id"),
        member_name=res.get("member_name"),
        num_cams=int(res.get("num_cams", len(cam_ids))),
        camera_ids=cam_ids,
        configured_camera_ids=res.get("configured_camera_ids"),
    )


@router.post("/stop", response_model=StopResponse)
def stop(reason: str = Query("user", description="Reason for stopping")) -> StopResponse:
    try:
        res = embedding_service.stop_extraction(reason=reason)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stop_extraction failed: {e}") from e

    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="stop_extraction returned invalid response")

    return StopResponse(
        status=str(res.get("status", "ok")),
        message=str(res.get("message", "")),
        member_id=res.get("member_id"),
        camera_ids=list(res.get("camera_ids") or []),
    )


@router.post("/extract", response_model=Dict[str, Any])
def extract(req: ExtractRequest) -> Dict[str, Any]:
    try:
        if req.sync:
            res = embedding_service.extract_embeddings_sync(req.member_id, req.camera_ids)
        else:
            res = embedding_service.extract_embeddings_async(req.member_id, req.camera_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"extract failed: {e}") from e

    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="extract returned invalid response")

    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Unknown error"))

    return res


@router.get("/progress/{member_id}", response_model=Dict[str, Any])
def progress(member_id: int, camera_id: Optional[int] = None) -> Dict[str, Any]:
    try:
        st = embedding_service.get_progress_for_member(member_id, camera_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_progress failed: {e}") from e
    return st if isinstance(st, dict) else {"status": "error", "message": "invalid progress"}


@router.delete("/remove/{member_id}", response_model=Dict[str, Any])
def remove(member_id: int, camera_id: Optional[int] = None) -> Dict[str, Any]:
    try:
        res = embedding_service.remove_embeddings(member_id, camera_id=camera_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"remove_embeddings failed: {e}") from e

    if isinstance(res, dict) and res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "unknown error"))
    return res if isinstance(res, dict) else {"status": "ok", "member_id": member_id, "camera_id": camera_id}


@router.get("/status", response_model=Dict[str, Any])
def status() -> Dict[str, Any]:
    try:
        res = embedding_service.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_status failed: {e}") from e

    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="get_status returned invalid response")
    return res
