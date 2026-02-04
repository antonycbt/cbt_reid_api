from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session, joinedload

from app.schemas.camera import (
    CameraCreate,
    CameraUpdate,
    CameraOut,
)
from app.schemas.common import MessageResponse
from app.services.camera_service import CameraService
from app.db.session import get_db
from app.db.models.camera import Camera
from app.db.models.site_location import SiteLocation
from app.core.constants import CAM_LOCATION_TYPES

router = APIRouter()

# FETCH all active cameras
@router.get("/allcameras", response_model=MessageResponse[list[dict]])
def list_active_cameras(db: Session = Depends(get_db)):
    cameras = CameraService.list_active_cameras(db)
    data = [{"id": cam.id, "name": cam.name} for cam in cameras]
    return {"message": "Active cameras fetched successfully", "data": data}


# FETCH site locations
@router.get("/load_site_locations", response_model=MessageResponse[list[dict]])
def list_site_locations(db: Session = Depends(get_db)):
    locations = db.query(SiteLocation).all()
    data = [{"id": loc.id, "name": loc.name} for loc in locations]
    return {"message": "Site locations fetched successfully", "data": data}


# CREATE camera
@router.post("", response_model=MessageResponse[CameraOut])
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    camera = CameraService.create_camera(db, payload)

    # 🔥 reload with relationship eager loaded
    camera = (
        db.query(Camera)
        .options(joinedload(Camera.site_location))
        .filter(Camera.id == camera.id)
        .first()
    )

    return {"message": "Camera created successfully", "data": camera}


# location types
@router.get("/location-types", response_model=MessageResponse[dict])
def list_location_types():
    return {
        "message": "Location types fetched successfully",
        "data": CAM_LOCATION_TYPES,
    }


# GET camera by ID
@router.get("/{camera_id}", response_model=MessageResponse[CameraOut])
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    camera = (
        db.query(Camera)
        .options(joinedload(Camera.site_location))
        .filter(Camera.id == camera_id)
        .first()
    )

    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return {"message": "Camera fetched successfully", "data": camera}


# LIST cameras
@router.get("", response_model=MessageResponse[list[CameraOut]])
def list_cameras(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    cameras, total = CameraService.list_cameras(db, search, page, page_size)

    return {
        "message": "Cameras fetched successfully",
        "data": cameras,
        "total": total,
    }


@router.put("/{camera_id}", response_model=MessageResponse[CameraOut])
def update_camera(camera_id: int, payload: CameraUpdate, db: Session = Depends(get_db)):
    """
    Update a camera. Requires site_location_id to be set.
    """
    camera = CameraService.update_camera(db, camera_id, payload)  # pass db as first argument
    return {"message": "Camera updated successfully", "data": camera}



# DELETE camera
@router.delete("/{camera_id}", response_model=MessageResponse[None])
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    success = CameraService.delete_camera(db, camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")

    return {"message": "Camera deleted successfully"}
