# camera_service.py
from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.repositories.camera_repo import CameraRepository
from app.schemas.camera import CameraCreate, CameraUpdate
from app.db.models.camera import Camera

class CameraService:

    @staticmethod
    def list_active_cameras(db: Session):
        return db.query(Camera).filter(Camera.is_active == True).all()

    @staticmethod
    def create_camera(db: Session, payload: CameraCreate) -> Camera:
        if payload.site_location_id is None:
            raise HTTPException(
                status_code=400,
                detail="site_location_id is required"
            )
        try:
            camera = CameraRepository.create(db, payload)
            return camera
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Camera with this name or IP already exists"
            )

    @staticmethod
    def get_camera(db: Session, camera_id: int) -> Camera:
        camera = CameraRepository.get_by_id(db, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return camera

    @staticmethod
    def list_cameras(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10
    ):
        return CameraRepository.list(db, search, page, page_size)

    @staticmethod
    def update_camera(db: Session, camera_id: int, payload: CameraUpdate) -> Camera:
        if payload.site_location_id is None:
            raise HTTPException(
                status_code=400,
                detail="site_location_id is required"
            )
        camera = CameraRepository.get_by_id(db, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        try:
            updated = CameraRepository.update(db, camera, payload)
            return updated
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Camera with this name or IP already exists"
            )

    @staticmethod
    def delete_camera(db: Session, camera_id: int) -> bool:
        camera = CameraRepository.get_by_id(db, camera_id)
        if not camera:
            return False
        CameraRepository.delete(db, camera)
        return True
