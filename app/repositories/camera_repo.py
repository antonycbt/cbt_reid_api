# camera_repo.py
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func
from app.db.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraUpdate

class CameraRepository:

    @staticmethod
    def create(db: Session, payload: CameraCreate) -> Camera:
        camera = Camera(
            name=payload.name,
            ip_address=payload.ip_address,
            site_location_id=payload.site_location_id,
            location_type=payload.location_type,
        )
        db.add(camera)
        db.commit()
        db.refresh(camera)  # eager-load after commit
        return camera

    @staticmethod
    def get_by_id(db: Session, camera_id: int) -> Camera | None:
        return (
            db.query(Camera)
            .options(joinedload(Camera.site_location))  # eager load
            .filter(Camera.id == camera_id)
            .first()
        )

    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10
    ) -> tuple[list[Camera], int]:

        stmt = db.query(Camera).options(joinedload(Camera.site_location))

        if search:
            stmt = stmt.filter(Camera.name.ilike(f"%{search}%"))

        total = stmt.count()

        cameras = stmt.order_by(
            Camera.is_active.desc(),
            Camera.name.asc()
        ).offset(page * page_size).limit(page_size).all()

        return cameras, total

    @staticmethod
    def update(db: Session, camera: Camera, payload: CameraUpdate) -> Camera:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(camera, field, value)
        db.commit()
        db.refresh(camera)  # reload relationships for response
        return camera

    @staticmethod
    def delete(db: Session, camera: Camera) -> None:
        db.delete(camera)
        db.commit()
