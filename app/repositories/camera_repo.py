# camera_repo.py
from typing import List, Set, Dict, Optional, Tuple
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, or_
from app.db.models import Camera, SiteLocation
from app.schemas.camera import CameraCreate, CameraUpdate
from fastapi import HTTPException
class CameraRepository:

    @staticmethod
    def create(db: Session, payload: CameraCreate) -> Camera:
        existing = db.execute(
            select(Camera.id).where(Camera.ip_address == payload.ip_address)
        ).first() 
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Camera with this IP address already exists",
            )

        camera = Camera(
            name=payload.name,
            ip_address=payload.ip_address,
            site_location_id=payload.site_location_id,
        )

        db.add(camera)
        db.commit()
        db.refresh(camera)
        return camera

    @staticmethod
    def get_by_id(db: Session, camera_id: int) -> Camera | None:
        return (
            db.query(Camera)
            .options(
                joinedload(Camera.site_location_rel)
                .joinedload(SiteLocation.site_hierarchy)
            )
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

        stmt = (
            db.query(Camera)
            .options(
                joinedload(Camera.site_location_rel)
                .joinedload(SiteLocation.site_hierarchy)
            )
        )  
        if search:
            stmt = stmt.filter(
                or_(
                    Camera.name.ilike(f"%{search}%"),
                    Camera.ip_address.ilike(f"%{search}%"),
                )
            )

        total = stmt.count()

        cameras = (
            stmt.order_by(Camera.is_active.desc(), Camera.name.asc())
            .offset(page * page_size)
            .limit(page_size)
            .all()
        )

        return cameras, total

    @staticmethod
    def update(db: Session, camera: Camera, payload: CameraUpdate) -> Camera:
        data = payload.model_dump(exclude_unset=True)

        # check duplicate IP only if ip_address is being updated
        if "ip_address" in data:
            existing = db.execute(
                select(Camera.id).where(
                    Camera.ip_address == data["ip_address"],
                    Camera.id != camera.id,  # exclude current camera
                )
            ).first()

            if existing:
                raise HTTPException(
                    status_code=400,
                    detail="Camera with this IP address already exists",
                )

        # apply updates
        for field, value in data.items():
            setattr(camera, field, value)

        db.commit()
        db.refresh(camera)
        return camera 
    
    @staticmethod
    def delete(db: Session, camera: Camera) -> None:
        db.delete(camera)
        db.commit()

    @staticmethod
    def get_existing_ip_addresses(db: Session, ip_addresses: List[str]) -> Set[str]:
        rows = db.execute(
            select(Camera.ip_address).where(
                Camera.ip_address.in_(ip_addresses)
            )
        ).scalars().all()
        return {ip for ip in rows}

    @staticmethod
    def get_site_location_name_map(db: Session) -> Dict[str, int]:
        from app.db.models.site_location import SiteLocation
        from app.db.models.site_hierarchy import SiteHierarchy
        rows = db.execute(
            select(SiteLocation.id, SiteHierarchy.name)
            .join(SiteHierarchy, SiteHierarchy.id == SiteLocation.site_hierarchy_id)
            .where(
                SiteLocation.is_active.is_(True),
                SiteHierarchy.is_active.is_(True),   # 👈 add this
            )
        ).all()
        return {row.name.strip().lower(): row.id for row in rows}

    @staticmethod
    def bulk_create(db: Session, cameras: List) -> List:
        db.add_all(cameras)
        db.commit()
        for camera in cameras:
            db.refresh(camera)
        return cameras      
