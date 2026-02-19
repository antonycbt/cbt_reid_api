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
    

    @staticmethod
    def bulk_import_cameras_from_rows(rows: list):
        from app.db.session import SessionLocal
        from app.db.models.camera import Camera

        db = SessionLocal()
        try:
            # ── Build site_location name → id map (only active) ──────────
            site_location_name_to_id = CameraRepository.get_site_location_name_map(db)

            # ── Get existing IPs ──────────────────────────────────────────
            raw_ips = [
                str(row[1]).strip() for row in rows
                if row and row[1]
            ]
            existing_ips = CameraRepository.get_existing_ip_addresses(db, raw_ips)

            # ── Process rows ──────────────────────────────────────────────
            cameras_to_create = []
            skipped = []

            for i, row in enumerate(rows, start=2):
                if not row or not row[0]:
                    continue

                name           = str(row[0]).strip() if row[0] else None
                ip_address     = str(row[1]).strip() if row[1] else None
                site_loc_raw   = str(row[2]).strip() if row[2] else None

                # ── Required fields ───────────────────────────────────────
                if not name or not ip_address or not site_loc_raw:
                    skipped.append({
                        "row": i,
                        "name": name or "N/A",
                        "reason": "Name, IP address, or Site Location is missing",
                    })
                    continue

                # ── Duplicate IP check (DB) ───────────────────────────────
                if ip_address in existing_ips:
                    skipped.append({
                        "row": i,
                        "name": name,
                        "reason": f"IP address '{ip_address}' already exists",
                    })
                    continue

                # ── Site location resolution ──────────────────────────────
                site_location_id = site_location_name_to_id.get(site_loc_raw.lower())
                if site_location_id is None:
                    skipped.append({
                        "row": i,
                        "name": name,
                        "reason": f"Site location '{site_loc_raw}' does not exist or is not active",
                    })
                    continue

                cameras_to_create.append(Camera(
                    name=name,
                    ip_address=ip_address,
                    site_location_id=site_location_id,
                    is_active=True,
                ))

                # ── Track within-batch duplicate IPs ──────────────────────
                existing_ips.add(ip_address)

            # ── Bulk insert ───────────────────────────────────────────────
            added_count = 0
            if cameras_to_create:
                db.add_all(cameras_to_create)
                db.commit()
                added_count = len(cameras_to_create)

            # ── Build response ────────────────────────────────────────────
            total_skipped = len(skipped)
            message = (
                f"{added_count} {'entry' if added_count == 1 else 'entries'} added successfully. "
                f"{total_skipped} {'entry' if total_skipped == 1 else 'entries'} skipped."
                if total_skipped
                else f"{added_count} {'entry' if added_count == 1 else 'entries'} added successfully."
            )

            return {
                "message": message,
                "added_count": added_count,
                "skipped_count": total_skipped,
                "skipped": skipped,
            }

        finally:
            db.close()
