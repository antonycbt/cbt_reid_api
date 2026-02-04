from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.db.session import SessionLocal, get_db
from app.repositories.site_location_access_repo import SiteLocationAccessRepository
from app.schemas.site_location_access import SiteLocationAccessCreate, SiteLocationAccessBulkCreate

class SiteLocationAccessService:

    # -------- SINGLE CREATE --------
    @staticmethod
    def create_site_location_access(payload: SiteLocationAccessCreate):
        db: Session = SessionLocal()
        try:
            if SiteLocationAccessRepository.exists(
                db, payload.site_location_id, payload.access_group_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Access already exists",
                )

            SiteLocationAccessRepository.create(
                db, payload.site_location_id, payload.access_group_id
            )

            return {
                "site_location_id": payload.site_location_id,
                "access_group_id": payload.access_group_id,
            }
        finally:
            db.close()

    # -------- BULK ASSIGN --------
    @staticmethod
    def bulk_assign_access(payload: SiteLocationAccessBulkCreate):
        db: Session = SessionLocal()
        try:
            created_count = SiteLocationAccessRepository.bulk_create(
                db, payload.site_location_id, payload.access_group_ids
            )

            if created_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="All selected access groups are already assigned",
                )

            return {"site_location_id": payload.site_location_id, "created_count": created_count}
        finally:
            db.close()

    # -------- LIST --------
    @staticmethod
    def list_site_location_access(search: str | None = None, page: int = 0, page_size: int = 10):
        db: Session = SessionLocal()
        try:
            data, total = SiteLocationAccessRepository.list(db, search, page, page_size)
            return data, total
        finally:
            db.close()

    # -------- DELETE SINGLE ACCESS --------
    @staticmethod
    def delete_single_access(site_location_id: int, access_group_id: int):
        """
        Delete a single access group from a site location
        """
        db: Session = SessionLocal()
        try:
            deleted_count = SiteLocationAccessRepository.delete(db, site_location_id, access_group_id)
            if deleted_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Access group not assigned to this site location"
                )
        finally:
            db.close()

    # -------- DELETE ALL ACCESS GROUPS FOR SITE LOCATION --------
    @staticmethod
    def delete_all_for_site_location(site_location_id: int):
        db: Session = SessionLocal()
        try:
            SiteLocationAccessRepository.delete_all_for_site_location(db, site_location_id)
        finally:
            db.close()
