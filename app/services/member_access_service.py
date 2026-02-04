from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.db.session import SessionLocal, get_db
from app.repositories.member_access_repo import MemberAccessRepository
from app.schemas.member_access import MemberAccessCreate, MemberAccessBulkCreate

class MemberAccessService:

    # -------- SINGLE CREATE --------
    @staticmethod
    def create_member_access(payload: MemberAccessCreate):
        db: Session = SessionLocal()
        try:
            if MemberAccessRepository.exists(
                db, payload.member_id, payload.access_group_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Access already exists",
                )

            MemberAccessRepository.create(
                db, payload.member_id, payload.access_group_id
            )

            return {
                "member_id": payload.member_id,
                "access_group_id": payload.access_group_id,
            }
        finally:
            db.close()

    # -------- BULK ASSIGN --------
    @staticmethod
    def bulk_assign_access(payload: MemberAccessBulkCreate):
        db: Session = SessionLocal()
        try:
            created_count = MemberAccessRepository.bulk_create(
                db, payload.member_id, payload.access_group_ids
            )

            if created_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="All selected access groups are already assigned",
                )

            return {"member_id": payload.member_id, "created_count": created_count}
        finally:
            db.close()

    # -------- LIST --------
    @staticmethod
    def list_member_access(search: str | None = None, page: int = 0, page_size: int = 10):
        db: Session = SessionLocal()
        try:
            data, total = MemberAccessRepository.list(db, search, page, page_size)
            return data, total
        finally:
            db.close()

    # -------- DELETE SINGLE ACCESS --------
    @staticmethod
    def delete_single_access(member_id: int, access_group_id: int):
        """
        Delete a single access group from a member access
        """
        db: Session = SessionLocal()
        try:
            deleted_count = MemberAccessRepository.delete(db, member_id, access_group_id)
            if deleted_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Access group not assigned to this member access"
                )
        finally:
            db.close()

    # -------- DELETE ALL ACCESS GROUPS FOR MEMBER ACCESS --------
    @staticmethod
    def delete_all_for_site_location(member_id: int):
        db: Session = SessionLocal()
        try:
            MemberAccessRepository.delete_all_for_site_location(db, member_id)
        finally:
            db.close()
