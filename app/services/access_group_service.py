from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.db.session import SessionLocal
from app.repositories.access_group_repo import AccessGroupRepository
from app.schemas.access_group import (
    AccessGroupUpdate,
) 
from sqlalchemy.orm import Session, selectinload
from app.db.models import AccessGroup,Member 
from app.db.models.associations import member_access, site_location_access


class AccessGroupService:

    # CREATE access group
    @staticmethod
    def create_access_group(payload):
        db = SessionLocal()
        try:
            if AccessGroupRepository.exists_with_name(
                db,
                payload.name,
                payload.parent_access_group_id,
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Access group name already exists under this parent"
                )

            return AccessGroupRepository.create(db, payload)

        finally:
            db.close()

    # UPDATE access group
    @staticmethod
    def update_access_group(
        access_group_id: int,
        payload: AccessGroupUpdate,
    ):
        db = SessionLocal()
        try:
            group = AccessGroupRepository.get_by_id(
                db, access_group_id
            )
            if not group:
                raise HTTPException(
                    status_code=404,
                    detail="Access group not found"
                )

            return AccessGroupRepository.update(
                db, group, payload
            )

        finally:
            db.close()

    # GET access group by ID
    @staticmethod
    def get_access_group(
        access_group_id: int,
    ) -> AccessGroup | None:
        db = SessionLocal()
        try:
            return AccessGroupRepository.get_by_id(
                db, access_group_id
            )
        finally:
            db.close()

    # LIST access groups (search + pagination)
    @staticmethod
    def list_access_groups(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return AccessGroupRepository.list(
                db, search, page, page_size
            )
        finally:
            db.close()

    # HARD DELETE access group
    @staticmethod
    def delete_access_group(
        access_group_id: int,
    ) -> bool:
        db = SessionLocal()
        try:
            group = AccessGroupRepository.get_by_id(
                db, access_group_id
            )
            if not group:
                return False

            AccessGroupRepository.delete(db, group)
            return True

        finally:
            db.close()

    @staticmethod
    def list_unlinked_members_by_access_groups(
    db: Session,
    access_group_id: int | None = None,
    ):
        query = db.query(Member).order_by(Member.id)

        if access_group_id:
            linked_ids = (
                db.query(member_access.c.member_id)
                .filter(member_access.c.access_group_id == access_group_id)
                .subquery()
            )

            query = query.filter(
                ~Member.id.in_(linked_ids),
                Member.is_active.is_(True),
            )

        return query.all() 
