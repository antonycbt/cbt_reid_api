from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func
from typing import List

from app.db.models.access_group import AccessGroup
from app.schemas.access_group import (
    AccessGroupCreate,
    AccessGroupUpdate,
)


class AccessGroupRepository:

    # CREATE
    @staticmethod
    def create(
        db: Session,
        payload: AccessGroupCreate,
    ) -> AccessGroup:
        group = AccessGroup(
            name=payload.name,
            parent_access_group_id=payload.parent_access_group_id,
        )
        db.add(group)
        db.commit()

        # 🔁 CRITICAL: re-fetch with parent eagerly loaded
        return (
            db.query(AccessGroup)
            .options(joinedload(AccessGroup.parent))
            .filter(AccessGroup.id == group.id)
            .one()
        )

    # GET BY ID
    @staticmethod
    def get_by_id(
        db: Session,
        group_id: int,
    ) -> AccessGroup | None:
        return db.get(AccessGroup, group_id)

    # DUPLICATION CHECK
    @staticmethod
    def exists_with_name(
        db: Session,
        name: str,
        parent_id: int | None,
        exclude_id: int | None = None,
    ) -> bool:
        stmt = select(AccessGroup).where(
            func.lower(AccessGroup.name) == name.lower(),
            AccessGroup.parent_access_group_id == parent_id,
        )

        if exclude_id:
            stmt = stmt.where(AccessGroup.id != exclude_id)

        return db.execute(stmt).scalars().first() is not None

    # UPDATE
    @staticmethod
    def update(
        db: Session,
        group: AccessGroup,
        payload: AccessGroupUpdate,
    ) -> AccessGroup:

        for field, value in payload.model_dump(
            exclude_unset=True
        ).items():
            setattr(group, field, value)

        db.commit()

        # 🔁 CRITICAL: re-fetch with parent eagerly loaded
        return (
            db.query(AccessGroup)
            .options(joinedload(AccessGroup.parent))
            .filter(AccessGroup.id == group.id)
            .one()
        )

    # LIST (search + pagination)
    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list[AccessGroup], int]:

        stmt = (
            select(AccessGroup)
            .options(joinedload(AccessGroup.parent))
        )

        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.where(
                func.lower(AccessGroup.name).like(search_term)
            )

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        stmt = stmt.order_by(
            AccessGroup.is_active.desc(),
            func.lower(AccessGroup.name).asc()
        )

        stmt = stmt.offset(page * page_size).limit(page_size)

        groups = db.execute(stmt).scalars().all()
        return groups, total

    # LIST ALL ACTIVE (for dropdowns)
    @staticmethod
    def list_all(db: Session):
        return (
            db.query(AccessGroup)
            .options(joinedload(AccessGroup.parent))
            .all()
        )

    # DELETE (HARD)
    @staticmethod
    def delete(
        db: Session,
        group: AccessGroup,
    ) -> None:
        db.delete(group)
        db.commit()

    @staticmethod
    def list_all(db):
        return (
            db.query(AccessGroup)
            .filter(AccessGroup.is_active.is_(True))
            .order_by(AccessGroup.name)
            .all()
        )