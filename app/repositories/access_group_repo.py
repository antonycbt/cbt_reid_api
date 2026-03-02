from sqlalchemy.orm import Session, joinedload, aliased
from sqlalchemy import select, func
from typing import List

from app.db.models.access_group import AccessGroup
from app.schemas.access_group import AccessGroupCreate, AccessGroupUpdate


class AccessGroupRepository:

    @staticmethod
    def create(db: Session, payload: AccessGroupCreate) -> AccessGroup:
        group = AccessGroup(
            name=payload.name,
            parent_access_group_id=payload.parent_access_group_id,
        )
        db.add(group)
        db.flush()  # ← service commits
        return group  # plain object, service re-fetches after commit

    @staticmethod
    def get_by_id(db: Session, group_id: int) -> AccessGroup | None:
        return db.get(AccessGroup, group_id)

    @staticmethod
    def get_with_relations(db: Session, group_id: int) -> AccessGroup | None:
        return (
            db.query(AccessGroup)
            .options(joinedload(AccessGroup.parent))
            .filter(AccessGroup.id == group_id)
            .one_or_none()
        )

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

    @staticmethod
    def update(db: Session, group: AccessGroup, payload: AccessGroupUpdate) -> AccessGroup:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(group, field, value)
        db.flush()  # ← service commits
        return group  # plain object, service re-fetches after commit

    @staticmethod
    def list_active_hierarchy(db: Session) -> tuple[list[AccessGroup], int]:
        anchor = (
            select(AccessGroup.id)
            .where(AccessGroup.parent_access_group_id.is_(None))
            .where(AccessGroup.is_active == True)
            .cte(name="active_hierarchy", recursive=True)
        )
        child = aliased(AccessGroup)
        recursive_part = (
            select(child.id)
            .join(anchor, child.parent_access_group_id == anchor.c.id)
            .where(child.is_active == True)
        )
        active_hierarchy = anchor.union_all(recursive_part)
        stmt = (
            select(AccessGroup)
            .options(joinedload(AccessGroup.parent))
            .where(AccessGroup.id.in_(select(active_hierarchy.c.id)))
            .order_by(func.lower(AccessGroup.name).asc())
        )
        groups = db.execute(stmt).scalars().all()
        return groups, len(groups)

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
            stmt = stmt.where(func.lower(AccessGroup.name).like(search_term))

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

    @staticmethod
    def list_all(db: Session):
        return (
            db.query(AccessGroup)
            .filter(AccessGroup.is_active.is_(True))
            .order_by(AccessGroup.name)
            .all()
        )

    @staticmethod
    def delete(db: Session, group: AccessGroup) -> None:
        db.delete(group)
        db.flush()  # ← service commits