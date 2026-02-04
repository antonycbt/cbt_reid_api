from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import select, func
from typing import List, Optional
from app.db.models.site_hierarchy import SiteHierarchy
from app.schemas.site_hierarchy import SiteHierarchyCreate, SiteHierarchyUpdate
from app.db.models.site_location import SiteLocation


class SiteHierarchyRepository:

    # CREATE
    @staticmethod
    def create(db: Session, payload: SiteHierarchyCreate) -> SiteHierarchy:
        site = SiteHierarchy(
            name=payload.name,
            parent_site_hierarchy_id=payload.parent_site_hierarchy_id,
        )
        db.add(site)
        db.commit()

        # 🔁 CRITICAL: re-fetch with parent eagerly loaded
        return (
            db.query(SiteHierarchy)
            .options(joinedload(SiteHierarchy.parent))
            .filter(SiteHierarchy.id == site.id)
            .one()
        )

    # GET BY ID
    @staticmethod
    def get_by_id(db: Session, site_id: int) -> SiteHierarchy | None:
        return db.get(SiteHierarchy, site_id)

    # DUPLICATION CHECK (THIS WAS MISSING)
    @staticmethod
    def exists_with_name(
        db: Session,
        name: str,
        parent_id: int | None,
        exclude_id: int | None = None,
    ) -> bool:
        stmt = select(SiteHierarchy).where(
            func.lower(SiteHierarchy.name) == name.lower(),
            SiteHierarchy.parent_site_hierarchy_id == parent_id,
        )

        if exclude_id:
            stmt = stmt.where(SiteHierarchy.id != exclude_id)

        return db.execute(stmt).scalars().first() is not None

    # UPDATE
    @staticmethod
    def update(
        db: Session,
        site: SiteHierarchy,
        payload: SiteHierarchyUpdate,
    ) -> SiteHierarchy:

        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(site, field, value)

        db.commit()

        # 🔁 CRITICAL: re-fetch with parent eagerly loaded
        return (
            db.query(SiteHierarchy)
            .options(joinedload(SiteHierarchy.parent))
            .filter(SiteHierarchy.id == site.id)
            .one()
        )

    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list[SiteHierarchy], int]:

        stmt = (
            select(SiteHierarchy)
            .options(joinedload(SiteHierarchy.parent))
            .where(SiteHierarchy.is_active.is_(True))  # ✅ filter active only
        )

        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.where(
                func.lower(SiteHierarchy.name).like(search_term)
            )

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        stmt = stmt.order_by(
            func.lower(SiteHierarchy.name).asc()
        )

        stmt = stmt.offset(page * page_size).limit(page_size)

        sites = db.execute(stmt).scalars().all()
        return sites, total 
    
    @staticmethod
    def list_all(db, search: str = None):
        stmt = select(SiteHierarchy).where(SiteHierarchy.is_active.is_(True))
        if search:
            stmt = stmt.where(func.lower(SiteHierarchy.name).like(f"%{search.lower()}%"))
        return db.execute(stmt).scalars().all()

    # LIST ALL ACTIVE (for dropdowns)
    @staticmethod
    def list_all_active(db: Session) -> List[SiteHierarchy]:
        stmt = (
            select(SiteHierarchy)
            .where(SiteHierarchy.is_active.is_(True))
            .order_by(SiteHierarchy.name.asc())
        )
        return db.execute(stmt).scalars().all()

    # DELETE (HARD)
    @staticmethod
    def delete(db: Session, site: SiteHierarchy) -> None:
        db.delete(site)
        db.commit()

    @staticmethod
    def list_hierarchies_with_locations(db: Session, include_inactive: bool = False, site_hierarchy_id: Optional[int] = None) -> List[SiteHierarchy]:
        """
        Load hierarchies and eager-load their site_locations and each location's cameras.
        Use selectinload to avoid N+1.
        Optionally filter by a specific site_hierarchy_id.
        """
        stmt = select(SiteHierarchy).options(
            selectinload(SiteHierarchy.site_locations).selectinload(SiteLocation.cameras)
        )

        if not include_inactive:
            stmt = stmt.where(SiteHierarchy.is_active.is_(True))

        if site_hierarchy_id is not None:
            stmt = stmt.where(SiteHierarchy.id == site_hierarchy_id)

        stmt = stmt.order_by(SiteHierarchy.id)
        return db.execute(stmt).scalars().all()