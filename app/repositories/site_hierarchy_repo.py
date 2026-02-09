from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import select, func
from typing import List, Optional
from app.db.models.site_hierarchy import SiteHierarchy
from app.schemas.site_hierarchy import SiteHierarchyCreate, SiteHierarchyUpdate
from app.db.models.site_location import SiteLocation
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

class SiteHierarchyRepository: 

    # DUPLICATION CHECK
    @staticmethod
    def exists_with_name(
        db: Session,
        name: str,
        exclude_id: int | None = None,
    ) -> bool:
        stmt = select(SiteHierarchy).where(
            func.lower(SiteHierarchy.name) == name.lower(),
        )

        if exclude_id:
            stmt = stmt.where(SiteHierarchy.id != exclude_id)

        return db.execute(stmt).scalars().first() is not None
    
    # CREATE
    @staticmethod
    def create(db: Session, payload: SiteHierarchyCreate) -> SiteHierarchy:
        site = SiteHierarchy(
            name=payload.name,
            parent_site_hierarchy_id=payload.parent_site_hierarchy_id,
        )

        db.add(site)

        try:
            db.flush()  # get ID before commit

            # -------------------------------------------------
            # 1. If parent exists → parent is no longer leaf
            # deactivate parent location
            # -------------------------------------------------
            if payload.parent_site_hierarchy_id:
                parent_location = (
                    db.query(SiteLocation)
                    .filter(
                        SiteLocation.site_hierarchy_id == payload.parent_site_hierarchy_id,
                        SiteLocation.is_active == True
                    )
                    .first()
                )

                if parent_location:
                    parent_location.is_active = False  # soft deactivate

            # -------------------------------------------------
            # 2. New node is leaf → create its location
            # -------------------------------------------------
            existing_location = db.query(SiteLocation).filter(
                SiteLocation.site_hierarchy_id == site.id
            ).first()

            if not existing_location:
                location = SiteLocation(
                    name=site.name,
                    site_hierarchy_id=site.id,
                    is_active=True,
                )
                db.add(location)

            db.commit()

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Site name already exists",
            )

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
    
    def sync_leaf_state(db: Session, hierarchy_id: int):
        # check if node has children
        has_children = db.query(SiteHierarchy.id).filter(
            SiteHierarchy.parent_site_hierarchy_id == hierarchy_id
        ).first() is not None

        location = db.query(SiteLocation).filter(
            SiteLocation.site_hierarchy_id == hierarchy_id
        ).first()

        if has_children:
            # node is NOT leaf → deactivate location
            if location and location.is_active:
                location.is_active = False
        else:
            # node IS leaf → ensure location exists
            if not location:
                db.add(SiteLocation(
                    name="Auto",
                    site_hierarchy_id=hierarchy_id,
                    is_active=True
                ))
            elif not location.is_active:
                location.is_active = True 

    # UPDATE
    @staticmethod
    def update(
        db: Session,
        site: SiteHierarchy,
        payload: SiteHierarchyUpdate,
    ) -> SiteHierarchy:

        old_parent_id = site.parent_site_hierarchy_id

        if SiteHierarchyRepository.exists_with_name(db, payload.name, exclude_id=site.id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Site name already exists",
            )

        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(site, field, value)

        db.flush()

        # ---------------------------------------
        # sync leaf lifecycle
        # ---------------------------------------

        # this node
        SiteHierarchyRepository.sync_leaf_state(db, site.id)

        # old parent may become leaf
        if old_parent_id and old_parent_id != site.parent_site_hierarchy_id:
            SiteHierarchyRepository.sync_leaf_state(db, old_parent_id)

        # new parent loses leaf status
        if site.parent_site_hierarchy_id:
            SiteHierarchyRepository.sync_leaf_state(db, site.parent_site_hierarchy_id)

        db.commit()

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