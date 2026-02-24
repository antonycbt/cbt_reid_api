from sqlalchemy.orm import Session
from sqlalchemy import select, insert, delete, func
from app.db.models.associations import site_location_access
from sqlalchemy.orm import joinedload, with_loader_criteria
from app.db.models import SiteLocation, AccessGroup,SiteHierarchy
from typing import List, Set
class SiteLocationAccessRepository:

    # -------- CREATE (single) --------
    @staticmethod
    def create(
        db: Session,
        site_location_id: int,
        access_group_id: int,
    ):
        db.execute(
            insert(site_location_access).values(
                site_location_id=site_location_id,
                access_group_id=access_group_id,
            )
        )
        db.commit()

    # -------- EXISTS --------
    @staticmethod
    def exists(
        db: Session,
        site_location_id: int,
        access_group_id: int,
    ) -> bool:
        stmt = select(site_location_access).where(
            site_location_access.c.site_location_id == site_location_id,
            site_location_access.c.access_group_id == access_group_id,
        )
        return db.execute(stmt).first() is not None

    # -------- BULK CREATE --------
    @staticmethod
    def bulk_create_for_access_group(
        db: Session,
        access_group_id: int,
        site_location_ids: List[int],
    ) -> int:

        if not site_location_ids:
            return 0

        existing_stmt = select(site_location_access.c.site_location_id).where(
            site_location_access.c.access_group_id == access_group_id,
            site_location_access.c.site_location_id.in_(site_location_ids),
        )

        existing_ids: Set[int] = {row[0] for row in db.execute(existing_stmt).all()}

        new_rows = [
            {
                "site_location_id": sl_id,
                "access_group_id": access_group_id,
            }
            for sl_id in site_location_ids
            if sl_id not in existing_ids
        ]

        if not new_rows:
            return 0

        db.execute(insert(site_location_access), new_rows)
        db.commit()
        return len(new_rows)

    # -------- LIST --------
    @staticmethod
    def list(db, search: str | None, page: int = 0, page_size: int = 10):

        # Pre-fetch all site hierarchies for ancestor traversal
        all_hierarchies = {s.id: s for s in db.query(SiteHierarchy).all()}
        def all_hierarchy_ancestors_active(site_hierarchy_id: int) -> bool:
            current = all_hierarchies.get(site_hierarchy_id)
            # ✅ check the node itself first
            if current is None or not current.is_active:
                return False
            while current.parent_site_hierarchy_id is not None:
                parent = all_hierarchies.get(current.parent_site_hierarchy_id)
                if parent is None or not parent.is_active:
                    return False
                current = parent
            return True

        # Pre-fetch all access groups for ancestor traversal
        all_access_groups = {ag.id: ag for ag in db.query(AccessGroup).all()}

        def all_access_group_ancestors_active(access_group_id: int) -> bool:
            current = all_access_groups.get(access_group_id)
            while current and current.parent_access_group_id is not None:
                parent = all_access_groups.get(current.parent_access_group_id)
                if parent is None or not parent.is_active:
                    return False
                current = parent
            return True

        query = (
            db.query(SiteLocation)
            .options(
                joinedload(SiteLocation.access_groups),
                joinedload(SiteLocation.site_hierarchy),
            )
            .filter(SiteLocation.is_active.is_(True))
        )

        if search:
            query = query.filter(SiteLocation.name.ilike(f"%{search}%"))

        total = query.count()
        locations = query.offset(page * page_size).limit(page_size).all()

        # Group by access group, filtering out inactive ancestor access groups
        access_group_map: dict[int, dict] = {}

        for loc in locations:
            # skip if location's own hierarchy or any ancestor is inactive
            if not all_hierarchy_ancestors_active(loc.site_hierarchy_id):
                continue

            for ag in loc.access_groups:
                if not ag.is_active or not all_access_group_ancestors_active(ag.id):
                    continue

                if ag.id not in access_group_map:
                    access_group_map[ag.id] = {
                        "access_group_id": ag.id,
                        "access_group_name": ag.name,
                        "site_locations": [],
                    }

                access_group_map[ag.id]["site_locations"].append({
                    "site_location_id": loc.id,
                    "site_location_name": loc.site_hierarchy.name if loc.site_hierarchy else None,
                })

        result = list(access_group_map.values())
        return result, total
    # -------- DELETE --------
    @staticmethod
    def delete(
        db: Session,
        site_location_id: int,
        access_group_id: int,
    ):
        db.execute(
            delete(site_location_access).where(
                site_location_access.c.site_location_id == site_location_id,
                site_location_access.c.access_group_id == access_group_id,
            )
        )
        db.commit()
