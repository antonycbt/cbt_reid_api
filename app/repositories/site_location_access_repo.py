from sqlalchemy.orm import Session
from sqlalchemy import select, insert, delete, func
from app.db.models.associations import site_location_access
from sqlalchemy.orm import joinedload, with_loader_criteria
from app.db.models import SiteLocation, AccessGroup
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
        """
        Returns site locations with their access groups.
        Output format:
        [
            {
                "site_location_id": 1,
                "site_location_name": "Location A",
                "access_groups": [
                    {"id": 1, "name": "Group 1"},
                    {"id": 2, "name": "Group 2"}
                ]
            },
            ...
        ]
        """
        query = (
            db.query(SiteLocation)
            .options(
                joinedload(SiteLocation.access_groups),
                joinedload(SiteLocation.site_hierarchy),  # 👈 add this
                with_loader_criteria(AccessGroup, AccessGroup.is_active.is_(True)),
            )
            .filter(SiteLocation.is_active.is_(True))
        ) 

        if search:
            query = query.filter(SiteLocation.name.ilike(f"%{search}%"))

        total = query.count()
        locations = query.offset(page * page_size).limit(page_size).all()

        # Build the response
        result = []
        for loc in locations:
            result.append({
                "site_location_id": loc.id,
                "site_location_name": loc.site_hierarchy.name if loc.site_hierarchy else None,
                "access_groups": [{"id": ag.id, "name": ag.name} for ag in loc.access_groups]
            })

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
