from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.session import SessionLocal
from app.repositories.site_hierarchy_repo import SiteHierarchyRepository
from app.schemas.site_hierarchy import SiteHierarchyUpdate
from app.db.models.site_hierarchy import SiteHierarchy
from app.db.models.site_location import SiteLocation
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.activity_helper import (
    snapshot,
    build_create_changes,
    build_update_changes,
    build_delete_changes,
)
from typing import List, Optional, Any
from app.core.constants import TARGET_TYPE

SITE_HIERARCHY_TARGET_TYPE = 3
SITE_HIERARCHY_ENTITY = TARGET_TYPE[SITE_HIERARCHY_TARGET_TYPE]["entity"]
SITE_HIERARCHY_EXCLUDE = {"id"}


def _get_hierarchy_name(db: Session, hierarchy_id: int) -> str | None:
    try:
        row = db.execute(
            text("SELECT name FROM site_hierarchies WHERE id = :id"),
            {"id": hierarchy_id},
        ).mappings().first()
        return row["name"] if row else None
    except Exception:
        return None


def _resolve_site_hierarchy_changes(
    db: Session,
    changes: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Resolve parent_site_hierarchy_id → name at write time."""
    resolved = {}
    for field, (old, new) in changes.items():
        if field == "parent_site_hierarchy_id":
            old_name = _get_hierarchy_name(db, old) if old is not None else None
            new_name = _get_hierarchy_name(db, new) if new is not None else None
            # only include if at least one side resolved
            if old_name is not None or new_name is not None:
                resolved["parent_site_hierarchy"] = [old_name, new_name]
        else:
            resolved[field] = [old, new]
    return resolved


class SiteHierarchyService:

    @staticmethod
    def create_site_hierarchy(db: Session, payload, actor_id: int):
        if SiteHierarchyRepository.exists_with_name(db, payload.name):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Site name already exists",
            )

        site = SiteHierarchyRepository.create(db, payload)

        detail = ActivityDetail(
            action="create",
            entity=SITE_HIERARCHY_ENTITY,
            changes=_resolve_site_hierarchy_changes(
                db,
                build_create_changes(site, exclude=SITE_HIERARCHY_EXCLUDE),
            ),
            meta={
                "actor_id": actor_id,
                "display_name": site.name,
            },
        )
        ActivityLogService.log(
            db=db,
            actor_id=actor_id,
            target_type=SITE_HIERARCHY_TARGET_TYPE,
            target_id=site.id,
            detail=detail,
        )

        db.commit()
        db.refresh(site)
        return site

    @staticmethod
    def update_site_hierarchy(
        db: Session,
        site_hierarchy_id: int,
        payload: SiteHierarchyUpdate,
        actor_id: int,
    ):
        site = SiteHierarchyRepository.get_by_id(db, site_hierarchy_id)
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")

        before = snapshot(site)
        updated_site = SiteHierarchyRepository.update(db, site, payload)

        detail = ActivityDetail(
            action="update",
            entity=SITE_HIERARCHY_ENTITY,
            changes=_resolve_site_hierarchy_changes(
                db,
                build_update_changes(before, updated_site, exclude=SITE_HIERARCHY_EXCLUDE),
            ),
            meta={
                "actor_id": actor_id,
                "display_name": updated_site.name,
            },
        )
        ActivityLogService.log(
            db=db,
            actor_id=actor_id,
            target_type=SITE_HIERARCHY_TARGET_TYPE,
            target_id=site_hierarchy_id,
            detail=detail,
        )

        db.commit()
        db.refresh(updated_site)
        return updated_site

    @staticmethod
    def delete_site_hierarchy(
        db: Session,
        site_hierarchy_id: int,
        actor_id: int,
    ) -> bool:
        site = SiteHierarchyRepository.get_by_id(db, site_hierarchy_id)
        if not site:
            return False

        before = snapshot(site)

        locations = db.query(SiteLocation).filter(
            SiteLocation.site_hierarchy_id == site_hierarchy_id
        ).all()
        for loc in locations:
            db.delete(loc)

        db.delete(site)
        db.flush()

        detail = ActivityDetail(
            action="delete",
            entity=SITE_HIERARCHY_ENTITY,
            changes=_resolve_site_hierarchy_changes(
                db,
                build_delete_changes(before, exclude=SITE_HIERARCHY_EXCLUDE),
            ),
            meta={
                "actor_id": actor_id,
                "display_name": before.get("name"),  # from snapshot before delete
            },
        )
        ActivityLogService.log(
            db=db,
            actor_id=actor_id,
            target_type=SITE_HIERARCHY_TARGET_TYPE,
            target_id=site_hierarchy_id,
            detail=detail,
        )

        db.commit()
        return True

    # ---- unchanged methods below ----

    @staticmethod
    def get_site_hierarchy(site_hierarchy_id: int) -> SiteHierarchy | None:
        db = SessionLocal()
        try:
            return SiteHierarchyRepository.get_by_id(db, site_hierarchy_id)
        finally:
            db.close()

    @staticmethod
    def list_site_hierarchies(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return SiteHierarchyRepository.list(db, search, page, page_size)
        finally:
            db.close()

    @staticmethod
    def get_tree(search: str = None):
        db = SessionLocal()
        try:
            nodes = SiteHierarchyRepository.list_all(db, search)
            return SiteHierarchyService.build_tree(nodes)
        finally:
            db.close()

    @staticmethod
    def build_tree(nodes: list):
        node_map = {node.id: dict(node.__dict__, children=[]) for node in nodes}
        roots = []
        for node in node_map.values():
            pid = node.get("parent_site_hierarchy_id")
            if pid and pid in node_map:
                node_map[pid]["children"].append(node)
            else:
                roots.append(node)
        return roots

    @staticmethod
    def _build_site_location_tree(locations: List[SiteLocation]) -> List[dict]:
        def serialize_loc(loc: SiteLocation) -> dict:
            return {
                "id": loc.id,
                "name": loc.name,
                "site_hierarchy_id": loc.site_hierarchy_id,
                "parent_site_location_id": loc.parent_site_location_id,
                "is_public": loc.is_public,
                "is_active": loc.is_active,
                "cameras": [
                    {
                        "id": cam.id,
                        "name": cam.name,
                        "ip_address": cam.ip_address,
                        "is_active": cam.is_active,
                    }
                    for cam in getattr(loc, "cameras", []) or []
                ],
                "children": []
            }

        node_map = {loc.id: serialize_loc(loc) for loc in locations}
        roots = []
        for node in node_map.values():
            pid = node["parent_site_location_id"]
            if pid and pid in node_map:
                node_map[pid]["children"].append(node)
            else:
                roots.append(node)
        return roots

    @staticmethod
    def _serialize_hierarchy_node(h: SiteHierarchy, loc_tree: List[dict]) -> dict:
        return {
            "id": h.id,
            "name": h.name,
            "parent_site_hierarchy_id": h.parent_site_hierarchy_id,
            "is_active": h.is_active,
            "site_locations": loc_tree,
            "children": []
        }

    @staticmethod
    def build_hierarchy_tree(hierarchies: List[SiteHierarchy]) -> List[dict]:
        node_map = {}
        for h in hierarchies:
            locs = getattr(h, "site_locations", []) or []
            loc_tree = SiteHierarchyService._build_site_location_tree(locs)
            node_map[h.id] = SiteHierarchyService._serialize_hierarchy_node(h, loc_tree)

        roots = []
        for node in node_map.values():
            pid = node["parent_site_hierarchy_id"]
            if pid and pid in node_map:
                node_map[pid]["children"].append(node)
            else:
                roots.append(node)
        return roots

    @staticmethod
    def get_full_hierarchy_tree(
        site_hierarchy_id: Optional[int] = None,
        include_inactive: bool = False,
    ) -> List[dict]:
        db = SessionLocal()
        try:
            hierarchies = SiteHierarchyRepository.list_hierarchies_with_locations(
                db, include_inactive=include_inactive, site_hierarchy_id=site_hierarchy_id
            )
            return SiteHierarchyService.build_hierarchy_tree(hierarchies)
        finally:
            db.close()