from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.db.session import SessionLocal
from app.repositories.site_hierarchy_repo import SiteHierarchyRepository
from app.schemas.site_hierarchy import ( 
    SiteHierarchyUpdate,
)
from app.db.models.site_hierarchy import SiteHierarchy
from typing import List, Optional
from app.db.models.site_location import SiteLocation

class SiteHierarchyService:

    # CREATE site hierarchy
    @staticmethod
    def create_site_hierarchy(payload):
        db = SessionLocal()
        try:
            if SiteHierarchyRepository.exists_with_name(
                db,
                payload.name,
                payload.parent_site_hierarchy_id,
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Site name already exists under this parent"
                )

            return SiteHierarchyRepository.create(db, payload)

        finally:
            db.close()

    @staticmethod 
    def update_site_hierarchy(
        site_hierarchy_id: int,
        payload: SiteHierarchyUpdate,
    ):
        db = SessionLocal()
        try:
            site = SiteHierarchyRepository.get_by_id(db, site_hierarchy_id)
            if not site:
                raise HTTPException(status_code=404, detail="Site not found")

            return SiteHierarchyRepository.update(db, site, payload)

        finally:
            db.close()

    # GET site hierarchy by ID
    @staticmethod
    def get_site_hierarchy(site_hierarchy_id: int) -> SiteHierarchy | None:
        db = SessionLocal()
        try:
            return SiteHierarchyRepository.get_by_id(db, site_hierarchy_id)
        finally:
            db.close()

    # LIST site hierarchies (search + pagination)
    @staticmethod
    def list_site_hierarchies(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return SiteHierarchyRepository.list(
                db, search, page, page_size
            )
        finally:
            db.close() 
            
    @staticmethod
    def get_tree(search: str = None):
        from app.repositories.site_hierarchy_repo import SiteHierarchyRepository
        db = SessionLocal()
        try:
            nodes = SiteHierarchyRepository.list_all(db, search)
            return SiteHierarchyService.build_tree(nodes)
        finally:
            db.close()

    @staticmethod
    def build_tree(nodes: list):
        # Convert to dict for easy child assignment
        node_map = {node.id: dict(node.__dict__, children=[]) for node in nodes}
        roots = []

        for node in node_map.values():
            pid = node.get("parent_site_hierarchy_id")
            if pid and pid in node_map:
                node_map[pid]["children"].append(node)
            else:
                roots.append(node)

        return roots



    # HARD DELETE site hierarchy
    @staticmethod
    def delete_site_hierarchy(site_hierarchy_id: int) -> bool:
        db = SessionLocal()
        try:
            site = SiteHierarchyRepository.get_by_id(
                db, site_hierarchy_id
            )
            if not site:
                return False

            SiteHierarchyRepository.delete(db, site)
            return True

        finally:
            db.close()
    @staticmethod
    def _build_site_location_tree(locations: List[SiteLocation]) -> List[dict]:
        # serialize locations and attach cameras, then nest by parent_site_location_id
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
                        "location_type": cam.location_type,
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
            "children": []  # filled by build_hierarchy_tree
        }

    @staticmethod
    def build_hierarchy_tree(hierarchies: List[SiteHierarchy]) -> List[dict]:
        """
        Build the tree of SiteHierarchy nodes (parent-child). Each hierarchy node
        will include its site_locations as a nested tree (via _build_site_location_tree).
        """
        # first serialize every hierarchy with its site_locations tree
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
    def get_full_hierarchy_tree(site_hierarchy_id: Optional[int] = None, include_inactive: bool = False) -> List[dict]:
        db = SessionLocal()
        try:
            hierarchies = SiteHierarchyRepository.list_hierarchies_with_locations(
                db, include_inactive=include_inactive, site_hierarchy_id=site_hierarchy_id
            )
            return SiteHierarchyService.build_hierarchy_tree(hierarchies)
        finally:
            db.close()

    